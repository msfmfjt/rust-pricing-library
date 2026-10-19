"""Four strategies on a small synthetic eSSVI/Local Volatility market history.

Run from the repository root with PYTHONPATH=backtesting. All prices and VegaKT
come from rust_pricing; bid/ask quotes are synthetic model prices plus spreads.
Select diagnostic mode or the numerically accepted P0 terminal replication
scope. Synthetic integration output does not establish hedge effectiveness.
"""

import argparse
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import rust_pricing as rp

from rust_pricing_backtest import (BacktestConfig, Contract, ExternalCashflow, NumeraireObservation, PaymentDiscount, Position, Quote, RiskBasis,
                                  RustPricingProvider, SettlementFixing, Snapshot, compare_strategies)


def build_case(path_count=2048, *, diagnostic_only=False, payment_lag_days=0, numeraire_rate=None, capital_flow_demo=False,
               gamma_relative_bump=None, market_seed=None, numerical_scope=None):
    if not isinstance(payment_lag_days, int) or payment_lag_days < 0:
        raise ValueError("payment_lag_days must be a nonnegative calendar-day count")
    if numeraire_rate is not None and not math.isfinite(numeraire_rate):
        raise ValueError("numeraire_rate must be a finite continuously compounded ACT/365F rate")
    if gamma_relative_bump is not None and (not math.isfinite(gamma_relative_bump) or gamma_relative_bump <= 0):
        raise ValueError("gamma_relative_bump must be positive and finite")
    if market_seed is not None and (type(market_seed) is not int or market_seed < 0):
        raise ValueError("market_seed must be a nonnegative integer")
    # Separate namespace from the fixed inner-pricing seed 7 and the ensemble
    # bootstrap. Outer shocks are illustrative, not fitted historical dynamics.
    market_rng = (np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x56454741, market_seed])))
                  if market_seed is not None else None)
    market_log_spot, market_log_vol = 0., 0.
    start, expiry = date(2026, 9, 7), date(2026, 9, 21)
    contracts = [Contract("target", expiry, 100, lot=.1),
                 Contract("call95", expiry, 95, lot=.1),
                 Contract("call105", expiry, 105, lot=.1),
                 Contract("call100_later", expiry+timedelta(days=7), 100, lot=.1)]
    contracts = [replace(c, settlement_rule="explicit_fixing", settlement_fixing_id="synthetic_close",
                         payment_date=c.expiry + timedelta(days=payment_lag_days) if payment_lag_days else None) for c in contracts]
    if any(c.payment_day.weekday() >= 5 for c in contracts):
        raise ValueError("the synthetic weekday schedule requires weekday payment dates; use e.g. --payment-lag-days 2")
    end = max(c.payment_day for c in contracts) if payment_lag_days else expiry
    sessions = [start + timedelta(days=i) for i in range((end-start).days+1)
                if (start+timedelta(days=i)).weekday() < 5]
    history = []
    provider = RustPricingProvider(diagnostic_only=diagnostic_only, numerical_scope=numerical_scope)
    first_price = None
    k = np.linspace(-.24, .24, 13)
    for i, day in enumerate(sessions):
        at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=16)
        spot = 100 * math.exp(.015 * math.sin(.7*i) + .001*i)
        vol = .22 + .02 * math.sin(.5*i)
        if market_rng is not None:
            if i:
                dt = (day-sessions[i-1]).days/365.
                spot_shock, vol_shock = market_rng.standard_normal(2)
                market_log_spot += -.5*.22**2*dt + .22*math.sqrt(dt)*spot_shock
                market_log_vol += -.5*.5**2*dt + .5*math.sqrt(dt)*vol_shock
            spot, vol = 100*math.exp(market_log_spot), .22*math.exp(market_log_vol)
        max_days = max(2, max((c.expiry-day).days for c in contracts))
        dates = tuple(day+timedelta(days=j) for j in range(1, max_days+1))
        times = np.array([(d-day).days/365 for d in dates])
        slices = [rp.EssviSlice(t, vol*vol*t, .08*t, -.015*t) for t in (7/365, 28/365)]
        basis = RiskBasis(day, dates, tuple(k))
        valuation_curve = rp.DiscountCurve(10, np.array([0., 1.]), np.array([1., math.exp(-.01)]))
        market = rp.Market.equity(2, 1, spot, valuation_curve,
                                 rp.DiscountCurve(11, np.array([0., 1.]), np.array([1., 1.])))
        common_model = rp.Model.local_volatility_from_essvi(slices, vol*vol, np.array([0., *times]), k, 1e-8, 4.)
        # Serialize the common surface sampling through the public API, so no
        # eSSVI derivatives or interpolation are reimplemented in the adapter.
        carrier = rp.PricingRequest(day, rp.Product.european_vanilla(1, 2, dates[-1], 100, 1, "call"),
                                    market, common_model, rp.Engine.pseudo_monte_carlo(7, path_count), rp.RiskRequest())
        common_basis = json.loads(carrier.to_json())["model"]["reporting_iv_basis"]
        requests = {}
        for c in contracts:
            if c.expiry <= day:
                continue
            t = (c.expiry-day).days/365
            model = rp.Model.local_volatility_from_essvi(slices, vol*vol, np.array([0., *times[times <= t]]), k, 1e-8, 4.)
            request = rp.PricingRequest(day, rp.Product.european_vanilla(1, 2, c.expiry, c.strike, 1, c.side), market,
                                        model, rp.Engine.pseudo_monte_carlo(7, path_count, antithetic=True),
                                        rp.RiskRequest(delta=True, vega=True, gamma_relative_bump=gamma_relative_bump,
                                                       vega_kt_maturity_nodes=dates,
                                                       vega_kt_log_forward_moneyness_nodes=k,
                                                       vega_kt_relative_density_threshold=1e-10))
            payload = json.loads(request.to_json())
            payload["model"]["reporting_iv_basis"] = common_basis
            requests[c.contract_id] = json.dumps(payload, allow_nan=False)
        provisional = Snapshot(f"synthetic-{day}", at, Quote(spot-.005, spot+.005, at, at), {},
                               basis, at, (f"synthetic_essvi_outer_diffusion_v1_seed_{market_seed}"
                                           if market_rng is not None else "synthetic_eSSVI_v1_same_model_marks"), requests,
                               lending_rate=.01, borrowing_rate=.03,
                               settlement_fixings={"synthetic_close": SettlementFixing(day, spot, at, at, "synthetic_close", "v1")}
                               if any(c.expiry == day for c in contracts) else {},
                               payment_discounts={payment: PaymentDiscount(day, valuation_curve.discount((payment-day).days/365.),
                                                                           at, at, "synthetic_valuation_curve", "v1")
                                                  for payment in sorted({c.payment_day for c in contracts if c.payment_day >= day})}
                               if payment_lag_days else {},
                               numeraire=NumeraireObservation("synthetic_money_market", 2, day,
                                                              math.exp(numeraire_rate*(day-start).days/365.),
                                                              at, at, "synthetic_constant_rate_act365f", "v1")
                               if numeraire_rate is not None else None)
        quotes = {}
        for c in contracts:
            if c.expiry <= day:
                continue
            value = provider.evaluate(provisional, c).price
            quotes[c.contract_id] = Quote(max(0., value-.02), value+.02, at, at)
            if i == 0 and c.contract_id == "target":
                first_price = value
        history.append(replace(provisional, option_quotes=quotes))
    if capital_flow_demo:
        for index, amount, flow_id in ((2, 100., "synthetic-deposit"), (len(history)-1, -40., "synthetic-withdrawal")):
            s = history[index]
            history[index] = replace(s, external_cashflows=[ExternalCashflow(
                flow_id, s.at, amount, 2, s.at, "synthetic_investor_capital", "v1")])
    return history, contracts, [Position("target", -10, first_price)], provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/vega-kt-backtest"))
    parser.add_argument("--paths", type=int, default=2048)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostic-only", action="store_true")
    mode.add_argument("--validated-p0", action="store_true")
    parser.add_argument("--gross-notional-limit", type=float, default=None,
                        help="enable joint decision-time cash/gross constraints; requires constraints extra")
    parser.add_argument("--cash-floor", type=float, default=-1_000_000.)
    parser.add_argument("--payment-lag-days", type=int, default=0,
                        help="calendar-day payment lag; dates must fit this synthetic weekday schedule")
    parser.add_argument("--numeraire-rate", type=float, default=None,
                        help="explicit common money-market rate (continuous ACT/365F); enables discounted P&L")
    parser.add_argument("--capital-flow-demo", action="store_true",
                        help="book capital +100 on the third session and -40 on the final session")
    parser.add_argument("--gamma-relative-bump", type=float, default=None,
                        help="request native spot Gamma; 0.0001 matches this adapter's existing Delta bump, other values also change Delta")
    args = parser.parse_args()
    history, contracts, portfolio, provider = build_case(args.paths, diagnostic_only=args.diagnostic_only,
                                                        payment_lag_days=args.payment_lag_days,
                                                        numeraire_rate=args.numeraire_rate, capital_flow_demo=args.capital_flow_demo,
                                                        gamma_relative_bump=args.gamma_relative_bump,
                                                        numerical_scope="p0_terminal_replication" if args.validated_p0 else None)
    config = BacktestConfig(ridge=1e-4, solver_tolerance=1e-8, option_fee=.01, stock_fee=.001,
                            cash_floor=args.cash_floor,
                            constraint_mode="decision_cash_gross" if args.gross_notional_limit is not None else "execution_cash",
                            gross_notional_limit=args.gross_notional_limit,
                            max_price_standard_error=2., max_abs_projection_residual=1000., max_abs_edge_sensitivity=1000.,
                            numeraire_id="synthetic_money_market" if args.numeraire_rate is not None else None)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    results = compare_strategies(config, history, contracts, portfolio, provider, code_revision=revision)
    summaries = {}
    for strategy, result in results.items():
        result.write(args.output / strategy)
        summaries[strategy] = result.summary
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
