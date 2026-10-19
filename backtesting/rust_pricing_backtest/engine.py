"""Next-snapshot execution, self-financing ledger and four-strategy comparison."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from typing import Mapping, Sequence

import numpy as np

from .domain import (BacktestConfig, BacktestError, Contract, PathContract, Position, Risk, RiskProvider, Snapshot,
                     canonical, checked_sum, data, fingerprint, require)
from .optimizer import round_lot, solve_hedge
from .pnl import POLICY as PNL_POLICY, attribute_pnl
from .numeraire import POLICY as NUMERAIRE_POLICY, discounted_attribution, discounted_ledger, discounted_summary
from . import budget, cost_basis, greeks, iv_alignment
from .market_shocks import MarketShockMap, REPLAY_POLICY as MARKET_SHOCK_POLICY
from . import path_history
from . import joint_replay


_TABLES = ("ledger", "orders", "fills", "cashflows", "risks", "decisions", "settlements", "pnl_components", "cost_basis", "greek_pnl", "iv_alignment")


@dataclass(frozen=True)
class BacktestResult:
    status: str
    manifest: dict
    ledger: tuple[dict, ...]
    orders: tuple[dict, ...]
    fills: tuple[dict, ...]
    cashflows: tuple[dict, ...]
    risks: tuple[dict, ...]
    decisions: tuple[dict, ...]
    summary: dict
    checkpoint: dict
    failure: dict | None = None
    settlements: tuple[dict, ...] = ()
    pnl_components: tuple[dict, ...] = ()
    cost_basis: tuple[dict, ...] = ()
    greek_pnl: tuple[dict, ...] = ()
    iv_alignment: tuple[dict, ...] = ()

    def to_json(self) -> str:
        return canonical(self) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> BacktestResult:
        value = json.loads(payload)
        version = value["manifest"]["schema_version"]
        require(version in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), "schema_version", "unsupported result version")
        if version >= 2:
            require("failure" in value and (value["failure"] is not None) == (value["status"] == "failed"),
                    "invalid_result", "failed results require failure details")
        if version < 3:
            value.setdefault("settlements", [])
        if version < 4:
            value.setdefault("pnl_components", [])
        if version < 7:
            value.setdefault("cost_basis", [])
        if version < 8:
            value.setdefault("greek_pnl", [])
        if version < 9:
            value.setdefault("iv_alignment", [])
        for key in _TABLES:
            value[key] = tuple(value[key])
        return cls(**value)

    def write(self, directory: str | Path) -> None:
        from .serialization import input_json
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "result.json").write_text(self.to_json(), encoding="utf-8")
        (path / "input.json").write_text(input_json(self), encoding="utf-8")
        (path / "checkpoint.json").write_text(canonical(self.checkpoint) + "\n", encoding="utf-8")
        for key in _TABLES:
            rows = getattr(self, key)
            with (path / f"{key}.csv").open("w", newline="", encoding="utf-8") as stream:
                names = sorted({k for row in rows for k in row})
                writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: canonical(v) if isinstance(v, (dict, tuple, list)) else v
                                     for k, v in row.items()})


class BacktestRunError(BacktestError):
    """An expected replay failure carrying the last committed state and report."""

    def __init__(self, error: BacktestError, result: BacktestResult):
        super().__init__(error.code, error.message)
        self.result = result


def _savepoint(state):
    # Tables are append-only except for the single pending order being filled.
    # Retain table lengths, not copies of the entire growing execution history.
    values = deepcopy({k: v for k, v in state.items() if k not in _TABLES})
    lengths = {k: len(state[k]) for k in _TABLES}
    pending_order = deepcopy(state["orders"][state["pending"]]) if state["pending"] is not None else None
    return values, lengths, pending_order


def _rollback(state, saved):
    values, lengths, pending_order = saved
    state.update(values)
    for key, length in lengths.items():
        del state[key][length:]
    if pending_order is not None:
        state["orders"][state["pending"]] = pending_order


def _summary(rows: list[dict], fills: list[dict]) -> dict:
    pnl = [r["daily_pnl"] for r in rows]
    high, drawdown = 0.0, 0.0
    for r in rows:
        high = max(high, r["cumulative_pnl"])
        drawdown = max(drawdown, high - r["cumulative_pnl"])
    final = rows[-1] if rows else None
    return {"cumulative_pnl": final["cumulative_pnl"] if final else None,
            "daily_mean": statistics.fmean(pnl) if pnl else None,
            "daily_standard_deviation": statistics.stdev(pnl) if len(pnl) > 1 else None,
            "daily_rmse": math.sqrt(statistics.fmean(x*x for x in pnl)) if pnl else None,
            "worst_day": min(pnl) if pnl else None, "max_drawdown_currency": drawdown if rows else None,
            "execution_cost": math.fsum(f["execution_cost"] for f in fills),
            "absolute_traded_value": math.fsum(abs(f["quantity"] * f["multiplier"] * f["price"]) for f in fills),
            "terminal_hedging_error": final["cumulative_pnl"] if final and final["fully_settled"] else None,
            "fully_settled": final["fully_settled"] if final else None, "observations": len(rows)}


def run_backtest(config: BacktestConfig, history: Sequence[Snapshot],
                 contracts: Sequence[Contract], portfolio: Sequence[Position],
                 provider: RiskProvider, *, checkpoint: Mapping | None = None,
                 stop_after: int | None = None, code_revision: str = "unrecorded",
                 on_error: str = "raise") -> BacktestResult:
    """Run an explicit daily schedule. Invalid market/risk data fails closed.

    A checkpoint resumes against the same schedule and unchanged processed prefix.
    Funding uses the previous snapshot's rates and actual calendar-day fractions.
    Investor capital books at snapshots after carry and before settlements/fills;
    it is removed from nominal and discounted P&L at its own booking time.
    Cash-settled European payoffs fix on expiry and pay on payment_day. Missing
    expiry/payment observations are rejected. Each contract specifies its fixing.
    Expected replay errors raise BacktestRunError with a failed result attached;
    on_error="return" returns that result. Each snapshot commits atomically.
    Invalid initial inputs/checkpoints and unexpected programming errors raise
    without a result. Neither mode continues after a failed snapshot.
    """
    require(on_error in ("raise", "return"), "invalid_error_policy", on_error)
    history, contracts, portfolio = tuple(history), tuple(contracts), tuple(portfolio)
    require(bool(history) and bool(portfolio), "empty_input", "history and portfolio are required")
    require(all(a.at < b.at and a.at.date() < b.at.date() for a, b in zip(history, history[1:])),
            "invalid_schedule", "one increasing timestamp per valuation date is required")
    require(len({s.snapshot_id for s in history}) == len(history), "duplicate_snapshot", "snapshot IDs must be unique")
    master = {c.contract_id: c for c in contracts}
    require(len(master) == len(contracts), "duplicate_contract", "contract IDs must be unique")
    require(len({(c.underlying_id, c.currency_id) for c in contracts}) == 1,
            "unsupported_portfolio", "single underlying and currency required")
    currency_id = contracts[0].currency_id
    require(all(p.contract_id in master for p in portfolio), "unknown_contract", "portfolio contract missing")
    require(all(master[p.contract_id].listed_on <= history[0].at.date() < master[p.contract_id].expiry for p in portfolio),
            "invalid_entry", "target must be listed and unexpired at entry")
    require(history[0].dividend_per_share == 0, "ambiguous_dividend", "initial snapshot must be post-event")
    if stop_after is not None:
        require(isinstance(stop_after, int) and 0 < stop_after <= len(history), "invalid_stop", "invalid snapshot count")
    constrained = config.constraint_mode == "decision_cash_gross"
    market_shocks_enabled = config.market_shock_log_moneyness is not None
    paths_enabled = any(isinstance(c,PathContract) for c in contracts)
    joint_enabled = config.max_joint_bucket_standard_error is not None
    require(not joint_enabled or callable(getattr(provider, 'joint_market_risk', None)),
            'joint_provider_scope', 'provider must supply paired joint market risk')
    checkpoint_version = 11 if joint_enabled else 10 if paths_enabled else 9 if market_shocks_enabled else 8
    config_data = data(config)
    if not joint_enabled:
        config_data.pop('max_joint_bucket_standard_error')
    if not market_shocks_enabled and not paths_enabled and not joint_enabled:
        # Keep the existing request/result contract when the feature is omitted.
        config_data.pop("market_shock_log_moneyness")
    optimizer_backend = {"policy": budget.POLICY if constrained else "box_l1_coordinate_descent_v1",
                         "scipy": budget.backend()[0].__version__ if constrained else None,
                         "numpy": np.__version__}
    identity = {"config": config_data, "contracts": data(sorted(contracts, key=lambda c: c.contract_id)),
                "optimizer_backend": optimizer_backend,
                "portfolio": data(portfolio), "provider": provider.identity, "code_revision": code_revision,
                "risk_validation": data(getattr(provider, "validation", {"status": "not_supplied"})),
                "implementation_hash": fingerprint({p.name: p.read_text(encoding="utf-8")
                                                     for p in sorted(Path(__file__).parent.glob("*.py"))}),
                "schedule": [s.at.isoformat() for s in history]}
    if market_shocks_enabled:
        identity["market_shock_policy"] = dict(MARKET_SHOCK_POLICY)
    if paths_enabled:
        identity["path_observation_policy"] = dict(path_history.POLICY)
    if joint_enabled:
        identity['joint_market_policy'] = dict(joint_replay.POLICY)
    identity_hash = fingerprint(identity)
    if checkpoint is None:
        state = {"next_index": 0, "cash": config.initial_cash, "stock": 0.0,
                 "targets": {}, "hedges": {}, "receivables": {}, "pending": None, "external_cashflow_ids": {},
                 "cost_basis_book": cost_basis.new_book(),
                 **{key: [] for key in _TABLES}}
        if paths_enabled:
            state["path_fixings"] = {}
        for p in portfolio:
            c = master[p.contract_id]
            state["targets"][c.contract_id] = state["targets"].get(c.contract_id, 0.0) + p.quantity
            cost_basis.trade(state["cost_basis_book"], "targets", c.contract_id, p.quantity, p.entry_price, c.multiplier)
            cash = -p.quantity * c.multiplier * p.entry_price - p.entry_fee
            state["cash"] += cash
            state["cashflows"].append({"at": history[0].at.isoformat(), "kind": "target_entry",
                                       "contract_id": c.contract_id, "amount": cash})
        require(state["cash"] >= config.cash_floor, "cash_limit", "initial premium exceeds cash budget")
    else:
        require(checkpoint.get("schema_version") == checkpoint_version and checkpoint.get("identity_hash") == identity_hash,
                "checkpoint_mismatch", "configuration, provider or schedule changed")
        require(fingerprint(checkpoint["state"]) == checkpoint.get("state_hash"),
                "checkpoint_mismatch", "checkpoint state is corrupted")
        state = json.loads(canonical(checkpoint["state"]))
        prefix = state["next_index"]
        require(isinstance(prefix, int) and 0 <= prefix <= len(history)
                and fingerprint(history[:prefix]) == checkpoint["history_prefix_hash"],
                "checkpoint_mismatch", "processed market history changed")
        require(stop_after is None or stop_after > prefix, "invalid_stop", "stop must follow checkpoint")
        if joint_enabled:
            joint_replay.verify_checkpoint(state, history)
    target_contracts = [master[p.contract_id] for p in portfolio]

    def cashflow(at, kind, amount, contract_id="", settlement=None):
        require(math.isfinite(amount), "nonfinite_cashflow", kind)
        state["cash"] += amount
        state["cashflows"].append({"at": at, "kind": kind, "contract_id": contract_id, "amount": amount})
        if settlement is not None:
            state["cashflows"][-1]["settlement"] = settlement

    def quote(s, cid):
        q = s.spot if cid == "__spot__" else s.option_quotes.get(cid)
        require(q is not None, "missing_quote", f"{s.snapshot_id}: {cid}")
        q.check(s.at, config.max_quote_age_seconds)
        return q

    def gross(s, hedges, stock):
        options = checked_sum((s.spot.mid * abs(quantity) * master[cid].multiplier
                            for group in (state["targets"], hedges) for cid, quantity in group.items()
                            if master[cid].expiry > s.at.date()), "nonfinite_budget", "option gross")
        return checked_sum((options, s.spot.mid*abs(stock)), "nonfinite_budget", "portfolio gross")

    def budget_inputs(s, ids, target_delta, stock_lower, stock_upper, risk):
        coefficients = [budget.execution_coefficients(quote(s, cid), master[cid].multiplier,
                                                      config.option_fee, config.slippage_bps) for cid in ids]
        coefficients.append(budget.execution_coefficients(quote(s, "__spot__"), 1., config.stock_fee, config.slippage_bps))
        return {"deltas": [master[cid].multiplier*risk(cid).delta for cid in ids],
                "target_delta": target_delta, "stock_previous": state["stock"],
                "stock_lower": stock_lower, "stock_upper": stock_upper,
                "linear": [pair[0] for pair in coefficients], "spreads": [pair[1] for pair in coefficients],
                "gross_weights": [s.spot.mid*master[cid].multiplier for cid in ids]+[s.spot.mid],
                "cash": state["cash"], "cash_floor": config.cash_floor, "fixed_gross": gross(s, {}, 0.),
                "gross_limit": config.gross_notional_limit}

    def settlement(s, contract, quantity):
        if isinstance(contract, PathContract):
            return path_history.settlement(contract,s,quantity)
        detail = {"rule": contract.settlement_rule, "quantity": quantity,
                  "multiplier": contract.multiplier, "expiry": contract.expiry.isoformat(),
                  "payment_date": contract.payment_day.isoformat()}
        if contract.settlement_rule == "explicit_fixing":
            fixing = s.settlement_fixings.get(contract.settlement_fixing_id)
            require(fixing is not None, "missing_settlement_fixing",
                    f"{s.snapshot_id}: {contract.contract_id}: {contract.settlement_fixing_id}")
            try:
                fixing.check_expiry(contract.expiry, s.at)
            except BacktestError as error:
                raise BacktestError(error.code, f"{s.snapshot_id}: {contract.contract_id}: {error.message}") from error
            price = fixing.value
            detail.update({"fixing_id": contract.settlement_fixing_id, "fixing": data(fixing),
                           "fixing_hash": fingerprint(fixing)})
        else:
            price = s.spot.mid
            detail.update({"snapshot_id": s.snapshot_id, "source": s.source, "spot_quote": data(s.spot)})
        payoff = contract.payoff(price)
        detail.update({"underlying_price": price, "payoff_per_unit": payoff})
        amount = quantity * contract.multiplier * payoff
        require(math.isfinite(amount), "nonfinite_settlement", contract.contract_id)
        return amount, detail

    def execute(s, desired, stock, order, liquidation=False):
        # Build the entire basket and check cash before committing any fill.
        rows = []
        for cid in sorted(set(desired) | set(state["hedges"])) + ["__spot__"]:
            old = state["stock"] if cid == "__spot__" else state["hedges"].get(cid, 0.0)
            new = stock if cid == "__spot__" else desired.get(cid, old)
            if cid != "__spot__" and master[cid].expiry <= s.at.date():
                continue
            change = new - old
            if abs(change) <= 1e-12:
                continue
            q = quote(s, cid)
            m = 1.0 if cid == "__spot__" else master[cid].multiplier
            fee = abs(change) * (config.stock_fee if cid == "__spot__" else config.option_fee)
            price = (q.ask if change > 0 else q.bid) * (1 + math.copysign(config.slippage_bps / 10_000, change))
            rows.append({"at": s.at.isoformat(), "decision_at": order["decision_at"], "contract_id": cid,
                         "quantity": change, "multiplier": m, "price": price, "mid": q.mid, "fee": fee,
                         "cash_change": -change * m * price - fee,
                         "execution_cost": change * m * (price - q.mid) + fee,
                         "liquidation": liquidation})
        proposed_cash = state["cash"] + math.fsum(r["cash_change"] for r in rows)
        require(math.isfinite(proposed_cash) and all(math.isfinite(r["execution_cost"]) for r in rows),
                "nonfinite_execution", "basket arithmetic overflowed")
        if constrained:
            proposed_hedges, proposed_stock = dict(state["hedges"]), state["stock"]
            for row in rows:
                if row["contract_id"] == "__spot__":
                    proposed_stock += row["quantity"]
                else:
                    cid = row["contract_id"]
                    proposed_hedges[cid] = proposed_hedges.get(cid, 0.)+row["quantity"]
            proposed_gross = gross(s, proposed_hedges, proposed_stock)
            require(math.isfinite(proposed_gross), "nonfinite_budget", "execution gross")
            order["execution_budget"] = {"cash_after": proposed_cash, "gross_notional": proposed_gross}
        if proposed_cash < config.cash_floor:
            require(not liquidation, "cash_limit", "terminal liquidation exceeds cash budget")
            order["status"] = "rejected_cash_limit"
            order["fill_at"] = s.at.isoformat()
            return
        if constrained:
            if not budget.feasible(order["execution_budget"], config.cash_floor, config.gross_notional_limit):
                require(not liquidation, "gross_limit", "terminal liquidation exceeds gross budget")
                order["status"] = "rejected_gross_limit"
                order["fill_at"] = s.at.isoformat()
                return
        for row in rows:
            cid = row["contract_id"]
            cashflow(row["at"], "hedge_trade", row["cash_change"], cid)
            cost_basis.trade(state["cost_basis_book"], "stock" if cid == "__spot__" else "hedges",
                             cid, row["quantity"], row["price"], row["multiplier"], mid=row["mid"])
            if cid == "__spot__":
                state["stock"] += row["quantity"]
            else:
                state["hedges"][cid] = state["hedges"].get(cid, 0.0) + row["quantity"]
            state["fills"].append(row)
        state["hedges"] = {k: v for k, v in state["hedges"].items() if abs(v) > 1e-12}
        order["status"] = "filled"
        order["fill_at"] = s.at.isoformat()

    def process_snapshot(index):
        s = history[index]
        at = s.at.isoformat()
        fill_start = len(state["fills"])
        cashflow_start = len(state["cashflows"]) if index else 0
        quote(s, "__spot__")
        require(s.surface_available_at <= s.at, "future_surface", s.snapshot_id)
        # Compile only the current snapshot. A future grid cannot influence an
        # earlier order or prevent a resumable prefix from being committed.
        shocks = (MarketShockMap(s.basis, replace(s.basis, log_moneyness=config.market_shock_log_moneyness))
                  if market_shocks_enabled else None)
        # Validate all supplied fixings before exposing this snapshot to a provider.
        for fixing_id, fixing in s.settlement_fixings.items():
            try:
                fixing.check_available(s.at)
            except BacktestError as error:
                raise BacktestError(error.code, f"{s.snapshot_id}: {fixing_id}: {error.message}") from error
        if paths_enabled:
            s = path_history.resolve(s,contracts,state["path_fixings"])
        for payment_day, discount in s.payment_discounts.items():
            discount.check(s.at, payment_day, config.max_quote_age_seconds)
        require(config.numeraire_id is None or s.numeraire is not None,
                "missing_numeraire", f"{s.snapshot_id}: {config.numeraire_id}")
        if s.numeraire is not None:
            s.numeraire.check(s.at, config.numeraire_id, currency_id, config.max_quote_age_seconds)
        external_flows = sorted(s.external_cashflows, key=lambda f: f.flow_id)
        require(len({f.flow_id for f in external_flows}) == len(external_flows),
                "duplicate_external_cashflow", f"{s.snapshot_id}: repeated capital ID in snapshot")
        for flow in external_flows:
            flow.check(s.at, currency_id)
            require(flow.flow_id not in state["external_cashflow_ids"],
                    "duplicate_external_cashflow", f"{s.snapshot_id}: {flow.flow_id} was already booked")
        external_net = checked_sum((f.amount for f in external_flows), "nonfinite_external_cashflow", s.snapshot_id)
        cumulative_external = checked_sum((state["ledger"][-1]["cumulative_external_net_cashflow"] if index else 0.0,
                                           external_net), "nonfinite_external_cashflow", s.snapshot_id)
        if index:
            previous = history[index - 1]
            dt = (s.at.date() - previous.at.date()).days / 365.0
            rate = previous.lending_rate if state["cash"] >= 0 else previous.borrowing_rate
            cashflow(at, "funding", state["cash"] * math.expm1(rate * dt))
            cashflow(at, "stock_borrow", -max(-state["stock"], 0) * previous.spot.mid * previous.stock_borrow_rate * dt)
            cashflow(at, "dividend", state["stock"] * s.dividend_per_share)
        if external_flows:
            # Simultaneous capital movements net as one batch; preserve every event for audit.
            proposed_cash = checked_sum((state["cash"], external_net), "nonfinite_external_cashflow", s.snapshot_id)
            for flow in external_flows:
                value = s.numeraire.value if config.numeraire_id is not None else None
                converted = flow.amount / value if value is not None else None
                require(converted is None or math.isfinite(converted), "nonfinite_discounted_pnl", flow.flow_id)
                flow_hash = fingerprint(flow)
                state["cashflows"].append({"at": at, "kind": "external_capital", "contract_id": "", "amount": flow.amount,
                                           "flow_id": flow.flow_id, "external_cashflow": data(flow),
                                           "external_cashflow_hash": flow_hash, "numeraire_value": value,
                                           "numeraire_hash": fingerprint(s.numeraire) if value is not None else None,
                                           "discounted_amount": converted})
                state["external_cashflow_ids"][flow.flow_id] = flow_hash
            state["cash"] = proposed_cash
        for group in ("targets", "hedges"):
            for cid, quantity in list(state[group].items()):
                c = master[cid]
                if not quantity:
                    del state[group][cid]
                elif c.expiry <= s.at.date():
                    require(c.expiry == s.at.date(), "missing_settlement", cid)
                    amount, detail = settlement(s, c, quantity)
                    claim_id = f"{group}:{cid}"
                    claim = {"claim_id": claim_id, "group": group, "contract_id": cid,
                             "fixed_at": at, "payment_date": c.payment_day.isoformat(),
                             "amount": amount, "settlement": detail}
                    cost_basis.fix(state["cost_basis_book"], group, cid, quantity, amount, claim_id,
                                   keep_claim=bool(amount) or c.payment_day == s.at.date())
                    state["settlements"].append({**claim, "at": at, "event": "fixed" if amount else "fixed_zero"})
                    if amount or c.payment_day == s.at.date():
                        state["receivables"][claim_id] = claim
                    del state[group][cid]
        for claim_id, claim in sorted(state["receivables"].copy().items(),
                                      key=lambda item: (item[1]["group"] != "targets", item[1]["contract_id"])):
            if claim["payment_date"] <= s.at.date().isoformat():
                require(claim["payment_date"] == s.at.date().isoformat(), "missing_payment_snapshot", claim_id)
                cashflow(at, f"{claim['group']}_settlement", claim["amount"], claim["contract_id"], settlement=claim["settlement"])
                cost_basis.pay(state["cost_basis_book"], claim_id, claim["amount"])
                state["settlements"].append({**claim, "at": at, "event": "paid"})
                del state["receivables"][claim_id]
        require(state["cash"] >= config.cash_floor, "cash_limit", "carry/capital/settlement exceeds cash budget")
        if state["pending"] is not None:
            order = state["orders"][state["pending"]]
            expired = [cid for cid in order["desired"] if master[cid].expiry <= s.at.date()]
            order["expired_contracts"] = expired
            execute(s, order["desired"], order["stock"], order)
            state["pending"] = None
        final = index == len(history) - 1
        if final and config.end_policy == "liquidate_hedges":
            order = {"decision_at": at, "status": "pending", "desired": {}, "stock": 0.0,
                     "kind": "precommitted_terminal_liquidation"}
            state["orders"].append(order)
            execute(s, {cid: 0.0 for cid in state["hedges"]}, 0.0, order, True)

        if constrained:
            held_budget = {"cash_after": state["cash"], "gross_notional": gross(s, state["hedges"], state["stock"])}
            require(math.isfinite(held_budget["gross_notional"]), "nonfinite_budget", "held gross")
            require(budget.feasible(held_budget, config.cash_floor, config.gross_notional_limit),
                    "gross_limit", "held gross exceeds cap after fills; snapshot rolled back")

        risk_cache: dict[str, Risk] = {}

        def risk(cid):
            if cid not in risk_cache:
                r = provider.evaluate(s, master[cid])
                c = master[cid]
                require(r.payment_date == c.payment_day or (r.payment_date is None and c.payment_day == c.expiry),
                        "payment_risk_mismatch", f"{cid}: provider must price the contractual payment date")
                require(r.basis_id == s.basis.basis_id and len(r.buckets) == s.basis.size,
                        "basis_mismatch", f"{s.snapshot_id}: {cid}")
                require(r.reporting_forwards is None or len(r.reporting_forwards) == len(s.basis.maturity_dates),
                        "iv_basis_mismatch", f"{s.snapshot_id}: {cid}")
                require(abs(r.signed_residual) <= config.max_abs_projection_residual
                        and r.edge_sensitivity_abs <= config.max_abs_edge_sensitivity,
                        "risk_quality", f"projection/edge quality: {cid}")
                require(r.price_standard_error is None or r.price_standard_error <= config.max_price_standard_error,
                        "risk_quality", f"price standard error: {cid}")
                require(r.bucket_standard_errors is None or max(r.bucket_standard_errors) <= config.max_bucket_standard_error,
                        "risk_quality", f"bucket standard error: {cid}")
                risk_cache[cid] = r
            return risk_cache[cid]

        v = np.zeros(s.basis.size)
        delta, target_value, target_residual_abs = 0.0, 0.0, 0.0
        for cid, quantity in sorted(state["targets"].items()):
            r = risk(cid)
            units = quantity * master[cid].multiplier
            mark = r.price if config.target_mark == "model" else quote(s, cid).mid
            target_value += units * mark
            delta += units * r.delta
            v += units * np.asarray(r.buckets)
            target_residual_abs += abs(units * r.signed_residual)
        hedge_value, actual_delta = 0.0, delta + state["stock"]
        actual_v = v.copy()
        for cid, quantity in sorted(state["hedges"].items()):
            r = risk(cid)
            units = quantity * master[cid].multiplier
            hedge_value += units * quote(s, cid).mid
            actual_delta += units * r.delta
            actual_v += units * np.asarray(r.buckets)
        receivable_marks = {}
        for claim_id, claim in sorted(state["receivables"].items()):
            payment_day = master[claim["contract_id"]].payment_day
            discount = s.payment_discounts.get(payment_day)
            require(discount is not None, "missing_payment_discount", f"{s.snapshot_id}: {claim_id}: {payment_day}")
            present_value = claim["amount"] * discount.factor
            require(math.isfinite(present_value), "nonfinite_receivable", claim_id)
            receivable_marks[claim_id] = {**claim, "discount": data(discount), "discount_hash": fingerprint(discount),
                                          "present_value": present_value}
        receivable_value = math.fsum(mark["present_value"] for mark in receivable_marks.values())
        equity = state["cash"] + target_value + hedge_value + state["stock"] * s.spot.mid + receivable_value
        require(math.isfinite(equity), "nonfinite_equity", s.snapshot_id)
        prior_equity = state["ledger"][-1]["equity"] if state["ledger"] else config.initial_cash
        state["ledger"].append({"at": at, "cash": state["cash"], "stock": state["stock"],
                                "target_value": target_value, "hedge_value": hedge_value,
                                "stock_value": state["stock"] * s.spot.mid, "hedges": dict(state["hedges"]),
                                "receivable_value": receivable_value, "receivables": receivable_marks,
                                "equity": equity, "daily_pnl": equity - prior_equity - external_net,
                                "cumulative_pnl": equity - config.initial_cash - cumulative_external,
                                "external_net_cashflow": external_net, "cumulative_external_net_cashflow": cumulative_external,
                                "delta": actual_delta, "raw_vega_l2": float(np.linalg.norm(actual_v)),
                                "fully_settled": not state["targets"] and not state["hedges"]
                                and not state["receivables"] and abs(state["stock"]) < 1e-12})
        state["ledger"][-1].update(discounted_ledger(
            state["ledger"][-1], state["ledger"][-2] if index else None,
            s.numeraire if config.numeraire_id is not None else None, initial_cash=config.initial_cash,
            initial_value=history[0].numeraire.value if config.numeraire_id is not None else None,
            external_cashflows=external_flows))
        state["pnl_components"].append(attribute_pnl(
            state["ledger"][-1], state["ledger"][-2] if index else None,
            state["fills"][fill_start:], state["cashflows"][cashflow_start:],
            initial_cash=config.initial_cash,
            entry_premium=-math.fsum(p.quantity * master[p.contract_id].multiplier * p.entry_price
                                     for p in portfolio) if not index else 0.0,
            entry_fees=math.fsum(p.entry_fee for p in portfolio) if not index else 0.0,
            previous_attribution=state["pnl_components"][-1] if index else None))
        state["pnl_components"][-1]["discounted"] = discounted_attribution(
            state["ledger"][-1], state["ledger"][-2] if index else None,
            state["pnl_components"][-1], state["pnl_components"][-2] if index else None,
            initial_cash=config.initial_cash)
        state["cost_basis"].append(cost_basis.report(
            state["cost_basis_book"], state["ledger"][-1], state["pnl_components"][-1],
            state["cost_basis"][-1] if index else None,
            {"targets": state["targets"], "hedges": state["hedges"], "stock": {"__spot__": state["stock"]}}))
        state["greek_pnl"].append(greeks.explain(
            s, history[index-1] if index else None,
            state["cost_basis"][-2] if index else None, state["risks"][-1] if index else None,
            state["pnl_components"][-1], state["ledger"][-1], state["greek_pnl"][-1] if index else None))
        state["risks"].append({"at": at, "basis": data(s.basis), "basis_id": s.basis.basis_id,
                               "target_buckets": v.tolist(), "held_buckets": actual_v.tolist(),
                               "target_delta": delta, "held_delta": actual_delta,
                               "target_projection_residual_abs": target_residual_abs})
        if shocks is not None:
            state["risks"][-1]["market_shocks"] = {**shocks.identity(),
                "target": shocks.risk_audit(v), "held": shocks.risk_audit(actual_v)}

        def joint_gate(desired=None):
            if not joint_enabled:
                return None
            row = state['risks'][-1]
            if 'joint_market' not in row:
                report = joint_replay.capture(provider, s, master, risk_cache)
                held = joint_replay.gate(report, s, master, state['targets'], state['hedges'],
                                         config.max_joint_bucket_standard_error)
                row['joint_market'] = dict(report=report, held=held)
            if desired is not None:
                return joint_replay.gate(row['joint_market']['report'], s, master, state['targets'], desired,
                                         config.max_joint_bucket_standard_error)

        if not final and index % config.rebalance_every == 0:
            desired = dict(state["hedges"])
            diagnostic = {"at": at, "strategy": config.strategy, "candidates": []}
            stock_lower = max(-config.stock_position_limit if config.allow_stock_shorts else 0,
                              state["stock"]-config.stock_trade_limit)
            stock_upper = min(config.stock_position_limit, state["stock"]+config.stock_trade_limit)
            ids, inputs = [], None
            if config.strategy in ("scalar_vega", "vega_kt"):
                eligible, tradeable = [], []
                for cid, c in sorted(master.items()):
                    if not (c.listed_on <= s.at.date() < c.expiry):
                        continue
                    if not config.allow_identical_hedge and any(path_history.payoff_identity(c)
                                                               == path_history.payoff_identity(t)
                                                               for t in target_contracts):
                        continue
                    q = s.option_quotes.get(cid)
                    if q is None or q.available_at > s.at or (s.at - q.observed_at).total_seconds() > config.max_quote_age_seconds:
                        continue
                    if q.mid <= 0 or (q.ask - q.bid) / q.mid > config.max_relative_spread:
                        continue
                    tradeable.append(cid)
                    r = risk(cid)
                    size = (abs(math.fsum(r.buckets)) if config.strategy == "scalar_vega"
                            else float(np.linalg.norm(shocks.project_risk(r.buckets) if shocks is not None else r.buckets)))
                    if c.multiplier * size >= config.min_vega:
                        eligible.append(cid)
                if config.strategy == "scalar_vega":
                    # Deterministic nearest-strike rule, using only current prices.
                    eligible = sorted(eligible, key=lambda cid: (abs(math.log(master[cid].strike / s.spot.mid)),
                                                                master[cid].expiry, cid))[:1]
                ids = sorted(set(eligible) | set(state["hedges"]))
                a = np.column_stack([np.asarray(risk(cid).buckets) * master[cid].multiplier for cid in ids]) if ids else np.zeros((len(v), 0))
                prev = np.array([state["hedges"].get(cid, 0.0) for cid in ids])
                lower, upper, costs = [], [], []
                for cid, old in zip(ids, prev):
                    if config.strategy == "scalar_vega" and cid not in eligible and cid in tradeable:
                        close = round_lot(min(abs(old), config.option_trade_limit), master[cid].lot)
                        lo = hi = old - math.copysign(close, old)
                    elif cid not in eligible:
                        lo = hi = old
                    else:
                        lo = max(-config.option_position_limit if config.allow_option_shorts else 0, old - config.option_trade_limit)
                        hi = min(config.option_position_limit, old + config.option_trade_limit)
                    lower.append(lo)
                    upper.append(hi)
                    q = quote(s, cid)
                    costs.append((q.ask - q.bid) / 2 * master[cid].multiplier + config.option_fee
                                 + q.mid * master[cid].multiplier * config.slippage_bps / 10_000)
                target = v if config.strategy == "vega_kt" else np.array([math.fsum(v)])
                matrix = a if config.strategy == "vega_kt" else np.sum(a, axis=0, keepdims=True)
                if shocks is not None and config.strategy == "vega_kt":
                    target, matrix = shocks.project_risk(v), shocks.project_hedges(a)
                if constrained:
                    inputs = budget_inputs(s, ids, delta, stock_lower, stock_upper, risk)
                solution = (budget.solve_budget if constrained else solve_hedge)(
                                       target, matrix, prev, lower, upper, costs, **(inputs or {}), ridge=config.ridge,
                                       turnover_penalty=config.turnover_penalty, tolerance=config.solver_tolerance,
                                       max_iterations=config.solver_iterations)
                for j, cid in enumerate(ids):
                    h = lower[j] if lower[j] == upper[j] else round_lot(solution.quantities[j], master[cid].lot, lower[j], upper[j])
                    require(lower[j] - 1e-10 <= h <= upper[j] + 1e-10, "lot_infeasible", cid)
                    desired[cid] = h
                diagnostic.update({"candidates": ids, "eligible": eligible, "solver": data(solution),
                                   "matrix": a.tolist(), "target": v.tolist(),
                                   "rounded_residual": (v + a @ np.array([desired[cid] for cid in ids])).tolist()})
                if shocks is not None:
                    diagnostic["market_shocks"] = {**shocks.identity(),
                        "objective": "fixed_market_iv" if config.strategy == "vega_kt" else "strategy_default",
                        "optimizer_input": {"target": target.tolist(), "matrix": matrix.tolist()},
                        "continuous": shocks.risk_audit(v+a@np.asarray(solution.quantities))}
                if constrained:
                    diagnostic["option_constraints"] = {"previous": prev.tolist(), "lower": lower, "upper": upper, "costs": costs}
            elif constrained:
                inputs = budget_inputs(s, [], delta, stock_lower, stock_upper, risk)
                if config.strategy == "delta":
                    solution = budget.solve_budget(v, np.zeros((len(v), 0)), [], [], [], [], **inputs,
                                                   tolerance=config.solver_tolerance)
                    diagnostic["solver"] = data(solution)
            n = state["stock"]
            if config.strategy != "unhedged":
                n = -(delta + math.fsum(h * master[cid].multiplier * risk(cid).delta for cid, h in desired.items() if h))
                lo = max(-config.stock_position_limit if config.allow_stock_shorts else 0,
                         state["stock"] - config.stock_trade_limit)
                hi = min(config.stock_position_limit, state["stock"] + config.stock_trade_limit)
                n = round_lot(min(hi, max(lo, n)), config.stock_lot, lo, hi)
                require(lo - 1e-10 <= n <= hi + 1e-10, "lot_infeasible", "stock")
            if constrained:
                metrics = budget.estimate([*(desired[cid] for cid in ids), n],
                    [*(state["hedges"].get(cid, 0.) for cid in ids), state["stock"]],
                    inputs["linear"], inputs["spreads"], inputs["gross_weights"],
                    cash=state["cash"], fixed_gross=inputs["fixed_gross"])
                require(budget.feasible(metrics, config.cash_floor, config.gross_notional_limit),
                        "rounded_budget_infeasible", "lot-rounded basket exceeds decision cash or gross budget")
                diagnostic["budget"] = {"inputs": inputs, "rounded": metrics,
                    "rounded_delta": delta+math.fsum(desired[cid]*master[cid].multiplier*risk(cid).delta for cid in ids)+n}
            order = {"decision_at": at, "desired": desired, "stock": n, "status": "pending", "kind": "rebalance"}
            if shocks is not None:
                proposed = v.copy()
                for cid, quantity in sorted(desired.items()):
                    if quantity:
                        proposed += quantity*master[cid].multiplier*np.asarray(risk(cid).buckets)
                audit = diagnostic.setdefault("market_shocks", {**shocks.identity(), "objective": "strategy_default"})
                audit["proposed"] = shocks.risk_audit(proposed)
                audit["proposed_reporting_buckets"] = proposed.tolist()
            if joint_enabled:
                diagnostic['joint_market'] = joint_gate(desired)
            state["orders"].append(order)
            state["pending"] = len(state["orders"]) - 1
            state["decisions"].append(diagnostic)
        state["risks"][-1]["instruments"] = {cid: data(r) for cid, r in sorted(risk_cache.items())}
        if joint_enabled:
            joint_gate()
            state['greek_pnl'][-1]['market_iv'] = joint_replay.explain(
                s, history[index-1] if index else None, state['risks'][-1], state['risks'][-2] if index else None,
                state['cost_basis'][-2] if index else None, state['pnl_components'][-1], state['ledger'][-1],
                state['greek_pnl'][-2]['market_iv'] if index else None)
        state["iv_alignment"].append(iv_alignment.audit(
            state["risks"][-1], state["risks"][-2] if index else None, state["cost_basis"][-2] if index else None))
        state["next_index"] = index + 1

    limit = stop_after if stop_after is not None else len(history)
    failure, replay_error = None, None
    for index in range(state["next_index"], limit):
        savepoint = _savepoint(state)
        try:
            process_snapshot(index)
        except BacktestError as error:
            _rollback(state, savepoint)
            replay_error = error
            failure = {"code": error.code, "message": error.message,
                       "snapshot_id": history[index].snapshot_id, "snapshot_index": index,
                       "attempted_at": history[index].at.isoformat(),
                       "last_completed_at": state["ledger"][-1]["at"] if state["ledger"] else None,
                       "state_boundary": "before_failed_snapshot",
                       "remaining_state": deepcopy({k: state[k] for k in ("cash", "stock", "targets", "hedges", "receivables")}),
                       "pending_order": deepcopy(state["orders"][state["pending"]]) if state["pending"] is not None else None}
            break

    count = state["next_index"]
    status = "failed" if failure else "completed" if count == len(history) else "partial"
    manifest = {"document_kind": "hedging_backtest_result", "schema_version": 13 if joint_enabled else 12 if paths_enabled else 11 if market_shocks_enabled else 10, **identity,
                "input_schema_version": 9 if joint_enabled else 8 if paths_enabled else 7 if market_shocks_enabled else 6, "settlement_policy": "fix_then_discounted_claim_then_pay_v1",
                "snapshot_spot_settlement_contracts": sorted(c.contract_id for c in contracts if c.settlement_rule == "snapshot_spot"),
                "identity_hash": identity_hash, "history_hash": fingerprint(history[:count]),
                "market_history": data(history), "full_history_hash": fingerprint(history),
                "processed_snapshots": count, "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "execution": "next_snapshot_bid_ask",
                "lot_rounding": "eight_ulp_integer_snap_then_toward_zero_with_bounds_v1",
                "funding": "previous_rate_act365f_continuous", "failure_policy": "fail_run",
                "snapshot_commit_policy": "atomic_snapshot_v1",
                "pnl_attribution": dict(PNL_POLICY),
                "cost_basis_policy": dict(cost_basis.POLICY),
                "greek_pnl_policy": dict(greeks.POLICY),
                "iv_alignment_policy": dict(iv_alignment.POLICY),
                "external_cashflow_policy": {"timing": "snapshot_after_carry_before_settlement_and_fills_v1",
                                             "simultaneous_flows": "net_batch_sorted_by_id", "currency_id": currency_id,
                                             "id_uniqueness": "whole_run", "initial_entry": "before_first_snapshot_capital",
                                             "input_hash": fingerprint([(s.at, s.external_cashflows) for s in history])},
                "numeraire_policy": {**NUMERAIRE_POLICY, "account_id": config.numeraire_id,
                                     "currency_id": currency_id, "enabled": config.numeraire_id is not None,
                                     "input_hash": fingerprint([(s.at, s.numeraire) for s in history])
                                     if config.numeraire_id is not None else None},
                "target_mark": config.target_mark, "risk_unit": "currency_per_unit_absolute_volatility"}
    saved = {"schema_version": checkpoint_version, "identity_hash": identity_hash,
             "history_prefix_hash": fingerprint(history[:count]), "state_hash": fingerprint(state), "state": data(state)}
    summary = _summary(state["ledger"], state["fills"])
    external = [f for f in state["cashflows"] if f["kind"] == "external_capital"]
    summary["external_capital"] = {"count": len(external), "net_cashflow": math.fsum(f["amount"] for f in external),
                                   "gross_inflows": math.fsum(f["amount"] for f in external if f["amount"] > 0),
                                   "gross_outflows": -math.fsum(f["amount"] for f in external if f["amount"] < 0)}
    summary["target_entry_fees"] = math.fsum(p.entry_fee for p in portfolio)
    summary["funding_pnl"] = math.fsum(f["amount"] for f in state["cashflows"] if f["kind"] == "funding")
    summary["stock_borrow_cost"] = -math.fsum(f["amount"] for f in state["cashflows"] if f["kind"] == "stock_borrow")
    attribution = state["pnl_components"]
    summary["discounted_pnl"] = discounted_summary(state["ledger"], attribution, status)
    summary["pnl_components"] = dict(attribution[-1]["cumulative"]) if attribution else None
    basis_rows = state["cost_basis"]
    summary["cost_basis"] = (
        {key: deepcopy(basis_rows[-1][key])
         for key in ("realized_pnl", "unrealized_pnl", "total_pnl", "groups", "cash_realized_components")}
        if basis_rows else None
    )
    explanations = state["greek_pnl"]
    intervals = [r for r in explanations if r["from_at"] is not None]
    summary["greek_pnl"] = {
        "scope": greeks.POLICY["scope"], "observations": len(explanations), "intervals": len(intervals),
        "intervals_with_complete_gamma": sum(all(g["gamma_status"] == "available" for g in r["groups"].values())
                                              for r in intervals),
        "vega_status": "not_computed_risk_and_delta_iv_convention_unvalidated",
        "cumulative": deepcopy(explanations[-1]["cumulative"]) if explanations else None,
        "max_abs_reconciliation_difference": max(abs(r["reconciliation_error"]) for r in explanations) if explanations else None,
        "cumulative_reconciliation_difference": explanations[-1]["cumulative_reconciliation_error"] if explanations else None}
    alignment_rows = state["iv_alignment"]
    summary["iv_alignment"] = {
        "observations": len(alignment_rows), "intervals": max(0, len(alignment_rows)-1),
        "intervals_with_all_nodes_matched": sum(r["status"] == "complete" for r in alignment_rows),
        "application": iv_alignment.POLICY["application"], "pnl_adopted": False,
        "last_status": alignment_rows[-1]["status"] if alignment_rows else None}
    summary["pnl_reconciliation"] = {
        "observations": len(attribution), "passed": True if attribution else None,
        "max_abs_daily_difference": max(abs(r["reconciliation_error"]) for r in attribution) if attribution else None,
        "cumulative_difference": attribution[-1]["cumulative_reconciliation_error"] if attribution else None}
    summary["risk_validation_status"] = identity["risk_validation"]["status"]
    if joint_enabled:
        market_rows = [row['market_iv'] for row in explanations]
        summary['joint_market'] = dict(held_checks=len(state['risks']), proposed_checks=len(state['decisions']),
            limit=config.max_joint_bucket_standard_error, intervals=max(0, len(market_rows)-1),
            covered_intervals=sum(r['status']=='available' for r in market_rows),
            pnl_status='complete' if market_rows and all(r['status'] in ('available','no_interval') for r in market_rows) else 'partial',
            max_abs_reconciliation_error=max((abs(r['reconciliation_error']) for r in market_rows), default=None),
            cumulative=deepcopy(market_rows[-1]['cumulative']) if market_rows else None)
    summary.update({"status": status, "scheduled_snapshots": len(history),
                    "coverage_fraction": count / len(history),
                    "outstanding_payments": len(state["receivables"]),
                    "receivable_value": state["ledger"][-1]["receivable_value"] if state["ledger"] else None,
                    "last_completed_at": state["ledger"][-1]["at"] if state["ledger"] else None})
    if status != "completed":
        summary["terminal_hedging_error"] = None
    result = BacktestResult(status, manifest, **{key: tuple(state[key]) for key in _TABLES},
                            summary=summary, checkpoint=saved, failure=failure)
    if replay_error is not None and on_error == "raise":
        raise BacktestRunError(replay_error, result) from replay_error
    return result


def compare_strategies(config, history, contracts, portfolio, provider, **kwargs):
    return {strategy: run_backtest(replace(config, strategy=strategy), history, contracts, portfolio, provider, **kwargs)
            for strategy in ("unhedged", "delta", "scalar_vega", "vega_kt")}
