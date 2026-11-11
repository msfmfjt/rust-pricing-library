"""Premium calibration, independently repriced market vegas and sampling errors."""
from math import erf, exp, log, sqrt
from pathlib import Path
import sys
import unittest

import rust_pricing as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
import cross_currency_autocall_quote_risk as example
import cross_currency_stochastic_dividend_autocall as hybrid


def black(f, k, t, vol, is_call):
    sd = vol * sqrt(t)
    d1 = log(f/k)/sd + sd/2
    n = lambda z: (1+erf(z/sqrt(2)))/2
    return f*n(d1)-k*n(d1-sd) if is_call else k*n(sd-d1)-f*n(-d1)


class QuoteRiskTests(unittest.TestCase):
    def test_two_stock_note_recovers_independent_local_prices(self):
        mc = rp.Engine.randomized_quasi_monte_carlo(512, 45233, scramble_count=4,
                                                    antithetic=True, brownian_bridge=True)
        report = example.seasoned_basket().bergomi_quote_risk(example.QUOTES,mc,replications=4,worker_threads=2)
        self.assertEqual(set(report.vegas), {10,20})
        self.assertGreater(report.present_value.value, 140000.)
        self.assertGreater(report.present_value.standard_error, 0.)
        for i, stock in enumerate([10,20]):
            for j, estimate in enumerate(report.states[stock]):
                # Fixture prices are independent simulations rounded to a cent.
                self.assertAlmostEqual(estimate.value,hybrid.APRIL_TWO_FACTOR[i][j],
                                       delta=5*estimate.standard_error+.025)
                self.assertGreater(estimate.standard_error,0.)
            for j, row in enumerate(example.QUOTES[stock]):
                maturity, side, strike, forward, premium = row
                t=hybrid.years(hybrid.REVALUATION,maturity)
                iv=report.implied_volatilities[stock][j]
                self.assertAlmostEqual(exp(-[.04,.025][i]*t)*black(forward,strike,t,iv,side=="call"),premium,places=8)
                residual=report.price_residuals[stock][j]
                self.assertAlmostEqual(residual.value,0.,delta=5*residual.standard_error+.01)
                risk=report.vegas[stock][j]
                self.assertGreater(risk.standard_error,0.)
                self.assertEqual(risk.effective_sampling_units,4)

    def test_replicating_option_vega_and_fixed_cash_after_call(self):
        from cross_currency_usd_eur_autocall import curve, USD, JPY
        # A single stock with a stochastic cash factor and zero rates. A fixed
        # FX rate of 150 and zero FX volatility turn a USD option into 150 times
        # its local premium. This gives an independent Black quote-vega oracle.
        short, long = "2026-07-10", "2027-04-10"
        zero = curve(1,0.)
        market=rp.Market.equity(USD,10,100.,zero,curve(4,0.),discrete_dividends=[
            rp.DividendEvent.fixed_cash(1,.5,2.)])
        model=rp.EquityModelSpec.stochastic_volatility(.2,
              rp.VolatilityFactorSpec.bergomi_two_factor([2.4,.4],.6,.4,[-.45,-.2],.25))
        div=rp.BuehlerDividendModel(mean_reversion=.8,equity_linkage=.6,
                                    dividend_volatility=.3,equity_dividend_correlation=.35)
        def basket(expiry, strike, option=True):
            g=rp.SourceGraphBuilder()
            if option:
                payoff=g.maximum(g.subtract(g.spot(10,expiry),g.literal(strike)),g.literal(0.))
            else:
                payoff=g.literal(20.)
            graph=g.finish([payoff],[expiry],currency_id=USD,underlying_ids=[10])
            legs=[graph]
            if not option:
                p=rp.SourceGraphBuilder()
                legs.append(p.finish([p.literal(150000.)],[expiry],currency_id=JPY,underlying_ids=[10]))
            b=rp.CrossCurrencyEquityBasket.from_legs(hybrid.REVALUATION,legs,[market],curve(JPY,0.),
                settlement_currency_id=JPY,equity_models=[model],dividend_models=[div],
                fx_quotes=[(USD,150.,0.)],correlations=[[1.,0.],[0.,1.]],maximum_step=1/12)
            if not option:
                b=b.with_leg_fx(0,schedule=[(expiry,"2026-04-07")],known_fixings=[("2026-04-07",147.)])
            return b.with_dividend_state(10,model_age=.3,volatility_factor=(0.,0.),
                                         equity_factor=.98,dividend_factor=1.1)
        mc=rp.Engine.randomized_quasi_monte_carlo(512,1984,scramble_count=4,antithetic=True,brownian_bridge=True)
        quotes={10:[(date,"call",k,f,basket(date,k).price_mc(mc).value/150.)
                    for date,k,f in [(short,100.,100.),(long,98.,98.)]]}
        engine=rp.Engine.randomized_quasi_monte_carlo(512,19276,scramble_count=4,antithetic=True,brownian_bridge=True)
        risk=basket(short,100.).bergomi_quote_risk(quotes,engine,replications=4,worker_threads=2)
        iv=risk.implied_volatilities[10][0]
        t=hybrid.years(hybrid.REVALUATION,short)
        analytic=150.*(black(100.,100.,t,iv+.005,True)-black(100.,100.,t,iv-.005,True))
        near,far=risk.vegas[10]
        self.assertAlmostEqual(near.value,analytic,delta=5*near.standard_error+.15)
        self.assertAlmostEqual(far.value,0.,delta=5*far.standard_error+.15)
        # Reconstruct the complete-replication mean and SE from two disjoint
        # blocks. This includes the uncertainty of refitting both option prices.
        half_reports=[]
        for first_run in (0,2):
            seed=(19276 + 3*first_run*0x9e3779b97f4a7c15) % (2**64)
            subset=rp.Engine.randomized_quasi_monte_carlo(512,seed,scramble_count=4,
                                                         antithetic=True,brownian_bridge=True)
            half_reports.append(basket(short,100.).bergomi_quote_risk(quotes,subset,replications=2,worker_threads=2))
        for get in [lambda r: r.present_value, lambda r: r.vegas[10][0], lambda r: r.vegas[10][1],
                    lambda r: r.states[10][1], lambda r: r.price_residuals[10][0]]:
            a,b=map(get,half_reports)
            mean=(a.value+b.value)/2
            sse=2*(a.standard_error**2+b.standard_error**2+(a.value-mean)**2+(b.value-mean)**2)
            self.assertAlmostEqual(get(risk).value,mean,places=10)
            self.assertAlmostEqual(get(risk).standard_error,sqrt(sse/12),places=10)
        fixed=basket("2026-04-14",0.,False).bergomi_quote_risk(quotes,engine,replications=2)
        self.assertEqual(fixed.present_value.value,150000.+147.*20.)
        self.assertEqual([(v.value,v.standard_error) for v in fixed.vegas[10]],[(0.,0.),(0.,0.)])

    def test_invalid_quotes_and_unidentified_states(self):
        b=example.seasoned_basket()
        e=rp.Engine.pseudo_monte_carlo(16,123)
        short,long=example.QUOTES[10]
        for quotes in [{},{10:[short]},{10:[long,short]},{10:[short,short]},
                       {10:[(*short[:-1],0.),long]},{10:[(*short[:-1],200.),long]},{90:[short,long]}]:
            with self.subTest(quotes=quotes), self.assertRaises(ValueError):
                b.bergomi_quote_risk(quotes,e)
        for opts in [dict(bump=0.),dict(bump=.5),dict(bump=1e-18),dict(replications=1)]:
            with self.assertRaises(ValueError):
                b.bergomi_quote_risk({10:[short,long]},e,**opts)
        for unsupported in [hybrid.note(hybrid.REVALUATION),hybrid.note(hybrid.VALUATION,two_factor=True),
                            hybrid.note(hybrid.REVALUATION,two_factor=True,vol_of_vol=(0.,.7))]:
            with self.assertRaises(ValueError):
                unsupported.bergomi_quote_risk({10:[short,long]},e)
