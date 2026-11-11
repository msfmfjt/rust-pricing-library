"""Trade currency, sign, contract size and independent fixed-portfolio quote risk."""
from math import exp, hypot
from pathlib import Path
import sys
import unittest

import rust_pricing as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
import cross_currency_autocall_quote_risk as example
from cross_currency_stochastic_dividend_autocall import years, REVALUATION
from cross_currency_usd_eur_autocall import curve, JPY, USD
from test_cross_currency_quote_risk import black


class QuoteHedgeTests(unittest.TestCase):
    def test_four_hedges_use_local_quotes_and_current_fx(self):
        engine = rp.Engine.randomized_quasi_monte_carlo(512, 45233, scramble_count=4,
                                                         antithetic=True, brownian_bridge=True)
        sizes = {10: [1., 100.], 20: [50., 1.]}
        hedge = example.seasoned_basket().bergomi_quote_hedge(
            example.QUOTES, engine, contract_sizes=sizes, whole_contracts=True, replications=4, worker_threads=2)
        before, theoretical, after = hedge.unhedged, hedge.theoretical_hedged, hedge.hedged
        for stock, fx, rate in [(10, 149., .04), (20, 171., .025)]:
            for j, (expiry, side, k, f, _) in enumerate(example.QUOTES[stock]):
                t = years(REVALUATION, expiry)
                iv = before.implied_volatilities[stock][j]
                premium_change = black(f,k,t,iv+.005,side=="call") - black(f,k,t,iv-.005,side=="call")
                unit = fx*sizes[stock][j]*exp(-rate*t)*premium_change
                self.assertAlmostEqual(hedge.unit_vegas[stock][j],unit,places=8)
                raw, held = before.vegas[stock][j], hedge.quantities[stock][j]
                ideal = hedge.theoretical_quantities[stock][j]
                self.assertAlmostEqual(ideal, -raw.value/unit, places=8)
                self.assertEqual(held, round(ideal))
                self.assertAlmostEqual(hedge.quantity_standard_errors[stock][j], raw.standard_error/unit, places=8)
                residual = theoretical.vegas[stock][j]
                actual = after.vegas[stock][j]
                self.assertEqual(theoretical.states[stock][j].value, after.states[stock][j].value)
                self.assertAlmostEqual(actual.value-residual.value, (held-ideal)*unit,
                                       delta=5*hypot(actual.standard_error,residual.standard_error)+.5)
                self.assertGreater(residual.standard_error, 0.)
                self.assertEqual(residual.effective_sampling_units, 4)
                # Across repeated sizing and verification, both estimators contribute.
                self.assertLess(abs(residual.value), 5*hypot(residual.standard_error,raw.standard_error)+.5)
                self.assertLess(abs(residual.value), abs(raw.value))

    def test_signed_composite_note_and_lot_rounding(self):
        # A JPY call on 2*S at strike 200 when USD/JPY is a constant 2.
        # One such note is exactly one USD call on S with strike 100.
        dates = ["2026-07-10","2027-04-10"]
        market = rp.Market.equity(USD,10,100.,curve(USD,0.),curve(4,0.))
        factor = rp.VolatilityFactorSpec.bergomi_two_factor([2.4,.4],.6,.4,[-.45,-.2],.25)
        model = rp.EquityModelSpec.stochastic_volatility(.2,factor)
        dividends = rp.BuehlerDividendModel(mean_reversion=.8,equity_linkage=.6,
                                             dividend_volatility=.3,equity_dividend_correlation=.35)
        def call(date):
            g = rp.SourceGraphBuilder()
            payoff = g.maximum(g.subtract(g.spot(10,date),g.literal(200.)),g.literal(0.))
            leg = g.finish([payoff],[date],currency_id=JPY,underlying_ids=[10])
            portfolio = rp.CrossCurrencyEquityBasket.from_legs(
                REVALUATION,[leg],[market],curve(JPY,0.),settlement_currency_id=JPY,equity_models=[model],
                dividend_models=[dividends],fx_quotes=[(USD,2.,0.)],correlations=[[1.,0.],[0.,1.]],
                performance="composite",maximum_step=1/8)
            return portfolio.with_dividend_state(10,model_age=.3,volatility_factor=[0.,0.],equity_factor=1.,dividend_factor=1.)
        engine = rp.Engine.randomized_quasi_monte_carlo(512,61284,scramble_count=4,antithetic=True,brownian_bridge=True)
        quotes = {10:[(date,"call",100.,100.,call(date).price_mc(engine).value/2.) for date in dates]}
        note = call(dates[0])
        def hedge(n):
            return note.bergomi_quote_hedge(quotes,engine,note_quantity=n,contract_sizes={10:[2.,50.]},
                                            whole_contracts=True,replications=4,worker_threads=2)
        result = hedge(-2.)
        self.assertEqual(result.quantities[10],(1.,0.))
        self.assertAlmostEqual(result.theoretical_quantities[10][0],1.,delta=.1)
        self.assertGreater(result.theoretical_hedged.vegas[10][0].standard_error,0.)
        for risk in result.hedged.vegas[10]:
            self.assertAlmostEqual(risk.value,0.,places=10)
            self.assertAlmostEqual(risk.standard_error,0.,places=10)
        # Rounding a small long position to zero leaves its entire risk in place.
        small = hedge(.2)
        self.assertEqual(small.quantities[10], (0.,0.))
        self.assertLess(abs(small.theoretical_hedged.vegas[10][0].value), .25*small.hedged.vegas[10][0].value)
        # Verify the held portfolio using a separate public call on its disjoint seeds.
        seed = (61284 + 12*0x9e3779b97f4a7c15) % 2**64
        independent = rp.Engine.randomized_quasi_monte_carlo(512,seed,scramble_count=4,antithetic=True,brownian_bridge=True)
        control = note.bergomi_quote_risk(quotes,independent,replications=4,worker_threads=2)
        for actual, expected in zip(small.hedged.vegas[10],control.vegas[10]):
            self.assertAlmostEqual(actual.value,.2*expected.value,places=10)
            self.assertAlmostEqual(actual.standard_error,.2*expected.standard_error,places=10)
        for kwargs in [dict(note_quantity=float("inf")),dict(contract_sizes={20:[1.,1.]}),
                       dict(contract_sizes={10:[0.,1.]}),dict(contract_sizes={10:[1.,float("nan")]})]:
            with self.assertRaises(ValueError):
                note.bergomi_quote_hedge(quotes,engine,**kwargs)
