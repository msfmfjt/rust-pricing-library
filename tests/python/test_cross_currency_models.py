"""Foreign-calibrated equity processes in settlement-currency pricing."""
import math
import unittest

import rust_pricing as rp

from test_rough_volatility_families import models as rough_models


TODAY, MID, END, PAY = "2026-01-01", "2026-07-01", "2027-01-01", "2027-01-15"
SETTLEMENTS = ("quanto", "floating_fx", "composite")


def models(stochastic):
    eta = .3 if stochastic else 0.
    local = rp.Model.local_volatility_from_grid(
        [0., .5, 1.], [-1., 0., 1.], ([.06, .04, .03] if stochastic else [.04]*3)*3, 1e-9, 2.)
    particles = rp.LsvParticleConfig(particle_count=64, calibration_seed=981,
                                     log_bandwidth=.5, minimum_effective_samples=2.)
    yield "BS", rp.Model.black_scholes(.2)
    yield "Black76", rp.Model.black_76(.2)
    yield "LV", local
    yield "Bass", rp.Model.bass_local_volatility(
        [.5, 1.], [-2., 0., 2.], [.2]*6, [-2+4*i/800 for i in range(801)],
        config=rp.BassLvConfig(grid_points=401, cdf_tolerance=1e-7))
    factors = [
        rp.VolatilityFactorSpec.bergomi(1.2, eta, -.4),
        rp.VolatilityFactorSpec.bergomi_two_factor([.5, 2.], eta, .35, [-.4, -.2], .2),
        rp.VolatilityFactorSpec.rough_bergomi(rp.RoughBergomiModel(.15, eta, equity_vol_correlation=-.4)),
    ]
    for name, factor in zip(("Bergomi", "Bergomi2", "roughBergomi"), factors):
        yield name, rp.EquityModelSpec.stochastic_volatility(.2, factor)
        yield name+" LSV", rp.EquityModelSpec.local_stochastic_volatility(local, factor, particles)
    for name, rough in zip(("roughHeston","liftedHeston","quadraticHeston","mixedBergomi","roughSABR","RFSV"),
                           rough_models(stochastic)):
        yield name, rp.EquityModelSpec.rough_volatility(rough)
        yield name+" LSV", rp.EquityModelSpec.rough_local_stochastic_volatility(local, rough, particles)


class CrossCurrencyModelTests(unittest.TestCase):
    def setUp(self):
        def curve(i,r):
            return rp.DiscountCurve(i,[0.,1.],[1.,math.exp(-r)])
        self.usd,self.jpy=curve(1,.04),curve(2,.01)
        self.market=rp.Market.equity(1,10,100.,self.usd,curve(3,.02))

    def price(self,product,model,settlement,**kwargs):
        return rp.CrossCurrencyEquityOption(TODAY,product,self.market,self.jpy,
            settlement=settlement,equity_model=model,fx_spot=150.,fx_volatility=.15,
            correlation=-.45,fixed_fx_rate=145. if settlement=="quanto" else None,
            maximum_step=.25,**kwargs)

    def test_model_matrix_against_constant_variance_formulas(self):
        engine=rp.Engine.randomized_quasi_monte_carlo(256, 86, scramble_count=8, antithetic=True, brownian_bridge=True)
        for settlement in SETTLEMENTS:
            product=rp.Product.european_vanilla(10,2,END,15000. if settlement=="composite" else 100.,1.,"call")
            oracle=self.price(product,rp.Model.black_scholes(.2),settlement,payment_date=PAY).analytical_price()
            for name, model in models(False):
                with self.subTest(settlement=settlement,model=name):
                    result=self.price(product,model,settlement,payment_date=PAY).price_mc(engine)
                    self.assertAlmostEqual(result.value,oracle,delta=6*result.standard_error + (6. if name=="Bass" else .01))

    def test_stochastic_autocall_in_all_settlements(self):
        engine=rp.Engine.pseudo_monte_carlo(9817,128,antithetic=True)
        for mode in SETTLEMENTS:
            unit=150. if mode=="composite" else 1.
            contract=rp.ContractBuilder("coupons",2,10,"close").autocall(
                100.*unit,100.*unit,70.*unit,[MID,END],["2026-07-15",PAY],
                [102.*unit,110.*unit],[85.*unit,85.*unit],[.03,.04])
            graph,=contract.to_graphs(TODAY,[10])
            product=rp.Product.from_graph(graph)
            for name,model in models(True):
                with self.subTest(settlement=mode,model=name):
                    note=self.price(product,model,mode)
                    if mode=="floating_fx":
                        note=note.with_payment_fx_fixings([(PAY,END)])
                    est=note.price_mc(engine)
                    self.assertGreater(est.value,10000.)
                    self.assertGreater(est.standard_error,0.)
                    self.assertLess(est.value,17000.)

    def test_foreign_native_prices_with_live_volatility(self):
        engine=rp.Engine.pseudo_monte_carlo(619, 1024, antithetic=True)
        foreign_call=rp.Product.european_vanilla(10,1,END,100.,1.,"call")
        domestic_call=rp.Product.european_vanilla(10,2,END,100.,1.,"call")
        for name,model in models(True):
            with self.subTest(model=name):
                standard=isinstance(model,rp.Model)
                source=rp.EquityModelSpec.standard(model) if standard else model
                # The standard/LSV native entry points choose their own grids.
                step=None if standard or name.endswith(" LSV") else .25
                native=rp.PricingSpecification(TODAY,foreign_call,self.market,rp.PricingModelSpec(source),
                    engine,rp.RiskRequest(),rp.PricingNumerics(worker_threads=1,maximum_step=step)).compile().evaluate()
                result=self.price(domestic_call,model,"floating_fx").price_mc(engine)
                if step is not None:
                    self.assertAlmostEqual(result.value/150.,native.value,places=9)
                    self.assertAlmostEqual(result.standard_error/150.,native.standard_error,places=9)
                else:
                    self.assertAlmostEqual(result.value/150.,native.value,
                        delta=5*math.hypot(native.standard_error,result.standard_error/150.))
