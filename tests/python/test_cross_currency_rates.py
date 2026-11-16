"""Independent joint-lognormal benchmarks for two correlated Hull-White rates."""
import math
import unittest
from datetime import date

import rust_pricing as rp

from test_cross_currency_models import models


TODAY, MID, END, PAY = "2026-01-01", "2026-07-01", "2027-01-01", "2027-03-01"
C = [[1., -.45, .2, .3], [-.45, 1., .35, -.4],
     [.2, .35, 1., .25], [.3, -.4, .25, 1.]]
A_D, A_F = .35, .12
S_VOL, X_VOL = .2, .17


def time(d):
    return (date.fromisoformat(d)-date.fromisoformat(TODAY)).days/365.


def vanilla_reference(mode):
    # Integrate the Gaussian loading functions directly on each smooth interval.
    # Variables are log domestic discount to P, log equity at T, and log FX at T.
    T, P = time(END), time(PAY)
    cov = [[0.]*3 for _ in range(3)]
    vd_p = vd_t = vf = foreign_shift = 0.
    def b(a,t):
        return -math.expm1(-a*t)/a
    for lo,hi in zip([0.,.35,.6,T],[.35,.6,T,P]):
        du=(hi-lo)/128
        for j in range(128):
            u=lo+(j+.5)*du
            sd=.04 if u<.35 else .025
            sf=.06 if u<.6 else .035
            dp=sd*b(A_D,P-u)
            dt=sd*b(A_D,T-u) if u<T else 0.
            ft=sf*b(A_F,T-u) if u<T else 0.
            vd_p+=dp*dp*du
            vd_t+=dt*dt*du
            vf+=ft*ft*du
            foreign_shift+=C[1][3]*X_VOL*ft*du
            loads=[[0.,0.,-dp,0.],[S_VOL if u<T else 0.,0.,0.,ft],
                   [0.,X_VOL if u<T else 0.,dt,-ft]]
            for i in range(3):
                for k in range(3):
                    cov[i][k]+=du*sum(loads[i][a]*C[a][b_]*loads[k][b_]
                                      for a in range(4) for b_ in range(4))
    mf=.045*T+.5*vf-foreign_shift
    md=.015*T+.5*vd_t
    mean=[-.015*P-.5*vd_p, math.log(100.)-.02*T+mf-.5*S_VOL**2*T-C[0][1]*S_VOL*X_VOL*T,
          math.log(150.)+md-mf-.5*X_VOL**2*T]
    # E[e^a (e^b-K)+] by exponential tilting of a joint normal vector.
    wa=[1.,0.,float(mode=="floating_fx")]
    wb=[0.,1.,float(mode=="composite")]
    def quadratic(v,w):
        return sum(v[i]*w[j]*cov[i][j] for i in range(3) for j in range(3))
    def em(w):
        return math.exp(sum(m*c for m,c in zip(mean,w))+.5*quadratic(w,w))
    multiplier=em(wa)*(145. if mode=="quanto" else 1.)
    f=em(wb)*math.exp(quadratic(wa,wb))
    k=15000. if mode=="composite" else 100.
    vol=math.sqrt(quadratic(wb,wb))
    d2=math.log(f/k)/vol-vol/2
    def cdf(z):
        return .5*math.erfc(-z/math.sqrt(2.))
    return multiplier*(f*cdf(d2+vol)-k*cdf(d2))


class CrossCurrencyRateTests(unittest.TestCase):
    def setUp(self):
        def curve(i,r):
            return rp.DiscountCurve(i,[0.,1.],[1.,math.exp(-r)])
        self.market=rp.Market.equity(1,10,100.,curve(1,.045),curve(3,.02))
        self.jpy=curve(2,.015)
        self.d=rp.HullWhiteModel(A_D,[0.,.35],[.04,.025])
        self.f=rp.HullWhiteModel(A_F,[0.,.6],[.06,.035])

    def option(self,product,mode,model=None,**kwargs):
        return rp.CrossCurrencyEquityOption(TODAY,product,self.market,self.jpy,
            settlement=mode,fx_spot=150.,fx_volatility=X_VOL,correlation=C[0][1],
            equity_model=model or rp.Model.black_scholes(S_VOL),maximum_step=.25,
            fixed_fx_rate=145. if mode=="quanto" else None, **kwargs)

    def test_rate_and_fx_correlations_against_joint_gaussian_calls(self):
        engine=rp.Engine.randomized_quasi_monte_carlo(2048,817,scramble_count=12,antithetic=True,brownian_bridge=True)
        for mode in ("quanto","floating_fx","composite"):
            with self.subTest(mode=mode):
                product=rp.Product.european_vanilla(10,2,END,15000. if mode=="composite" else 100.,1.,"call")
                pricer=self.option(product,mode,payment_date=PAY).with_hull_white_rates(self.d,self.f,correlations=C)
                result=pricer.price_mc(engine)
                self.assertAlmostEqual(result.value,vanilla_reference(mode),delta=6*result.standard_error)

    def test_zero_coupon_bonds_in_both_currencies_and_ho_lee_limit(self):
        builder=rp.SourceGraphBuilder()
        amounts=[10.,-3.]
        graph=builder.finish([builder.literal(v) for v in amounts],[MID,PAY],currency_id=2,underlying_ids=[10])
        product=rp.Product.from_graph(graph)
        engine=rp.Engine.randomized_quasi_monte_carlo(1024,175,scramble_count=12,antithetic=True)
        for mode, rate, scale in [("quanto",.015,145.),("floating_fx",.045,150.),("composite",.015,1.)]:
            with self.subTest(mode=mode):
                r=rp.HullWhiteModel(0.,[0.],[.08])
                note=self.option(product,mode).with_hull_white_rates(r,r,correlations=C)
                result=note.price_mc(engine)
                expected=scale*sum(a*math.exp(-rate*time(p)) for a,p in zip(amounts,[MID,PAY]))
                self.assertAlmostEqual(result.value,expected,delta=6*result.standard_error+1e-10)

    def test_autocall_with_delayed_fx_in_every_equity_family(self):
        engine=rp.Engine.pseudo_monte_carlo(73,128,antithetic=True)
        for mode in ("quanto","floating_fx","composite"):
            unit=150. if mode=="composite" else 1.
            note=rp.ContractBuilder("coupon",2,10,"close").autocall(
                100.*unit,100.*unit,60.*unit,[MID,END],["2026-08-01",PAY],
                [104.*unit,105.*unit],[80.*unit,80.*unit],[.04,.05])
            graph,=note.to_graphs(TODAY,[10])
            for name,m in models(True):
                with self.subTest(mode=mode,model=name):
                    option=self.option(rp.Product.from_graph(graph),mode,m).with_hull_white_rates(self.d,self.f,correlations=C)
                    if mode=="floating_fx":
                        option=option.with_payment_fx_fixings([("2026-08-01",MID),(PAY,END)])
                    result=option.price_mc(engine)
                    self.assertTrue(11000.<result.value<17000.)
                    self.assertGreater(result.standard_error,0.)

    def test_memory_autocall_at_payment_fx_matches_foreign_hull_white_price(self):
        observations = [
            rp.AutocallObservation("2026-04-01", "2026-04-15", coupon_amount=2., coupon_level=1.05),
            rp.AutocallObservation(MID, "2026-07-15", coupon_amount=3., coupon_level=.9, call_level=1.03),
        ]
        def product(currency):
            note = rp.MultiAssetProduct.autocallable([10], [100.], observations, END, PAY,
                notional=100., final_barrier=.7, memory=True, on_autocall="pay",
                on_maturity="forfeit", currency_id=currency)
            return note.to_graph()

        engine = rp.Engine.randomized_quasi_monte_carlo(1024, 873, scramble_count=12,
                                                       antithetic=True, brownian_bridge=True)
        actual = self.option(rp.Product.from_graph(product(2)), "floating_fx").with_hull_white_rates(
            self.d, self.f, correlations=C).price_mc(engine)
        correlation = rp.CorrelationSchedule([10], [TODAY], [[[1.]]],
            symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12, psd_rel_tol=1e-12,
            zero_pivot_abs_tol=1e-14, zero_pivot_rel_tol=1e-14)
        foreign_hw = rp.MultiAssetHullWhiteConfig(self.f, [C[0][3]])
        model = rp.MultiAssetModelSpec([rp.Model.black_scholes(S_VOL)], correlation, hull_white=foreign_hw)
        source = rp.MultiAssetSpecification(TODAY, product(1), [self.market], model, engine,
                                            maximum_step=.25, worker_threads=1)
        native = rp.PricingTask.from_source(source).compile().evaluate().native_result
        self.assertAlmostEqual(actual.value / 150., native.value,
                               delta=6*math.hypot(actual.standard_error / 150., native.standard_error))

    def test_zero_rate_volatility_recovers_existing_prices(self):
        zero = rp.HullWhiteModel(.2, [0., .5], [0., 0.])
        engine = rp.Engine.pseudo_monte_carlo(91, 128, antithetic=True)
        for mode in ["quanto", "floating_fx", "composite"]:
            with self.subTest(mode=mode):
                strike = 15000. if mode == "composite" else 100.
                option = self.option(rp.Product.european_vanilla(10, 2, END, strike, 1., "call"), mode)
                base = option.price_mc(engine)
                result = option.with_hull_white_rates(zero, zero, correlations=C).price_mc(engine)
                self.assertEqual((result.value, result.standard_error), (base.value, base.standard_error))

    def test_reject_inconsistent_driver_correlations(self):
        option=self.option(rp.Product.european_vanilla(10,2,END,100.,1.,"call"),"quanto")
        for matrix in ([[1.]*4]*4, [row[:] for row in C]):
            matrix[2][3]=matrix[3][2]=-.999
            with self.assertRaises(ValueError):
                option.with_hull_white_rates(self.d,self.f,correlations=matrix)

    def test_fx_fixing_lag_includes_domestic_bond_covariance(self):
        builder=rp.SourceGraphBuilder()
        graph=builder.finish([builder.literal(100.)],[PAY],currency_id=2,underlying_ids=[10])
        base=self.option(rp.Product.from_graph(graph),"floating_fx")
        engine=rp.Engine.randomized_quasi_monte_carlo(1024,867,scramble_count=16,antithetic=True)
        fix,pay=time(MID),time(PAY)
        cov=0.
        for lo,hi in [(0.,.35),(.35,fix)]:
            step=(hi-lo)/100
            for j in range(100):
                u=lo+(j+.5)*step
                sd=.04 if u<.35 else .025
                bd=-math.expm1(-A_D*(fix-u))/A_D
                bf=-math.expm1(-A_F*(fix-u))/A_F
                bp=-math.expm1(-A_D*(pay-u))/A_D
                cov+=sd*(bp-bd)*(sd*bd-C[2][3]*.06*bf+C[1][2]*X_VOL)*step
        expected=15000.*math.exp(-.045*fix-.015*(pay-fix)-cov)
        with_rates=base.with_hull_white_rates(self.d,self.f,correlations=C)
        for p in (with_rates.with_payment_fx_fixings([(PAY,MID)]),
                  base.with_payment_fx_fixings([(PAY,MID)]).with_hull_white_rates(self.d,self.f,correlations=C)):
            result=p.price_mc(engine)
            self.assertAlmostEqual(result.value,expected,delta=6*result.standard_error)

    def test_foreign_density_drives_hull_white_lsv_calibration(self):
        density=rp.HullWhiteLsvTarget.flat(.2,[0.,.5,1.],[-.35,0.,.35])
        particles=rp.LsvParticleConfig(particle_count=256,calibration_seed=314,
                                       log_bandwidth=.6,minimum_effective_samples=2.)
        factors=[
            rp.VolatilityFactorSpec.bergomi(1.,.25,-.4),
            rp.VolatilityFactorSpec.bergomi_two_factor([.5,2.],.2,.3,[-.4,-.2],.2),
            rp.VolatilityFactorSpec.rough_bergomi(rp.RoughBergomiModel(.15,.25,equity_vol_correlation=-.4)),
        ]
        product=rp.Product.european_vanilla(10,2,END,100.,1.,"call")
        native_product=rp.Product.european_vanilla(10,1,END,100.,1.,"call")
        foreign_rate=rp.HullWhiteModel(.12,[0.],[.006])
        engine=rp.Engine.randomized_quasi_monte_carlo(512,134,scramble_count=8,antithetic=True)
        for i,factor in enumerate(factors):
            with self.subTest(factor=i):
                m=rp.EquityModelSpec.local_stochastic_volatility(density.model,factor,particles,forward_density=density)
                actual=self.option(product,"floating_fx",m).with_hull_white_rates(self.d,foreign_rate,correlations=C).price_mc(engine)
                rate=rp.RateModelSpec.hull_white(foreign_rate,equity_rate_correlation=C[0][3],
                    vol_rate_correlations=[v*C[0][3] for v in ([-.4,-.2] if i==1 else [-.4])])
                native=rp.PricingSpecification(TODAY,native_product,self.market,rp.PricingModelSpec(m,rates=rate),
                    engine,rp.RiskRequest(),rp.PricingNumerics(worker_threads=1)).compile().evaluate()
                self.assertAlmostEqual(actual.value/150.,native.value,
                    delta=6*math.hypot(actual.standard_error/150.,native.standard_error)+.03)
        mismatch=rp.EquityModelSpec.local_stochastic_volatility(
            rp.HullWhiteLsvTarget.flat(.3,[0.,.5,1.],[-.35,0.,.35]).model,factors[0],particles,forward_density=density)
        with self.assertRaisesRegex(rp.PricingError,"target grids"):
            self.option(product,"quanto",mismatch).with_hull_white_rates(self.d,foreign_rate).price_mc(engine)
