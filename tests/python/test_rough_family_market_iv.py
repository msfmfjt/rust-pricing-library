"""Market-IV quote risk, independent Dupire rebuilds and immutable contracts."""
import bisect
import math
import unittest
import rust_pricing as rp
from test_rough_volatility_families import models

QT = [0.2, 0.6, 1.2]
QX = [-0.9, -0.45, 0.05, 0.5, 0.9]
TT = [0.0, 0.13, 0.37, 0.7, 1.0]
TX = [-0.55, -0.13, 0.18, 0.6]
IV = [0.2 + 0.002*i - 0.003*x + 0.002*x*x for i in range(3) for x in QX]


def independent_dupire(vols):
    """Independent natural spline via tridiagonal equations, then direct Dupire.

    No production surface, transpose, grid or calibration routine is used to
    construct these target node values. Calibration/pricing below is shared.
    """
    def spatial(row, x):
        y = [QT[row]*v*v for v in vols[row*5:(row+1)*5]]
        h = [b-a for a,b in zip(QX,QX[1:])]
        diag = [2*(h[i]+h[i+1]) for i in range(3)]
        rhs = [6*((y[i+2]-y[i+1])/h[i+1]-(y[i+1]-y[i])/h[i]) for i in range(3)]
        for i in range(1,3):
            q = h[i]/diag[i-1]
            diag[i] -= q*h[i]
            rhs[i] -= q*rhs[i-1]
        second = [0.0]*5
        second[3] = rhs[2]/diag[2]
        for i in [1,0]:
            second[i+1] = (rhs[i]-h[i+1]*second[i+2])/diag[i]
        j = min(max(bisect.bisect_right(QX,x)-1,0),3)
        a = (QX[j+1]-x)/h[j]; b = 1-a
        # Local polynomial and derivatives, rather than a quote basis transpose.
        w = a*y[j]+b*y[j+1]+h[j]**2/6*((a**3-a)*second[j]+(b**3-b)*second[j+1])
        wx = (y[j+1]-y[j])/h[j]+h[j]/6*((1-3*a*a)*second[j]+(3*b*b-1)*second[j+1])
        wxx = a*second[j]+b*second[j+1]
        return w,wx,wxx
    out = []
    for t in TT:
        t = TT[1] if t == 0 else t
        for x in TX:
            if t < QT[0] or t >= QT[-1]:
                row = 0 if t < QT[0] else 2
                z = spatial(row,x)
                w,wx,wxx = [v*t/QT[row] for v in z]
                wt = z[0]/QT[row]
            else:
                j = bisect.bisect_right(QT,t)-1
                a,b = spatial(j,x),spatial(j+1,x)
                h = QT[j+1]-QT[j]; q = (t-QT[j])/h
                w,wx,wxx = [(1-q)*u+q*v for u,v in zip(a,b)]
                wt = (b[0]-a[0])/h
            g = (1-x*wx/(2*w))**2-(wx*wx/4)*(1/w+0.25)+wxx/2
            if not (w > 0 and wt > 0 and g > 0):
                raise ValueError('invalid independent Dupire input')
            out.append(wt/g)
    return rp.Model.local_volatility_from_grid(TT, TX, out, 1e-8, 4.0)


def make_request(vols=IV, *, qmc=True, product=None, independent=False, smooth=None, notional=1.0):
    source = rp.MarketIvSurface(QT, QX, vols)
    target = independent_dupire(vols) if independent else source.local_volatility_model(TT,TX)
    discount = rp.DiscountCurve(10,[0.0,1.0,2.0],[1.0,0.95,0.95**2])
    repo = rp.DiscountCurve(11,[0.0,1.0,2.0],[1.0,0.98,0.98**2])
    market = rp.Market.equity(2,1,100.0,discount,repo,discrete_dividends=[
        rp.DividendEvent.fixed_cash(1,0.25,3.0),rp.DividendEvent.fixed_cash(2,1.5,4.0)])
    engine = (rp.Engine.randomized_quasi_monte_carlo(64,819,scramble_count=4,
        antithetic=True,brownian_bridge=True) if qmc else
        rp.Engine.pseudo_monte_carlo(819,128,antithetic=True,brownian_bridge=False))
    req = rp.PricingRequest('2026-09-04',product or
        rp.Product.european_vanilla(1,2,'2027-09-04',100.0,notional,'call'),
        market,target,engine,rp.RiskRequest(payoff_smoothing_half_width=smooth))
    return req,source


def build(model, vols=IV, *, trace=True, workers=1, **kwargs):
    request,source = make_request(vols,**kwargs)
    return rp.RoughFamilyLsvPlan.compile(request,model,particle_count=128,
        calibration_seed=429,log_bandwidth=0.5,minimum_effective_samples=3.0,
        retain_reverse_trace=trace,worker_threads=workers,reduction_block_size=64),source


class RoughMarketIvTest(unittest.TestCase):
    def test_all_families_and_independent_dupire_recalibration_directions(self):
        direction = [math.cos(0.7*i) for i in range(15)]
        for model in models():
            plan,source = build(model)
            result = plan.market_iv_risk_plan(source).evaluate()
            self.assertEqual(result.price.value,plan.evaluate().value)
            self.assertEqual(result.price.standard_error,plan.evaluate().standard_error)
            got = math.fsum(a*d for a,d in zip(result.quote_adjoints,direction))
            for eps in [1e-6,5e-7]:
                up,_ = build(model,[v+eps*d for v,d in zip(IV,direction)],independent=True,trace=False)
                dn,_ = build(model,[v-eps*d for v,d in zip(IV,direction)],independent=True,trace=False)
                fd = (up.evaluate().value-dn.evaluate().value)/(2*eps)
                self.assertAlmostEqual(got,fd,delta=3e-6*(1+abs(fd)))

    def test_individual_quotes_parallel_vega_and_units_with_asian_payment(self):
        obs = [rp.AsianObservation.unknown(d,1/3) for d in
               ['2026-12-04','2027-03-04','2027-09-04']]
        product = rp.Product.arithmetic_asian(1,2,100.0,1.0,'call',obs,'2027-12-04')
        for model in [models()[0],models()[-1]]:
            plan,source=build(model,product=product)
            risk=plan.market_iv_risk_plan(source).evaluate()
            for index in [0,7,14,None]:
                got=risk.parallel_vega if index is None else risk.quote_adjoints[index]
                for e in [1e-6,5e-7]:
                    d=[1.0 if index is None or j==index else 0.0 for j in range(15)]
                    up,_=build(model,[v+e*a for v,a in zip(IV,d)],product=product,trace=False)
                    dn,_=build(model,[v-e*a for v,a in zip(IV,d)],product=product,trace=False)
                    fd=(up.evaluate().value-dn.evaluate().value)/(2*e)
                    self.assertAlmostEqual(got,fd,delta=3e-6*(1+abs(fd)))
            self.assertAlmostEqual(risk.parallel_vega,math.fsum(risk.quote_adjoints),delta=1e-10)

    def test_owned_fields_determinism_fingerprint_and_mc_error_semantics(self):
        model=models()[0]
        p,s=build(model)
        risk_plan=p.market_iv_risk_plan(s); a=risk_plan.evaluate()
        other,source=build(model,workers=3); b=other.market_iv_risk_plan(source).evaluate()
        self.assertEqual(a.quote_adjoints,b.quote_adjoints)
        self.assertEqual(a.standard_errors,b.standard_errors)
        self.assertEqual(a.parallel_standard_error,b.parallel_standard_error)
        self.assertEqual(a.risk_fingerprint,risk_plan.risk_fingerprint)
        self.assertNotEqual(a.risk_fingerprint,p.plan_fingerprint)
        self.assertEqual(a.coordinate,'market_black_iv_nodes_in_relative_log_moneyness')
        self.assertEqual(a.method,'rough-family-market-iv-particle-adjoint-v1')
        self.assertEqual(a.interpolation,s.interpolation)
        self.assertEqual(a.uncertainty_scope,'pricing_conditional_on_calibration')
        self.assertEqual(a.maturity_nodes,QT);self.assertEqual(a.log_moneyness_nodes,QX)
        self.assertEqual(a.implied_volatilities,IV)
        copy=a.quote_adjoints;copy[0]=math.nan
        self.assertTrue(math.isfinite(a.quote_adjoints[0]))
        for name in ['price','quote_adjoints','standard_errors','parallel_vega','parallel_standard_error','risk_fingerprint','coordinate','method']:
            with self.assertRaises(AttributeError):setattr(a,name,0)
        p,s=build(model,qmc=False); b=p.market_iv_risk_plan(s).evaluate()
        self.assertIsNone(b.standard_errors);self.assertIsNone(b.parallel_standard_error)
        self.assertTrue(math.isfinite(b.parallel_vega))

    def test_missing_trace_mismatched_source_and_unsmoothed_digital_fail(self):
        p,s=build(models()[0],trace=False)
        with self.assertRaises(rp.PricingError):p.market_iv_risk_plan(s)
        p,s=build(models()[0])
        with self.assertRaises(rp.PricingError):
            p.market_iv_risk_plan(rp.MarketIvSurface(QT,QX,[v+0.001 for v in IV]))
        product=rp.Product.digital(1,2,'2027-09-04',100.0,1.0,'call','cash')
        p,s=build(models()[0],product=product)
        with self.assertRaises(rp.PricingError):p.market_iv_risk_plan(s)
        p,s=build(models()[0],product=product,smooth=1.0)
        result=p.market_iv_risk_plan(s).evaluate()
        self.assertTrue(all(math.isfinite(a) for a in result.quote_adjoints))

if __name__=='__main__':unittest.main()
