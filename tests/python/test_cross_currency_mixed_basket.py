"""USD and EUR stock observations, foreign cash payouts and JPY discounting."""
import math
import unittest

import rust_pricing as rp

from test_cross_currency_basket import TODAY, OBS, FIX, PAY, IDS, curve, years

USD, JPY, EUR = 1, 2, 3
# US stock, European stock, JPY/USD, JPY/EUR, JPY rate, USD rate, EUR rate.
C = [
    [1., .42, -.32, .17, .10, .26, -.16],
    [.42, 1., .12, .31, -.12, -.14, .28],
    [-.32, .12, 1., .38, .16, -.18, .10],
    [.17, .31, .38, 1., -.20, .12, .25],
    [.10, -.12, .16, -.20, 1., .22, .17],
    [.26, -.14, -.18, .12, .22, 1., -.15],
    [-.16, .28, .10, .25, .17, -.15, 1.],
]
QUOTES = [(USD, 150., .16), (EUR, 173., .23)]
INITIAL_RATES = [.012, .048, .029]
A = [.35, .12, .50]
STOCK_VOL, YIELD = [.21, .28], [.013, .022]


def rate_vol(i, u):
    return [[.028, .037], [.045, .032], [.052, .025]][i][int(u >= .38)]


def minimum_reference(mode, cash_currency):
    # Gaussian tilting of the discounted minimum of two lognormal performances.
    # Build means and covariance directly from continuous-time Brownian kernels.
    t, f, p = years(OBS), years(FIX), years(PAY)
    mean = [0., 0., 0.]
    rates = [[(0, p, -1.)], [(1, t, 1.)], [(2, t, 1.)]]
    diffusions = [[], [(0, t, STOCK_VOL[0])], [(1, t, STOCK_VOL[1])]]

    def add_fx(row, k, h):
        sigma = QUOTES[k][2]
        mean[row] -= .5*sigma**2*h
        diffusions[row].append((2+k, h, sigma))
        rates[row].extend([(0, h, 1.), (1+k, h, -1.)])

    if mode == "floating_fx":
        k = [USD, EUR].index(cash_currency)
        mean[0] = math.log(QUOTES[k][1])
        add_fx(0, k, f)
    elif mode == "quanto" and cash_currency == USD:
        mean[0] = math.log(140.)
    for i in range(2):
        mean[1+i] -= t*(YIELD[i]+.5*STOCK_VOL[i]**2+C[i][2+i]*STOCK_VOL[i]*QUOTES[i][2])
        if mode == "composite":
            add_fx(1+i, i, t)
    for row, terms in enumerate(rates):
        mean[row] += sum(INITIAL_RATES[k]*h*w for k,h,w in terms)
    cov = [[0.]*3 for _ in mean]
    breaks = sorted(set([0., .38, t, f, p]))
    for lo, hi in zip(breaks, breaks[1:]):
        step = (hi-lo)/128
        for cell in range(128):
            u = lo+(cell+.5)*step
            loads = [[0.]*7 for _ in mean]
            for row in range(3):
                for k,h,w in diffusions[row]:
                    if u < h:
                        loads[row][k] += w
                for k,h,w in rates[row]:
                    if u >= h:
                        continue
                    kernel = -rate_vol(k,u)*math.expm1(-A[k]*(h-u))/A[k]
                    loads[row][4+k] += w*kernel
                    tilt = 0. if k == 0 else QUOTES[k-1][2]*C[1+k][4+k]
                    mean[row] += step*w*(.5*kernel**2-tilt*kernel)
            for i in range(3):
                for j in range(3):
                    cov[i][j] += step*sum(x*y*C[a][b] for a,x in enumerate(loads[i])
                                         for b,y in enumerate(loads[j]))
    sd = math.sqrt(cov[1][1]+cov[2][2]-2*cov[1][2])
    value = 0.
    for i,j in [(1,2),(2,1)]:
        # E[discount * U_i * 1(log U_j - log U_i >= 0)]
        tilted = mean[j]-mean[i]+cov[j][0]-cov[i][0]+cov[j][i]-cov[i][i]
        multiplier = math.exp(mean[0]+mean[i]+.5*(cov[0][0]+cov[i][i]+2*cov[0][i]))
        value += multiplier*.5*math.erfc(-tilted/(sd*math.sqrt(2.)))
    return value


class MixedBasketTests(unittest.TestCase):
    def setUp(self):
        self.jpy, self.usd, self.eur = [curve(i+1, r) for i,r in enumerate(INITIAL_RATES)]
        self.markets = [rp.Market.equity(USD, 10, 100., self.usd, curve(4,YIELD[0])),
                        rp.Market.equity(EUR, 20, 80., self.eur, curve(5,YIELD[1]))]
        self.models = [rp.Model.black_scholes(sigma) for sigma in STOCK_VOL]
        self.rates = [rp.HullWhiteModel(A[i], [0., .38], [rate_vol(i,0.),rate_vol(i,1.)])
                      for i in range(3)]
        self.engine = rp.Engine.randomized_quasi_monte_carlo(2048, 172, scramble_count=12,
                                                            antithetic=True, brownian_bridge=True)

    def minimum(self, mode):
        b = rp.SourceGraphBuilder()
        levels = [100.,80.]
        if mode == "composite":
            levels = [v*q[1] for v,q in zip(levels,QUOTES)]
        names = [b.divide(b.spot(i,OBS),b.literal(level)) for i,level in zip(IDS, levels)]
        return b.finish([b.minimum(*names)], [PAY], currency_id=JPY, underlying_ids=IDS)

    def option(self, graph, mode, **changes):
        markets = changes.pop("markets", self.markets)
        args = dict(equity_models=self.models, settlement=mode, fx_quotes=QUOTES,
                    correlations=C, rates=self.rates, maximum_step=.25)
        return rp.CrossCurrencyEquityBasket.from_currencies(TODAY, graph, markets, self.jpy, **(args|changes))

    def test_discounted_worst_name_against_gaussian_minimum(self):
        for mode, payout in [("quanto", JPY), ("quanto", USD), ("floating_fx", USD),
                             ("floating_fx", EUR), ("composite", JPY)]:
            with self.subTest(mode=mode, payout=payout):
                option = self.option(self.minimum(mode), mode, payout_currency_id=payout,
                                     fixed_fx_rate=140. if (mode,payout)==("quanto",USD) else None)
                if mode == "floating_fx":
                    option = option.with_payment_fx_fixings([(PAY,FIX)])
                v = option.price_mc(self.engine)
                self.assertAlmostEqual(v.value, minimum_reference(mode,payout),
                                       delta=6*v.standard_error+1.e-7)

    def test_cash_bonds_in_each_currency(self):
        b = rp.SourceGraphBuilder()
        cash = b.finish([b.literal(10.), b.literal(-3.)], [OBS,PAY], currency_id=JPY, underlying_ids=IDS)
        for mode, currency, spot, r in [("quanto", JPY, 1., .012), ("floating_fx", USD, 150., .048),
                                        ("floating_fx", EUR, 173., .029)]:
            with self.subTest(currency=currency):
                result = self.option(cash,mode,payout_currency_id=currency).price_mc(self.engine)
                expected = spot*(10.*math.exp(-r*years(OBS))-3.*math.exp(-r*years(PAY)))
                self.assertAlmostEqual(result.value, expected, delta=6*result.standard_error+1e-8)

    def test_local_and_yen_returns_change_autocall_date_and_memory(self):
        # Both local share prices grow. EUR depreciates sufficiently to miss the
        # first JPY coupon and call, then earns the two coupons at the second call.
        jpy = curve(10,0.)
        eur = curve(12,.22)
        markets = [rp.Market.equity(USD,10,100.,jpy,curve(13,-.08)),
                   rp.Market.equity(EUR,20,80.,eur,curve(14,.05))]
        obs_dates = ["2026-07-01",OBS]
        pay_dates = ["2026-07-08",FIX]
        zero = [rp.Model.black_scholes(0.)]*2
        for mode,payout in [("quanto",JPY),("floating_fx",EUR),("composite",JPY)]:
            unit = 1000.
            refs = [15000.,80.*173.] if mode=="composite" else [100.,80.]
            obs = [rp.AutocallObservation(d,p,coupon_amount=v,coupon_level=k,call_level=k)
                   for d,p,v,k in zip(obs_dates,pay_dates,[20.,30.],[1.02,.93])]
            graph = rp.MultiAssetProduct.autocallable(IDS,refs,obs,"2027-01-01",PAY,notional=unit,
                final_barrier=.70,memory=True,on_autocall="pay",on_maturity="forfeit",currency_id=JPY).to_graph()
            opt = rp.CrossCurrencyEquityBasket.from_currencies(TODAY,graph,markets,jpy,equity_models=zero,
                settlement=mode,fx_quotes=[(USD,150.,0.),(EUR,173.,0.)],correlations=[r[:4] for r in C[:4]],
                payout_currency_id=payout, maximum_step=.25)
            if mode=="floating_fx":
                opt=opt.with_payment_fx_fixings(list(zip(pay_dates,obs_dates)))
            pv=opt.price_mc(rp.Engine.pseudo_monte_carlo(81,16))
            expected = (1050. if mode=="composite" else 1020.)
            if mode=="floating_fx":
                expected *= 173.*math.exp(-.22*years(obs_dates[0]))
            self.assertAlmostEqual(pv.value, expected, places=8)

    def test_same_day_maps_fx_by_currency_id_and_allows_shared_foreign_currency(self):
        b=rp.SourceGraphBuilder()
        ids=[20,10,30]
        total=b.add(b.spot(20,TODAY), b.add(b.spot(10,TODAY), b.spot(30,TODAY)))
        g=b.finish([total],[TODAY],currency_id=JPY,underlying_ids=ids)
        markets=[self.markets[1],self.markets[0],rp.Market.equity(EUR,30,75.,self.eur,curve(6,.01))]
        ident=[[float(i==j) for j in range(5)] for i in range(5)]
        for quotes in (QUOTES, QUOTES[::-1]):
            pv=self.option(g,"composite",markets=markets,equity_models=self.models+[self.models[1]],
                fx_quotes=quotes,correlations=ident,rates=None).price_mc(self.engine)
            self.assertEqual((pv.value,pv.standard_error),((80.+75.)*173.+100.*150.,0.))

    def test_permuted_fx_and_rate_order_preserves_euro_settlement(self):
        # Currency IDs select markets. Permuting quote and rate order also
        # permutes the corresponding rows and columns of the driver matrix.
        order = [0, 1, 3, 2, 4, 6, 5]
        alt = self.option(self.minimum("floating_fx"), "floating_fx", payout_currency_id=EUR,
            fx_quotes=QUOTES[::-1], rates=[self.rates[i] for i in [0,2,1]],
            correlations=[[C[i][j] for j in order] for i in order]).with_payment_fx_fixings([(PAY,FIX)])
        result = alt.price_mc(self.engine)
        self.assertAlmostEqual(result.value, minimum_reference("floating_fx",EUR),
                               delta=6*result.standard_error+1e-7)

    def test_equivalent_currencies_reduce_to_shared_fx_autocall(self):
        obs = [rp.AutocallObservation("2026-04-01", "2026-04-15", coupon_amount=2., coupon_level=1.05),
               rp.AutocallObservation(OBS, FIX, coupon_amount=3., coupon_level=.9, call_level=1.02)]
        graph = rp.MultiAssetProduct.autocallable(IDS, [100.,80.], obs, "2027-01-01", PAY,
            notional=100., final_barrier=.7, memory=True, on_autocall="pay", on_maturity="forfeit",
            currency_id=JPY).to_graph()
        foreign_curve = self.usd
        eur_market = rp.Market.equity(EUR,20,80.,foreign_curve,curve(5,YIELD[1]))
        usd_market = rp.Market.equity(USD,20,80.,foreign_curve,curve(5,YIELD[1]))
        one = [0,1,2,4,5]
        base = [[C[i][j] for j in one] for i in one]
        duplicate = [0,1,2,2,3,4,4]
        pay_fixings = [(FIX,OBS),(PAY,"2027-01-01")]
        solo = rp.CrossCurrencyEquityBasket(TODAY,graph,[self.markets[0],usd_market],self.jpy,
            equity_models=self.models,settlement="floating_fx",fx_spot=150.,fx_volatility=.16,
            rates=tuple(self.rates[:2]),correlations=base,maximum_step=.25).with_payment_fx_fixings(pay_fixings)
        mixed = self.option(graph,"floating_fx",payout_currency_id=EUR,markets=[self.markets[0],eur_market],
            fx_quotes=[(USD,150.,.16),(EUR,150.,.16)], rates=[*self.rates[:2],self.rates[1]],
            correlations=[[base[i][j] for j in duplicate] for i in duplicate]).with_payment_fx_fixings(pay_fixings)
        a, b = [option.price_mc(self.engine) for option in [solo,mixed]]
        self.assertAlmostEqual(a.value,b.value,delta=6*math.hypot(a.standard_error,b.standard_error))

    def test_euro_dividend_uses_euro_short_rate(self):
        b = rp.SourceGraphBuilder()
        cash = b.subtract(b.spot(20,OBS,pre_dividend=True),b.spot(20,OBS))
        graph = b.finish([cash],[OBS],currency_id=JPY,underlying_ids=IDS)
        t = years(OBS)
        eur = rp.Market.equity(EUR,20,80.,self.eur,curve(5,YIELD[1]),
                               discrete_dividends=[rp.DividendEvent.fixed_cash(1,t,6.)])
        div = rp.BuehlerDividendModel(mean_reversion=0.,equity_linkage=.5,
                                      dividend_volatility=.36,equity_dividend_correlation=.65)
        actual = self.option(graph,"floating_fx",payout_currency_id=EUR,markets=[self.markets[0],eur],
                             dividend_models=[None,div]).price_mc(self.engine)
        # Under the euro measure the discount/dividend covariance uses EUR rate volatility.
        a = A[2]
        def integral(h):
            return (h+math.expm1(-a*h)/a)/a
        kernel = rate_vol(2,0.)*(integral(t)-integral(t-.38))+rate_vol(2,t)*integral(t-.38)
        expected = 173.*6.*math.exp(-INITIAL_RATES[2]*t-.36*.65*C[1][6]*kernel)
        self.assertAlmostEqual(actual.value,expected,delta=6*actual.standard_error)

    def test_currency_and_dimension_errors(self):
        graph=self.minimum("quanto")
        for kw in [dict(fx_quotes=QUOTES[:1]),dict(fx_quotes=[QUOTES[0]]*2),dict(fx_quotes=QUOTES+[(4,1.,.1)]),
                   dict(fx_quotes=[(JPY,1.,.1),QUOTES[1]]),dict(rates=self.rates[:2]),
                   dict(correlations=[r[:5] for r in C[:5]]),dict(fx_quotes=[QUOTES[0],(EUR,-1.,.2)]),
                   dict(payout_currency_id=4),dict(payout_currency_id=EUR),dict(fixed_fx_rate=140.)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.option(graph,"quanto",**kw)
        for mode,kw in [("floating_fx",{}),("floating_fx",dict(payout_currency_id=JPY)),
                        ("composite",dict(payout_currency_id=USD))]:
            with self.assertRaises(ValueError):
                self.option(graph,mode,**kw)
