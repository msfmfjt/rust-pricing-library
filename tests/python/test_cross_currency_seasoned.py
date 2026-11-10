"""Mid-life mixed-currency notes, payment lags, published fixings and settlements."""
import json
import math
import unittest

import rust_pricing as rp

from test_cross_currency_autocall_legs import legs, note_terms, FIRST, FIRST_PAY, END
from test_cross_currency_basket import OBS, PAY, IDS, curve
from test_cross_currency_mixed_basket import USD, EUR, JPY


class SeasonedNoteTests(unittest.TestCase):
    def setUp(self):
        self.as_of = "2026-11-01"
        domestic = self.domestic = curve(JPY,0.)
        self.stocks = [rp.Market.equity(USD,10,100.,domestic,domestic),
                       rp.Market.equity(EUR,20,80.,domestic,domestic)]
        self.engine = rp.Engine.pseudo_monte_carlo(7,16)
        self.quotes = [(USD,150.,0.),(EUR,175.,0.)]
        self.first = [(10,FIRST,100.),(20,FIRST,80.)]

    def price(self, residual, historical_fx=False):
        count = len(residual[0].underlying_ids)
        factors = 2*count
        basket = rp.CrossCurrencyEquityBasket.from_legs(self.as_of,residual,self.stocks[:count],self.domestic,
            settlement_currency_id=JPY,equity_models=[rp.Model.black_scholes(0.)]*count,
            fx_quotes=self.quotes[:count],correlations=[[float(i==j) for j in range(factors)] for i in range(factors)])
        if historical_fx:
            basket = basket.with_leg_fx(0,schedule=[(PAY,OBS)],known_fixings=[(OBS,141.)])
        return basket.price_mc(self.engine).value

    def test_past_call_omits_later_unobserved_stocks_and_future_cash(self):
        obs = [rp.AutocallObservation(FIRST,PAY,coupon_amount=20.,coupon_level=.9,call_level=.95,
                                      call_coupon_amount=8.),
               rp.AutocallObservation(OBS,PAY,coupon_amount=90.,coupon_level=.7,call_level=.85)]
        products = legs(note_terms(obs,notional=10000.))
        residual = [p.remaining_cashflows(self.as_of,fixings=self.first) for p in products]
        self.assertEqual([p.payment_dates for p in residual],[[PAY],[PAY]])
        self.assertEqual(self.price(residual,True),10000.+28.*141.)
        restored = [rp.GraphProduct.from_json(p.to_json()) for p in residual]
        self.assertEqual(self.price(restored,True),10000.+28.*141.)

    def test_published_and_unpaid_memory_coupon_survives_until_recovery(self):
        # A paid ordinary coupon, then a missed coupon. Recovery at END uses
        # the later FX, regardless of the exchange rate on the missed date.
        obs = [rp.AutocallObservation(FIRST,FIRST_PAY,coupon_amount=5.,coupon_level=.9),
               rp.AutocallObservation(OBS,OBS,coupon_amount=12.,coupon_level=1.1),
               rp.AutocallObservation(END,PAY,coupon_amount=15.,coupon_level=.9,call_level=.95)]
        products = legs(note_terms(obs,notional=10000.))
        history = self.first + [(10,OBS,100.),(20,OBS,80.)]
        coupon, principal = products
        with self.assertRaises(ValueError):
            coupon.remaining_cashflows(self.as_of,fixings=history)  # first payment is overdue
        coupon = coupon.remaining_cashflows(self.as_of,fixings=history,settled_outputs=[0])
        principal = principal.remaining_cashflows(self.as_of,fixings=history)
        self.assertEqual(self.price([coupon,principal]),10000.+(12.+15.)*150.)
        with self.assertRaises(ValueError):
            products[0].remaining_cashflows(self.as_of,fixings=self.first,settled_outputs=[0])

    def test_paid_or_cancelled_note_has_exact_zero_value(self):
        obs = [rp.AutocallObservation(FIRST,FIRST_PAY,coupon_amount=5.,coupon_level=.9,call_level=.95),
               rp.AutocallObservation(OBS,PAY,coupon_amount=90.,coupon_level=.7,call_level=.85)]
        products = legs(note_terms(obs))
        residual = [p.remaining_cashflows(self.as_of,fixings=self.first,settled_outputs=[0])
                    for p in products]
        self.assertEqual(self.price(residual),0.)

    def test_branch_selection_and_pre_dividend_fixings(self):
        b = rp.SourceGraphBuilder()
        post = b.spot(10,FIRST)
        before = b.spot(10,FIRST,pre_dividend=True)
        unused = b.spot(20,OBS)
        condition = b.indicator(b.subtract(post,b.literal(90.)))
        chosen = b.subtract(before,post)
        wire = json.loads(b.finish([chosen],[PAY],currency_id=JPY,underlying_ids=IDS).to_json())
        wire["schema_version"] = 2
        nodes = wire["product"]["nodes"]
        select = max(n["id"] for n in nodes)+1
        nodes.append(dict(id=select, operation=dict(type="select",condition=condition,
                                                   then_value=chosen,else_value=unused)))
        wire["product"]["cashflows"][0]["output_node"] = select
        graph = rp.GraphProduct.from_json(json.dumps(wire))
        residual = graph.remaining_cashflows(self.as_of,fixings=[(10,FIRST,95.)],
                                              pre_dividend_fixings=[(10,FIRST,102.)])
        self.assertEqual(self.price([residual]),7.)
        for kw in [dict(fixings=[(10,FIRST,95.)]), dict(fixings=[(10,END,95.)]),
                   dict(fixings=[(10,FIRST,95.)]*2),dict(settled_outputs=[0]),
                   dict(fixings=[(1000,FIRST,95.)]),dict(fixings=[(10,FIRST,math.nan)])]:
            with self.assertRaises(ValueError):
                graph.remaining_cashflows(self.as_of,**kw)

    def test_native_dated_products_export_for_settlement_after_expiry(self):
        product = rp.Product.european_vanilla(10,JPY,OBS,90.,1.,"call")
        issued = product.to_graph("2026-01-01",currency_id=USD,payment_date=PAY)
        unpaid = issued.remaining_cashflows(self.as_of,fixings=[(10,OBS,105.)])
        basket = rp.CrossCurrencyEquityBasket.from_legs(
            self.as_of,[unpaid],self.stocks[:1],self.domestic,settlement_currency_id=JPY,
            equity_models=[rp.Model.black_scholes(.2)],fx_quotes=self.quotes[:1],
            correlations=[[1.,.2],[.2,1.]])
        basket = basket.with_leg_fx(0,schedule=[(PAY,OBS)],known_fixings=[(OBS,135.)])
        self.assertEqual(basket.price_mc(self.engine).value,15.*135.)
        self.assertEqual(basket.sensitivity("fx_spot",USD,1.,self.engine).value,0.)
        for special in [rp.Product.american_vanilla(10,JPY,END,100.,1.,"put",[OBS,END]),
                        rp.Product.barrier(10,JPY,END,100.,140.,1.,"call","up","knock_out",
                                           "continuous",[FIRST,END],PAY)]:
            with self.assertRaises(ValueError):
                special.to_graph("2026-01-01")

    def test_replay_of_lapsed_and_exercised_rights(self):
        issue = rp.Product.american_vanilla(10,JPY,END,105.,3.,"put",[FIRST,OBS,END])
        live = issue.after_exercise_history(self.as_of,lapsed_dates=[FIRST,OBS])
        native = rp.CrossCurrencyEquityOption(self.as_of,live,self.stocks[0],self.domestic,
            settlement="floating_fx",fx_spot=150.,fx_volatility=0.,correlation=0.,equity_volatility=0.)
        self.assertEqual(native.price_mc(self.engine,lsm=rp.LsmConfig(self.engine)).value,2250.)
        exercised = issue.after_exercise_history(FIRST,lapsed_dates=[],exercised_on=FIRST)
        due = exercised.to_graph(FIRST,currency_id=USD).remaining_cashflows(
            FIRST,fixings=[(10,FIRST,92.)])
        basket = rp.CrossCurrencyEquityBasket.from_legs(FIRST,[due],self.stocks[:1],self.domestic,
            settlement_currency_id=JPY,equity_models=[rp.Model.black_scholes(.2)],
            fx_quotes=self.quotes[:1],correlations=[[1.,0.],[0.,1.]])
        basket = basket.with_leg_fx(0,schedule=[(FIRST,FIRST)],known_fixings=[(FIRST,147.)])
        self.assertEqual(basket.price_mc(self.engine).value,3.*(105.-92.)*147.)
        settled = due.remaining_cashflows(self.as_of,settled_outputs=[0])
        self.assertEqual(self.price([settled]),0.)
        lapsed = issue.after_exercise_history(END,lapsed_dates=[FIRST,OBS,END])
        self.assertEqual(lapsed.to_graph(END).payment_dates,[END])
        for kwargs in [dict(lapsed_dates=[FIRST]),dict(lapsed_dates=[FIRST,FIRST,OBS]),
                       dict(lapsed_dates=[FIRST,OBS,END]),dict(lapsed_dates=[FIRST],exercised_on=FIRST)]:
            with self.assertRaises(ValueError):
                issue.after_exercise_history(self.as_of,**kwargs)

    def test_resolved_continuous_barrier_between_expiry_and_payment(self):
        for style, hit, cash in [("knock_out",True,4.),("knock_in",True,30.),
                                 ("knock_out",False,30.),("knock_in",False,4.)]:
            p = rp.Product.barrier(10,JPY,OBS,100.,120.,2.,"call","up",style,"continuous",
                                   [FIRST,OBS],PAY,rebate=4.,historical_hit=hit)
            remaining = p.to_graph(self.as_of,currency_id=USD).remaining_cashflows(
                self.as_of,fixings=[(10,OBS,115.)])
            self.assertEqual(self.price([remaining],historical_fx=True),cash*141.)
