import unittest
from rust_pricing_backtest.multi_step_provider import gamma_boundaries


class PutBoundaryTests(unittest.TestCase):
    def test_fixed_minimum_and_strike_are_included_without_changing_legacy_bins(self):
        p={'product':dict(type='fixed_lookback',side={'type':'put'},strike=100.,historical_extremum=98.)}
        legacy=[100*x for x in (.8,.9,1.,1.1,1.2)]
        self.assertEqual(gamma_boundaries(p,100.),legacy)
        self.assertEqual(gamma_boundaries(p,100.,put_knots=True),sorted([*legacy,98.]))
        p['product']['valuation_date_fixing']=90.
        self.assertEqual(gamma_boundaries(p,100.,put_knots=True),legacy)
        p['product']['valuation_date_fixing']=105.
        self.assertIn(98.,gamma_boundaries(p,108.,put_knots=True))
        p['product']['historical_extremum']=120.
        self.assertIn(100.,gamma_boundaries(p,108.,put_knots=True))

    def test_call_and_asian_bins_do_not_change(self):
        for kind,side in [('fixed_lookback','call'),('arithmetic_asian','put')]:
            p={'product':dict(type=kind,side={'type':side},strike=100.,historical_extremum=98.)}
            self.assertEqual(gamma_boundaries(p,108.,put_knots=True),gamma_boundaries(p,108.))


if __name__=='__main__':unittest.main()
