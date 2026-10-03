"""Controls for the independent finite-grid rough-LSV hard Barrier reference."""
import json
from pathlib import Path
import unittest

from rough_dividend_conditional_barrier import price_delta


class ConditionalHardBarrierReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'
        cls.fixture = json.loads((directory / 'rough-conditional-barrier-reference.json').read_text())
        cls.market = json.loads((directory / cls.fixture['market_contract_fixture']).read_text())

    def test_quadrature_orders_tails_and_retained_values(self):
        for case in self.fixture['cases']:
            for order, inner, tail in ((12, 24, 10), (24, 32, 10), (40, 40, 12)):
                actual = price_delta(case, self.market, order, inner, tail=tail)
                for value, expected in zip(actual, (case['price'], case['delta'])):
                    self.assertAlmostEqual(value, expected, delta=self.fixture['quadrature_agreement_tolerance'])

    def test_zero_eta_kappa_recovers_separate_bivariate_reference(self):
        for case in self.fixture['cases']:
            limit = dict(case, eta=0.0, kappa=0.0, squared_leverage=[0.04] * 9)
            price, delta = price_delta(limit, self.market, 24, 32)
            self.assertAlmostEqual(price, self.market['price'], delta=2e-8)
            self.assertAlmostEqual(delta, self.market['delta'], delta=2e-8)

    def test_analytic_delta_and_first_monitoring_boundary(self):
        for case in self.fixture['cases']:
            for spot in (95.0, 100.0, 105.0):
                analytic = price_delta(case, self.market, 20, 24, spot=spot)[1]
                for bump in (0.005, 0.0025):
                    upper = price_delta(case, self.market, 20, 24, spot=spot + bump)[0]
                    lower = price_delta(case, self.market, 20, 24, spot=spot - bump)[0]
                    self.assertAlmostEqual(analytic, (upper - lower) / (2 * bump), delta=3e-8)
            full = price_delta(case, self.market, 20, 24)
            omitted = price_delta(case, self.market, 20, 24, include_first_boundary=False)
            self.assertEqual(full[0], omitted[0])
            self.assertGreater(full[1] - omitted[1], 0.2)


if __name__ == '__main__':
    unittest.main()
