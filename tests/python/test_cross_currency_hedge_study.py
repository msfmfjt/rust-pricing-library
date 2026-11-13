"""Independent sizing-cycle variance and paired rounding diagnostics."""
from math import sqrt
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
from cross_currency_autocall_hedge_study import summarize


class NestedHedgeSummaryTests(unittest.TestCase):
    def test_fixed_vs_full_variance_and_paired_rounding(self):
        cycles = []
        for i, (ideal, error, lots) in enumerate([(-2., 1., 4), (0., 2., 5), (2., 3., 4)]):
            row = dict(stock=20, expiry="2027-04-10", theoretical=4.4, contracts=lots,
                       ideal=ideal, actual=ideal+1.5, ideal_se=error, actual_se=error)
            cycles.append(dict(seed=i, rows=[row]))
        result, = summarize(cycles)
        self.assertAlmostEqual(result["ideal"]["fixed_se"], sqrt(14/3))
        self.assertAlmostEqual(result["ideal"]["full_se"], 2.)
        self.assertAlmostEqual(result["ideal"]["mean_se"], 2./sqrt(3))
        self.assertEqual(result["ideal"]["negative"], 1)
        self.assertEqual(result["contracts"], {4:2, 5:1})
        self.assertEqual(result["rounding"], dict(mean=1.5, mean_se=0.))
        with self.assertRaises(ValueError):
            summarize(cycles[:1])
        with self.assertRaises(ValueError):
            summarize(cycles + [cycles[0]])
        with self.assertRaises(ValueError):
            summarize([cycles[0], dict(seed=4, rows=[])])
