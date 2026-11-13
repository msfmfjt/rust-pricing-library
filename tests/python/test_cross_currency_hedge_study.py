"""Independent sizing-cycle variance and paired rounding diagnostics."""
from contextlib import redirect_stdout
from io import StringIO
import json
from math import sqrt
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
from cross_currency_autocall_hedge_study import main, summarize


class NestedHedgeSummaryTests(unittest.TestCase):
    def test_independent_batches_and_reference_exclusion(self):
        def sample(index, worker_threads):
            row = dict(stock=20, expiry="2027-04-10", theoretical=0, contracts=0,
                       ideal=index, actual=index, ideal_se=1, actual_se=1)
            return dict(index=index, seed=index, rows=[row])

        batches = [([], [0, 1, 2], 1.5), (["--first-cycle", "17"], [17, 18], 17.5)]
        for options, indices, mean in batches:
            with self.subTest(options=options):
                output = StringIO()
                with (
                    redirect_stdout(output),
                    patch.object(sys, "argv", ["study", "--cycles", "2", *options]),
                    patch("cross_currency_autocall_hedge_study.run_cycle", side_effect=sample),
                ):
                    main()
                *cycles, report = map(json.loads, output.getvalue().splitlines())
                self.assertEqual([cycle["index"] for cycle in cycles], indices)
                self.assertEqual(report["cycles"], 2)
                self.assertEqual(report["summary"][0]["ideal"]["mean"], mean)

    def test_fixed_vs_full_variance_and_paired_rounding(self):
        cycles = []
        for i, (ideal, error, lots) in enumerate([(-2., 1., 4), (0., 2., 5), (2., 3., 4)]):
            row = dict(stock=20, expiry="2027-04-10", theoretical=4.4, contracts=lots,
                       ideal=ideal, actual=ideal+(lots-4.5)*3., ideal_se=error, actual_se=error)
            cycles.append(dict(seed=i, rows=[row]))
        result, = summarize(cycles)
        self.assertAlmostEqual(result["ideal"]["fixed_se"], sqrt(14/3))
        self.assertAlmostEqual(result["ideal"]["full_se"], 2.)
        self.assertAlmostEqual(result["ideal"]["mean_se"], 2./sqrt(3))
        self.assertEqual(result["ideal"]["negative"], 1)
        self.assertEqual(result["contracts"], {4:2, 5:1})
        self.assertEqual(result["rounding"], dict(mean=-.5, mean_se=1., mean_absolute=1.5))
        with self.assertRaises(ValueError):
            summarize(cycles[:1])
        with self.assertRaises(ValueError):
            summarize(cycles + [cycles[0]])
        with self.assertRaises(ValueError):
            summarize([cycles[0], dict(seed=4, rows=[])])
