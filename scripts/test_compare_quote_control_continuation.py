"""Regression against the retained, accepted two-day numerical evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare_quote_control_continuation as comparison
from continue_quote_control_replay import sha
from check_portfolio_holdout import sealed
from check_portfolio_market import sources, write
from rust_pricing_backtest.domain import BacktestError, fingerprint


class DailyComparisonRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prior = comparison.ROOT/'benchmark-results/vega-kt-quote-control-provider-24'
        # Only a test adapter around retained numerical evidence, not a new replay claim.
        result = self.root/'result.json.gz'
        result.hardlink_to(self.prior/'pilot/resumed/result.json.gz')
        self.record = dict(status='partial', failure=None, source_sha256=sources(),
            acceptance_sha256=sha(comparison.ACCEPTANCE), preceding_snapshots=1, processed_snapshots=2,
            attempted_snapshot_id='accepted-benchmark-001', annual_completed=False,
            files={'result.json.gz':{'sha256':sha(result)}})
        self.save_record(self.record)

    def save_record(self, record):
        value = dict(record); value['report_hash'] = fingerprint(value)
        write(self.root/'report.json', value)

    def invoke(self, oracle=None):
        with redirect_stdout(io.StringIO()):
            return comparison.run(self.root, oracle or self.prior/'oracle-1.json', self.root/'comparison.json')

    def test_new_daily_checker_exactly_matches_accepted_old_date(self):
        actual = self.invoke()
        expected = [c for c in sealed(self.prior/'comparison.json')['checks'] if c['day']==1]
        self.assertEqual(actual['checks'], expected)
        self.assertEqual(actual['status'], 'passed'); self.assertEqual(actual['independent_checks'], 1024)
        self.assertEqual(actual['independent_native_panels'], 30); self.assertFalse(actual['earlier_dates_recounted'])

    def test_changed_bundle_hash_is_rejected_before_numerical_comparison(self):
        changed = deepcopy(self.record); changed['files']['result.json.gz']['sha256'] = 'changed'
        self.save_record(changed)
        with self.assertRaisesRegex(BacktestError, 'saved bundle changed'): self.invoke()
        self.assertFalse((self.root/'comparison.json').exists())

    def test_oracle_for_a_different_dated_request_is_rejected(self):
        with self.assertRaisesRegex(BacktestError, 'independent request differs'):
            self.invoke(self.prior/'oracle-0.json')
        self.assertFalse((self.root/'comparison.json').exists())


if __name__ == '__main__': unittest.main()
