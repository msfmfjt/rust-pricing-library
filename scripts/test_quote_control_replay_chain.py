from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import seal_quote_control_replay_chain as chain
from rust_pricing_backtest.domain import BacktestError


def fixture(n=3, total=252):
    prior = dict(status='passed',failed=0,committed_snapshots=n,planned_snapshots=total,
        sampling_policy='rqmc_quote_control_512x24',original_precision_limit=10.,
        held=[.1]*n,proposed=[.2]*n,cumulative_native_product_days=30*n,cumulative_independent_checks=1024*n)
    previous = dict(status='partial',failure=None,processed_snapshots=n,planned_snapshots=total,
        sampling_policy=prior['sampling_policy'],held=list(prior['held']),proposed=list(prior['proposed']))
    current = dict(status='completed' if n+1==total else 'partial',failure=None,
        preceding_snapshots=n,processed_snapshots=n+1,planned_snapshots=total,
        annual_completed=n+1==total,sampling_policy=prior['sampling_policy'],original_sampling_error_limit=10.,
        held=[*prior['held'],.3],proposed=[*prior['proposed'],.4],fresh_native_provider=True,
        retained_native_audits_imported=False,checkpoint_prefix_preserved=True)
    return prior,previous,current


class ReplayChainTests(unittest.TestCase):
    def test_arbitrary_next_day_and_true_final_day_are_allowed(self):
        for n in (3,10,251):
            with self.subTest(n=n):self.assertEqual(chain.check_link(*fixture(n)),n)

    def test_skips_and_duplicate_days_are_rejected(self):
        for field,value in [('preceding_snapshots',2),('preceding_snapshots',4),('processed_snapshots',3),('processed_snapshots',5)]:
            p,old,new=fixture();new[field]=value
            with self.subTest(field=field,value=value),self.assertRaisesRegex(BacktestError,'replay_chain_sequence'):
                chain.check_link(p,old,new)

    def test_mismatched_preceding_schedule_is_rejected(self):
        p,old,new=fixture();old['planned_snapshots']=251
        with self.assertRaisesRegex(BacktestError,'replay_chain_sequence'):chain.check_link(p,old,new)
        p,old,new=fixture();old['processed_snapshots']=2
        with self.assertRaisesRegex(BacktestError,'replay_chain_sequence'):chain.check_link(p,old,new)

    def test_mutated_history_and_invalid_precision_are_rejected(self):
        for field in ('held','proposed'):
            for index,value in [(0,.9),(-1,10.01),(-1,-1.),(-1,float('nan'))]:
                p,old,new=fixture();new[field][index]=value
                with self.subTest(field=field,index=index,value=value),self.assertRaisesRegex(BacktestError,'replay_chain_history'):
                    chain.check_link(p,old,new)

    def test_changed_estimator_or_relaxed_limit_is_rejected(self):
        for field,value in [('sampling_policy','different'),('original_sampling_error_limit',11.)]:
            p,old,new=fixture();new[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(BacktestError,'replay_chain_policy'):chain.check_link(p,old,new)

    def test_failed_or_falsely_complete_day_is_rejected(self):
        for changes in [dict(status='failed',failure={'code':'native_failure'}),dict(annual_completed=True),dict(status='completed')]:
            p,old,new=fixture();new.update(changes)
            with self.subTest(changes=changes),self.assertRaisesRegex(BacktestError,'replay_chain_status'):chain.check_link(p,old,new)
        p,old,new=fixture(251);new['annual_completed']=False
        with self.assertRaisesRegex(BacktestError,'replay_chain_status'):chain.check_link(p,old,new)

    def test_imported_audits_or_unpreserved_prefix_cannot_enter_chain(self):
        for field,value in [('fresh_native_provider',False),('retained_native_audits_imported',True),('checkpoint_prefix_preserved',False)]:
            p,old,new=fixture();new[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(BacktestError,'replay_chain_native'):chain.check_link(p,old,new)

    def test_inflated_baseline_counts_are_rejected(self):
        for field in ('cumulative_native_product_days','cumulative_independent_checks'):
            p,old,new=fixture();p[field]+=1
            with self.subTest(field=field),self.assertRaisesRegex(BacktestError,'replay_chain_counts'):chain.check_link(p,old,new)

    def test_transitively_bound_evidence_and_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(chain,'ROOT',Path(temp)):
            file=Path(temp)/'bound';file.write_text('original')
            for section in ('source_sha256','evidence_sha256'):
                record={'source_sha256':{},'evidence_sha256':{}};record[section]['bound']=chain.sha(file)
                chain.verify_manifest(record);file.write_text('changed')
                with self.subTest(section=section),self.assertRaisesRegex(BacktestError,'replay_chain_integrity'):
                    chain.verify_manifest(record)
                file.write_text('original')


if __name__=='__main__':unittest.main()
