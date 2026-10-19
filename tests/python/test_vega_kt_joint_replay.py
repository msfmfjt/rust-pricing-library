"""Actual Gamma replay: joint gates, dated attribution and atomic resume."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from test_vega_kt_extension_provider import six_interval_history
from test_vega_kt_multi_step_provider import advance
from rust_pricing_backtest import RustPricingProvider,BacktestConfig,Position,Quote,run_backtest,BacktestResult
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.portfolio_market_risk import implied_surface,explain_backtest
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document


def fixture():
    target,hedge,original=six_interval_history('arithmetic_asian');first=original[0]
    at=first.at+timedelta(days=1);spot=100.1;nodes=[-.25,-.1,0.,.1,.25];requests={}
    for cid,raw in first.pricing_requests.items():
        p=advance(json.loads(raw),1,spot)
        p['model']['reporting_iv_basis']['log_forward_moneyness_nodes']=nodes
        p['model']['reporting_iv_basis']['implied_volatilities']=implied_surface(p).ravel().tolist()
        p['risk']['vega_kt']['log_forward_moneyness_nodes']=nodes
        requests[cid]=json.dumps(p)
    second=replace(first,snapshot_id='roll-1',at=at,spot=Quote(spot,spot,at,at),
        basis=replace(first.basis,valuation_date=at.date(),log_moneyness=tuple(nodes)),surface_available_at=at,
        pricing_requests=requests,settlement_fixings={},option_quotes={'hedge':Quote(5.99,6.01,at,at)})
    return dict(config=BacktestConfig(strategy='vega_kt',max_joint_bucket_standard_error=10.),
        history=[first,second,original[-1]],contracts=[target,hedge],portfolio=[Position('target',-1.,5.,entry_fee=.1)])


def provider():
    return RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',variance_extension='all_rows',worker_threads=4)


class NativeJointReplayTests(unittest.TestCase):
    def test_actual_ledger_joint_covariance_saved_pnl_and_resume(self):
        args=fixture();p=provider()
        result=run_backtest(**args,provider=p)
        legacy=run_backtest(**(args|{'config':replace(args['config'],max_joint_bucket_standard_error=None)}),provider=p)
        self.assertEqual(result.ledger,legacy.ledger);self.assertEqual(result.orders,legacy.orders)
        self.assertEqual(result.fills,legacy.fills);self.assertEqual(result.pnl_components,legacy.pnl_components)
        self.assertEqual(result.summary['joint_market']['covered_intervals'],1)
        self.assertEqual(result.greek_pnl[1]['market_iv']['status'],'available')
        self.assertEqual(result.greek_pnl[-1]['market_iv']['status'],'unavailable')
        for row in result.risks[:-1]:
            joint=row['joint_market']['report'];q=len(joint['coordinate_order'])
            covariance=np.array(joint['market_mean_covariance'])
            self.assertGreater(np.max(abs(covariance[:q,q:])),0.)
            self.assertLessEqual(row['joint_market']['held']['max_bucket_standard_error'],10.)
        partial=run_backtest(**args,provider=p,stop_after=1)
        resumed=run_backtest(**args,provider=provider(),checkpoint=partial.checkpoint)
        self.assertEqual(resumed.to_json(),result.to_json())
        self.assertEqual(BacktestResult.from_json(result.to_json()).to_json(),result.to_json())
        for document in [json.loads(result.to_json()),json.loads(input_json(result)),result.checkpoint]:
            self.assertEqual(validate_document(document)['status'],'valid')
        # Independent existing report API recomputes opening risk. Its monetary
        # explanation must agree with the new saved-opening-report integration.
        detached=explain_backtest(legacy,worker_threads=4)
        for old,new in zip(detached['rows'],result.greek_pnl):
            new=new['market_iv'];self.assertEqual(old['status'],new['status'])
            for key in ('available_greek_pnl','unexplained_pnl','non_asset_pnl','reconciliation_error'):
                self.assertAlmostEqual(old[key],new[key],places=10)
        out=ROOT/'benchmark-results/vega-kt-joint-replay';out.mkdir(parents=True,exist_ok=True)
        result.write(out/'api');partial.write(out/'partial');legacy.write(out/'disabled')
        (out/'detached-pnl.json').write_text(json.dumps(detached,allow_nan=False)+'\n')
        (out/'native-validation.json').write_text(json.dumps(dict(status='passed',identity_hash=result.manifest['identity_hash'],
            joint_pnl_covered_intervals=1,expiry_interval_explicitly_unavailable=True,
            legacy_ledger_and_orders_exact=True,resume_byte_identical=True,
            result_hash=fingerprint(result),input_hash=fingerprint(json.loads(input_json(result)))) ,indent=2)+'\n')


if __name__=='__main__':unittest.main()
