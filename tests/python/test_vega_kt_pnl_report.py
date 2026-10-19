"""Real replay integration: dated IV alignment, opening books and exact ledger."""
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
from rust_pricing_backtest.validation import validate_document


class PnlReportTests(unittest.TestCase):
    def test_actual_ledger_signed_positions_costs_and_daily_roll(self):
        c,h,old=six_interval_history('arithmetic_asian');first=old[0];history=[first]
        for i in (1,2):
            at=first.at+timedelta(days=i);spot=100.+i*.1
            payloads={}
            nodes=[-.2-i*.05,-.1,0.,.1,.2+i*.05]
            for cid,raw in first.pricing_requests.items():
                p=advance(json.loads(raw),i,spot)
                p['model']['reporting_iv_basis']['log_forward_moneyness_nodes']=nodes
                p['model']['reporting_iv_basis']['implied_volatilities']=implied_surface(p).ravel().tolist()
                p['risk']['vega_kt']['log_forward_moneyness_nodes']=nodes
                payloads[cid]=json.dumps(p)
            history.append(replace(first,snapshot_id=f'roll-{i}',at=at,spot=Quote(spot,spot,at,at),
                basis=replace(first.basis,valuation_date=at.date(),log_moneyness=tuple(nodes)),surface_available_at=at,
                pricing_requests=payloads,settlement_fixings={},option_quotes={'hedge':Quote(5.99,6.01,at,at)}))
        history.append(old[-1])
        provider=RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',variance_extension='all_rows',worker_threads=4)
        result=run_backtest(BacktestConfig(strategy='vega_kt'),history,[c,h],[Position('target',-1.,5.,entry_fee=.1)],provider,stop_after=3)
        before=result.to_json();report=explain_backtest(result,worker_threads=4)
        self.assertEqual(report['status'],'complete');self.assertEqual(report['covered_intervals'],2)
        self.assertEqual(result.to_json(),before)
        self.assertEqual(report['source_result_hash'],fingerprint(result))
        for row in report['rows']:
            self.assertLess(abs(row['reconciliation_error']),1e-9)
            self.assertLess(abs(row['cumulative_reconciliation_error']),1e-9)
        self.assertTrue(any(abs(g.get('vega_pnl',0.))>0 for r in report['rows'] for g in r['groups'].values()))
        self.assertTrue(any(len(e['joint_risk']['instrument_order'])==2 for e in report['opening_joint_risks']))
        self.assertEqual(validate_document(json.loads(before))['status'],'valid')
        self.assertEqual(BacktestResult.from_json(before).to_json(),before)
        out=ROOT/'benchmark-results/vega-kt-portfolio-market';out.mkdir(parents=True,exist_ok=True)
        (out/'pnl-replay-result.json').write_text(before)
        (out/'pnl-replay-report.json').write_text(json.dumps(report,allow_nan=False)+'\n')

if __name__=='__main__':unittest.main()
