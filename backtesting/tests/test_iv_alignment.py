from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, Risk, RiskBasis, run_backtest
from rust_pricing_backtest import iv_alignment
from test_backtest import START, EXPIRY, snapshot


DATES = (START+timedelta(days=7), START+timedelta(days=21))


def observation(i, spot=100., axis=(-.2, 0., .2), dates=DATES, **kwargs):
    s = snapshot(i, spot=spot, **kwargs)
    return replace(s, basis=RiskBasis(s.at.date(), dates, axis))


class SurfaceProvider:
    identity = "fixed_strike_iv_audit_fixture_v1"

    def __init__(self, shift=.01):
        self.shift = shift

    def evaluate(self, s, c):
        # sigma(T,K)=.2+.001*(T-START)+.1*log(K/100)+daily_shift.
        # Its current-k samples change when the forward changes, even if the
        # fixed-strike surface is unchanged. Later bucket values deliberately
        # differ: interval diagnostics must use the opening values.
        day = int(s.snapshot_id[1:])
        vols = tuple(.2+.001*(t-START).days+.1*(k+math.log(s.spot.mid/100.))+day*self.shift
                     for t in s.basis.maturity_dates for k in s.basis.log_moneyness)
        return Risk(5., 0., tuple((j+1)*(100. if day else 1.) for j in range(s.basis.size)), s.basis.basis_id,
                    reporting_iv=vols, reporting_forwards=(s.spot.mid,)*len(s.basis.maturity_dates))


class IvAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(strategy="unhedged")
        self.contracts = [Contract("target", EXPIRY, 100., multiplier=10.)]
        self.portfolio = [Position("target", -2., 5.)]
        self.history = [observation(0), observation(1, spot=110., axis=(-.4, 0., .4))]
        self.provider = SurfaceProvider()

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=self.contracts,
                    portfolio=self.portfolio, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def test_fixed_strike_shift_weights_opening_risk_and_units(self):
        result = self.run_case()
        row = result.iv_alignment[1]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["matched_nodes"], 6)
        for node in row["nodes"]:
            self.assertAlmostEqual(node["delta_iv"], .01)
            self.assertAlmostEqual(node["current_k"], node["opening_k"]+math.log(100./110.))
            self.assertAlmostEqual(node["canonical_strike"], 100.*math.exp(node["opening_k"]))
            self.assertAlmostEqual(sum(node["weights"]), 1.)
            current_iv = result.risks[1]["instruments"]["target"]["reporting_iv"]
            self.assertAlmostEqual(node["current_iv"], sum(current_iv[j]*w for j,w in zip(node["current_indices"],node["weights"])))
        group = row["groups"]["target"]
        self.assertEqual(group["opening_raw_buckets"], [-20., -40., -60., -80., -100., -120.])
        self.assertAlmostEqual(group["covered_bucket_dot"], -4.2)
        self.assertAlmostEqual(group["complete_bucket_dot"], -4.2)
        self.assertEqual(group["absolute_risk_coverage"], 1.)
        self.assertFalse(row["pnl_adopted"])
        self.assertEqual(result.greek_pnl[1]["available_greek_pnl"], 0.)
        self.assertIsNone(result.greek_pnl[1]["groups"]["target"]["vega_pnl"])
        self.assertEqual(result.ledger[1]["daily_pnl"], 0.)

    def test_forward_move_of_static_strike_surface_is_not_vol_change(self):
        result = self.run_case(provider=SurfaceProvider(shift=0.))
        for node in result.iv_alignment[1]["nodes"]:
            self.assertAlmostEqual(node["delta_iv"], 0., places=14)
        first, last = [r["instruments"]["target"]["reporting_iv"] for r in result.risks]
        self.assertGreater(abs(last[1]-first[1]), .009)
        self.assertAlmostEqual(result.iv_alignment[1]["groups"]["target"]["complete_bucket_dot"], 0.)

    def test_relative_tenor_roll_does_not_match_different_absolute_dates(self):
        result = self.run_case(history=[observation(0), observation(1, dates=tuple(t+timedelta(days=1) for t in DATES))])
        row = result.iv_alignment[1]
        self.assertEqual({n["status"] for n in row["nodes"]}, {"missing_absolute_maturity"})
        self.assertEqual(row["groups"]["target"]["uncovered_absolute_risk"], 420.)
        self.assertIsNone(row["groups"]["target"]["complete_bucket_dot"])

    def test_expired_nodes_are_missing_and_remaining_dates_align(self):
        result = self.run_case(history=[observation(0, dates=(START+timedelta(days=1),EXPIRY)),
                                       observation(1, dates=(START+timedelta(days=2),EXPIRY))])
        row = result.iv_alignment[1]
        self.assertEqual([n["status"] for n in row["nodes"]], ["expired_maturity"]*3+["matched"]*3)
        self.assertEqual(row["groups"]["target"]["uncovered_absolute_risk"], 120.)
        self.assertAlmostEqual(row["groups"]["target"]["absolute_risk_coverage"], 5./7.)
        self.assertTrue(all(n["delta_iv"] is None for n in row["nodes"][:3]))

    def test_strike_domain_is_not_clamped_or_extrapolated(self):
        result = self.run_case(history=[observation(0), observation(1, spot=200.)])
        row = result.iv_alignment[1]
        self.assertEqual({n["status"] for n in row["nodes"]}, {"outside_strike_domain"})
        self.assertEqual(row["groups"]["target"]["covered_bucket_dot"], 0.)
        self.assertEqual(row["groups"]["target"]["absolute_risk_coverage"], 0.)
        self.assertIsNone(row["groups"]["target"]["complete_bucket_dot"])

    def test_exact_node_and_boundary_match_have_unit_weights(self):
        result = self.run_case(history=[observation(0),observation(1)])
        for j,node in enumerate(result.iv_alignment[1]["nodes"]):
            self.assertEqual(node["current_indices"], [j])
            self.assertEqual(node["weights"], [1.])
            self.assertEqual(node["current_k"], node["opening_k"])

    def test_offsetting_positions_do_not_hide_missing_gross_risk(self):
        result = self.run_case(contracts=[*self.contracts,replace(self.contracts[0],contract_id="opposite")],
                               portfolio=[Position("target",1.,5.),Position("opposite",-1.,5.)],
                               history=[observation(0),observation(1,spot=200.)])
        group = result.iv_alignment[1]["groups"]["target"]
        self.assertEqual(group["opening_raw_buckets"], [0.]*6)
        self.assertEqual(group["uncovered_absolute_risk"], 420.)
        self.assertEqual(group["covered_bucket_dot"], 0.)
        self.assertIsNone(group["complete_bucket_dot"])

    def test_absent_or_unsupported_surfaces_remain_explicit(self):
        class Missing(SurfaceProvider):
            def evaluate(self,s,c):
                return replace(super().evaluate(s,c), reporting_iv=None, reporting_forwards=None)
        result = self.run_case(provider=Missing())
        self.assertEqual({n["status"] for n in result.iv_alignment[1]["nodes"]}, {"missing_opening_surface"})
        self.assertIsNone(result.iv_alignment[1]["groups"]["target"]["complete_bucket_dot"])
        class MissingCurrent(SurfaceProvider):
            def evaluate(self,s,c):
                r = super().evaluate(s,c)
                return replace(r,reporting_iv=None,reporting_forwards=None) if s.snapshot_id == "s1" else r
        current = self.run_case(provider=MissingCurrent())
        self.assertEqual({n["status"] for n in current.iv_alignment[1]["nodes"]}, {"missing_current_surface"})
        history = [replace(s,basis=replace(s.basis,coordinate="physical_spot_iv")) for s in self.history]
        result = self.run_case(history=history)
        self.assertEqual(result.iv_alignment[1]["status"], "unsupported_coordinate")
        self.assertEqual(result.iv_alignment[1]["nodes"], [])
        self.assertFalse(result.iv_alignment[1]["pnl_adopted"])

    def test_reporting_surface_input_shape_and_values(self):
        r = self.provider.evaluate(self.history[0],self.contracts[0])
        for change in ({"reporting_iv":None},{"reporting_iv":(.2,)},{"reporting_forwards":(100.,)},
                       {"reporting_iv":(0.,)*6},{"reporting_iv":(float("nan"),)*6},{"reporting_forwards":(-1.,100.)}):
            with self.subTest(change=change),self.assertRaisesRegex(BacktestError,"invalid_reporting_surface"):
                replace(r,**change)

    def test_inconsistent_surface_after_fill_rolls_back_and_resumes(self):
        args = dict(config=BacktestConfig(ridge=0.,end_policy="mark_to_market"),contracts=[*self.contracts,replace(self.contracts[0],contract_id="hedge",strike=101.)],
                    history=[replace(s,option_quotes={"hedge":replace(s.spot,bid=5.,ask=5.)}) for s in self.history])
        whole = self.run_case(**args)
        self.assertTrue(whole.fills)
        original = iv_alignment.audit
        for field in ("reporting_iv","reporting_forwards"):
            def corrupted(current,*rest):
                if current["at"] == self.history[1].at.isoformat():
                    current = deepcopy(current)
                    current["instruments"]["hedge"][field][0] += .01
                return original(current,*rest)
            with patch("rust_pricing_backtest.iv_alignment.audit",side_effect=corrupted):
                failed = self.run_case(**args,on_error="return")
            self.assertEqual(failed.failure["code"],"iv_surface_mismatch")
            self.assertEqual(failed.fills,())
            self.assertEqual(failed.checkpoint,self.run_case(**args,stop_after=1).checkpoint)
            self.assertEqual(self.run_case(**args,checkpoint=failed.checkpoint).to_json(),whole.to_json())

    def test_future_surface_does_not_change_prior_alignment(self):
        class Later(SurfaceProvider):
            def evaluate(self,s,c):
                r = super().evaluate(s,c)
                return replace(r,reporting_iv=tuple(x+.1 for x in r.reporting_iv)) if s.snapshot_id == "s2" else r
        history = [*self.history,observation(2,axis=(-.5,0.,.5))]
        first,changed = self.run_case(history=history),self.run_case(history=history,provider=Later())
        self.assertEqual(first.iv_alignment[:2],changed.iv_alignment[:2])
        self.assertNotEqual(first.iv_alignment[2],changed.iv_alignment[2])

    def test_export_resume_legacy_and_timing(self):
        whole,partial = self.run_case(),self.run_case(stop_after=1)
        self.assertEqual(self.run_case(checkpoint=json.loads(json.dumps(partial.checkpoint))).to_json(),whole.to_json())
        with tempfile.TemporaryDirectory() as directory:
            whole.write(directory)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))),11)
            with Path(directory,"iv_alignment.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(json.loads(rows[1]["nodes"]),whole.iv_alignment[1]["nodes"])
            self.assertEqual(BacktestResult.from_json(Path(directory,"result.json").read_text()).to_json(),whole.to_json())
        for version in range(1,9):
            old = json.loads(whole.to_json())
            old["manifest"]["schema_version"] = version
            del old["iv_alignment"]
            self.assertEqual(BacktestResult.from_json(json.dumps(old)).iv_alignment,())
        with self.assertRaisesRegex(BacktestError,"checkpoint_mismatch"):
            self.run_case(checkpoint={**partial.checkpoint,"schema_version":6})
        with self.assertRaisesRegex(BacktestError,"iv_alignment_timing"):
            iv_alignment.audit(whole.risks[1],whole.risks[0],{**whole.cost_basis[0],"at":"wrong"})

    def test_overflow_in_strike_reconstruction_is_typed_and_atomic(self):
        history = [observation(i,axis=(1000.,1001.)) for i in range(2)]
        failed = self.run_case(history=history,on_error="return")
        self.assertEqual(failed.failure["code"],"nonfinite_iv_alignment")
        self.assertEqual(failed.checkpoint,self.run_case(history=history,stop_after=1).checkpoint)


if __name__ == "__main__":
    unittest.main()
