"""Observation must preserve real calls and never manufacture ledger evidence."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtesting"))
from rust_pricing_backtest.domain import fingerprint
from provider_evidence import capture_provider_evidence


@dataclass(frozen=True)
class ObservedRisk:
    provenance: str
    price: float = 5.


class CountingProvider:
    """A one-snapshot cache with separate public-call and computation counters."""

    def __init__(self):
        self.calls = []
        self.computations = 0
        self.audits = 0
        self.active = None
        self.cache = {}
        self.failure = False

    def _evaluate(self, snapshot, contract):
        if self.failure:
            raise ValueError("native failure")
        s, c = fingerprint(snapshot), fingerprint(contract)
        if s != self.active:
            self.cache = {}
            self.active = s
        if c not in self.cache:
            self.computations += 1
            risk = ObservedRisk(fingerprint((s, c)))
            self.cache[c] = risk, dict(
                provenance=dict(snapshot=s, contract=c), risk_provenance=risk.provenance,
                nested=dict(values=[1., 2.]))
        return self.cache[c]

    def evaluate(self, snapshot, contract):
        self.calls.append((snapshot, contract))
        return self._evaluate(snapshot, contract)[0]

    def gamma_audit(self, snapshot, contract):
        self.audits += 1
        return self._evaluate(snapshot, contract)[1]


class ProviderEvidenceTests(unittest.TestCase):
    def setUp(self):
        mode = patch.dict(os.environ, PRICING_TEST_ORIGINAL_EVIDENCE="0")
        mode.start()
        self.addCleanup(mode.stop)

    def test_original_mode_preserves_the_unobserved_native_call_sequence(self):
        p = CountingProvider()
        original = p.evaluate
        inputs = [(dict(day=day), dict(contract=contract))
                  for day in (0, 1) for contract in ("target", "hedge")]
        with patch.dict(os.environ, PRICING_TEST_ORIGINAL_EVIDENCE="1"):
            with capture_provider_evidence(p) as evidence:
                self.assertEqual(p.evaluate, original)
                for _ in range(4):
                    for snapshot, contract in inputs:
                        p.evaluate(snapshot, contract)
        self.assertEqual(p.computations, 16)
        self.assertEqual(p.audits, 0)
        for snapshot, contract in inputs:
            evidence.get(snapshot, contract)
        evidence.assert_complete()
        self.assertEqual(p.calls, inputs * 5)
        self.assertEqual(p.computations, 20)
        self.assertEqual(p.audits, 4)
        self.assertEqual(p.evaluate, original)

    def test_every_pricing_call_and_snapshot_cache_transition_still_execute(self):
        p = CountingProvider()
        inputs = [(dict(day=day), dict(contract=contract))
                  for day in (0, 1) for contract in ("target", "hedge")]
        original = p.evaluate
        with capture_provider_evidence(p) as evidence:
            for _ in range(4):
                for snapshot, contract in inputs:
                    p.evaluate(snapshot, contract)
        self.assertEqual(p.evaluate, original)
        self.assertEqual(p.calls, inputs * 4)
        self.assertEqual(p.computations, 16)
        self.assertEqual(p.audits, 4)
        for snapshot, contract in inputs:
            risk, audit = evidence.get(snapshot, contract)
            self.assertEqual(risk.provenance, audit['risk_provenance'])
        evidence.assert_complete()
        self.assertEqual(p.computations, 16)  # Reading evidence cannot reprice.
        p.evaluate(*inputs[0])
        self.assertEqual(p.computations, 17)  # Independent replay remains cold.

    def test_captured_audits_and_returned_copies_are_detached(self):
        p = CountingProvider(); s, c = dict(day=0), dict(contract="target")
        with capture_provider_evidence(p) as evidence:
            p.evaluate(s, c)
        p.gamma_audit(s, c)['nested']['values'][0] = 99.
        _, audit = evidence.get(s, c)
        self.assertEqual(audit['nested']['values'], [1., 2.])
        audit['nested']['values'][1] = 88.
        self.assertEqual(evidence.get(s, c)[1]['nested']['values'], [1., 2.])
        evidence.assert_complete()

    def test_missing_changed_or_unconsumed_inputs_fail(self):
        p = CountingProvider(); s, c = dict(day=0), dict(contract="target")
        with capture_provider_evidence(p) as evidence:
            p.evaluate(s, c)
        with self.assertRaisesRegex(AssertionError, "missing from evidence"):
            evidence.assert_complete()
        for changed in [(dict(day=1), c), (s, dict(contract="hedge"))]:
            with self.assertRaisesRegex(AssertionError, "did not price"):
                evidence.get(*changed)
        evidence.get(s, c)
        evidence.assert_complete()

    def test_failed_native_call_is_not_recorded_and_observer_is_restored(self):
        p = CountingProvider(); p.failure = True
        original = p.evaluate
        with self.assertRaisesRegex(ValueError, "native failure"):
            with capture_provider_evidence(p) as evidence:
                p.evaluate(dict(day=0), dict(contract="target"))
        self.assertEqual(p.evaluate, original)
        self.assertEqual(p.audits, 0)
        with self.assertRaisesRegex(AssertionError, "did not price"):
            evidence.get(dict(day=0), dict(contract="target"))

    def test_wrong_audit_inputs_or_risk_are_rejected(self):
        for field in ("snapshot", "contract", "risk_provenance"):
            p = CountingProvider(); s, c = dict(day=0), dict(contract="target")
            p.evaluate(s, c)
            audit = p.gamma_audit(s, c)
            if field == "risk_provenance":
                audit[field] = "wrong"
            else:
                audit['provenance'][field] = "wrong"
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, "different"):
                with capture_provider_evidence(p):
                    p.evaluate(s, c)

    def test_repeated_input_with_different_native_risk_is_rejected(self):
        p = CountingProvider(); s, c = dict(day=0), dict(contract="target")
        with capture_provider_evidence(p):
            p.evaluate(s, c)
            key = fingerprint(c)
            risk, audit = p.cache[key]
            p.cache[key] = ObservedRisk(risk.provenance, 6.), audit
            with self.assertRaisesRegex(AssertionError, "repeated native pricing changed"):
                p.evaluate(s, c)


if __name__ == "__main__":
    unittest.main()
