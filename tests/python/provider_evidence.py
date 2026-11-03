"""Observe real ledger pricing while its same-snapshot audit is still cached.

This helper never answers a pricing call. Every call reaches the original
provider, including all repeated strategy evaluations. Only the later evidence
export uses the detached observation, avoiding an otherwise redundant history
reprice. Partial/resumed backtests run after the observer has been removed.
Set PRICING_TEST_ORIGINAL_EVIDENCE=1 to retain the original evidence-repricing
sequence when investigating allocation history or long-lived native failures.
"""

from contextlib import contextmanager
from copy import deepcopy
import os
from unittest.mock import patch

from rust_pricing_backtest.domain import fingerprint


class ProviderEvidence:
    def __init__(self):
        self._records = {}
        self._read = set()

    @staticmethod
    def _key(snapshot, contract):
        return fingerprint(snapshot), fingerprint(contract)

    def observe(self, provider, evaluate, snapshot, contract):
        # Do not consult the observations before evaluating: this is not a
        # memoizer, and the provider's real validation/cache lifetime is tested.
        risk = evaluate(snapshot, contract)
        key = self._key(snapshot, contract)
        if key in self._records:
            if risk != self._records[key][0]:
                raise AssertionError("repeated native pricing changed for identical inputs")
        else:
            audit = provider.gamma_audit(snapshot, contract)
            provenance = audit["provenance"]
            if (provenance["snapshot"], provenance["contract"]) != key:
                raise AssertionError("observed Gamma audit belongs to different inputs")
            if audit["risk_provenance"] != risk.provenance:
                raise AssertionError("observed Gamma audit belongs to a different risk")
            self._records[key] = deepcopy((risk, audit))
        return risk

    def get(self, snapshot, contract):
        key = self._key(snapshot, contract)
        if key not in self._records:
            raise AssertionError("ledger did not price the requested snapshot/contract")
        self._read.add(key)
        return deepcopy(self._records[key])

    def assert_complete(self):
        if self._read != self._records.keys():
            raise AssertionError("ledger priced snapshot/contract pairs missing from evidence")
        # Exported audits are already detached. Do not retain another complete
        # copy while the independent partial/resume checks perform native work.
        self._records.clear()
        self._read.clear()


class OriginalProviderEvidence:
    """Keep the original native call sequence for memory/lifetime diagnostics."""

    def __init__(self, provider):
        self.provider = provider

    def get(self, snapshot, contract):
        risk = self.provider.evaluate(snapshot, contract)
        return risk, self.provider.gamma_audit(snapshot, contract)

    def assert_complete(self):
        pass


@contextmanager
def capture_provider_evidence(provider):
    if os.environ.get("PRICING_TEST_ORIGINAL_EVIDENCE") == "1":
        print("wheel-provider evidence=original-reprice", flush=True)
        yield OriginalProviderEvidence(provider)
        return
    print("wheel-provider evidence=observed-native-calls", flush=True)
    evidence = ProviderEvidence()
    evaluate = provider.evaluate

    def observed(snapshot, contract):
        return evidence.observe(provider, evaluate, snapshot, contract)

    with patch.object(provider, "evaluate", new=observed):
        yield evidence
