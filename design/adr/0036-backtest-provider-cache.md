# ADR 0036: Bound backtest provider cache lifetime

Date: 2026-10-22. Status: implemented.

## Context and decision

`RustPricingProvider` kept risks, Gamma audits and shared-surface hashes for every
snapshot it encountered. The quote-control sample cache had its own one-snapshot
lifetime. A long replay therefore accumulated duplicated calculation evidence in
the provider even though the ledger already retained its required output.

Implement the input-identity and performance requirements in the
[backtest requirements](../vega-kt-hedging-backtest-requirements-v0.1.md) with one
private snapshot cache, selected by the full snapshot fingerprint. Replace it
when the snapshot changes or the caller invokes `clear_cache()`. Cache results
and their optional Gamma audits as one value; return detached audit copies.

Add an optional positive integer constructor setting `cache_max_entries`, default
128. Bound the result cache and native quote-control cache separately to that
many entries, evicting the least recently used entry. Keep the active snapshot's
successful shared-surface fingerprint independently of entry eviction. Check it
before reuse and recheck at publication, so concurrently completed contracts
cannot publish contradictory surfaces. Failed evaluations publish neither a risk
nor a successful shared-surface fingerprint.

Hold cache references locally across native calls. Short locks protect active
snapshot replacement, individual cache operations and result publication; pricing
and calibration do not run under those locks. A call already in flight can finish
using the old state but cannot write it into the new active snapshot. Native
quote-control evaluation receives the same captured cache, including the existing
complete request/tangent/execution/gate key. Simultaneous identical misses may
compute twice; no single-flight scheduler is introduced.

## Compatibility and numerical impact

- The constructor keyword and `clear_cache()` are additive companion-package
  APIs. Pricing requests, native APIs, output schemas and ledger formats do not
  change. Internal separate risk/audit dictionaries become a paired cache.
- Returning to an expired snapshot behaves like starting a fresh provider for
  that snapshot: numerical gates rerun and the shared-surface guard is rebuilt
  from successful evaluations. Same-snapshot entry eviction keeps that guard.
- Price, Delta, Gamma, bucket units, calibration, random streams, sample pairing,
  covariance and numerical acceptance thresholds are unchanged. Native audit
  rejection remains mandatory on both cache hits and recomputation.
- Cache capacity affects retention and repeated work, not numerical identity.
  It is not serialized or added to checkpoint identity. Existing code/binary
  hashes still prevent importing checkpoints or audit evidence from a different
  implementation. Include the new cache module in multi-step provider hashing.
- Retained entry counts are bounded. Bytes per entry, concurrent workspaces,
  caller-held outputs and the ledger's own historical tables are not bounded by
  this setting. In-flight references expire when their calls finish.
- Replaying multiple strategies or revisiting dates can do more work than the
  previous unbounded cache. Use separate providers for independent histories;
  do not claim a runtime speedup from this memory-lifetime change.

## Validation

Exercise repeated same-snapshot calls, LRU eviction, 100 snapshot transitions,
old-state release, detached audits, explicit clearing, rejected native-audit
eviction/recomputation, and complete-content identity with reused snapshot IDs.
Synchronize concurrent tests to cover mismatched first results and a snapshot
change or clear while an audit is in flight; use no timing-dependent sleeps.

Compare actual installed-native results across eviction, revisits and clearing,
including complete Gamma/calibration/covariance audits. Compare serialized full
replay with a fresh-provider checkpoint resume at a different cache capacity.
Retain existing numerical conformance and cross-platform CI gates. Synthetic
cache tests establish lifecycle behavior, not financial accuracy.
