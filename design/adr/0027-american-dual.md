# 0027: Andersen–Broadie bounds using the existing LSM policy

## Decision

Implement the requested Andersen–Broadie (2004) policy-based primal-dual method
as `pricing::dual::AndersenBroadiePlan`. Use the existing AmericanVanilla
request, LSM training and exact constant-volatility transitions. Keep execution
and estimate construction in the private engine, re-exporting only the public
adapter and its configuration, result and error types.

The first slice supports deterministic-rate Black–Scholes/Black-76 Call/Put
prices with finite exercise dates and optional escrowed affine dividends.
It rejects unsupported models, RQMC and Greeks. This is an additive opt-in
API; none of the accepted E0–E8 behavior or exercise comparisons changes.

## Impact analysis

- API: new Rust module and immutable Python config, plan and result wrappers;
  no changed signatures or existing return types. Python estimates reuse
  `DiagnosticEstimate`; compilation and evaluation release the GIL.
- Serialization: existing request/result schemas, fingerprints and migration
  fixtures stay unchanged. There is no versioned dual-result wire adapter.
- Numerics: LSM continues on ties. Inner rollouts follow the same frozen rule;
  the value function is the policy value, including in continuation regions.
  No regression prediction is substituted for the nested expectation.
- Uncertainty: lower and correction share outer samples. Estimate the variance
  of their sum, preserving covariance and antithetic sampling units. Bounds
  apply in expectation to the declared grid; intervals are asymptotic and
  conditional on training.
- Replay: a new scheme tag and private Philox counter namespace separate inner
  randomness without extending the serialized `RandomDomain` enum. Nested
  counts and seed enter the new dual fingerprint. No previous random
  coordinates or accepted replay fixtures change.
- Resources: stream inner rollouts and stop them on future exercise. Check
  coordinate products before execution; reuse training matrix limits.
- Acceptance: exact enumerated-tree identities, an exhaustive finite-inner
  noise example, deterministic dividends/time-zero exercise, expiry-only
  equivalence, existing-LSM equivalence, worker replay, and independent lattice
  comparisons with inner-count refinement. Record results separately from the
  historical E0–E8 gates. Python tests check existing-LSM equivalence, replay,
  deterministic limits, immutable metadata and structured boundary errors;
  wheel checks include the stub contract and runnable Python example.

Future model extensions require a conditional path restart for the complete
state and their own acceptance evidence. No general-model support is inferred
from the duality theorem alone. See the [calculation contract](../../docs/library/american-dual.md)
and [validation record](../validation/american-dual.md).
