# Stochastic-dividend implementation and validation

Date: 2026-09-23. Baseline PR #83:
`cc06779bcfbc892f754669018f697bac8d346a7e`, tree
`131daba40e3339bb03f92e6640e00dd5375c3a84`.

[ADR 0013](../adr/0013-stochastic-cash-dividends.md) and the
[model specification](../../docs/models/stochastic-dividends.md) define scope.

## Added checks

The Rust integration suite covers validated parameters/states, zero and perfect
correlation, exact kappa=0 paths, fixed-cash retirement and the non-fixed nu=0,
alpha>0 case. It rejects nonfinite normals and numerical underflow rather than
clipping. Two-dimensional 12-point Gaussian quadrature checks actual one-step
first and second moments at three arbitrary states against independently
integrated split moments, including the future-forecast tower property.

Path tests fund all cash including post-expiry dates, reconstruct initial Spot,
check pre/post jumps for arbitrary observed states, compare the fixed-cash stock
identity, preserve the simulation horizon and reject omitted ex-dates. Public
pseudo-MC/RQMC tests cover replay across one/three workers, scheme/fingerprint
identity and parameter changes. Fingerprints intentionally retain the base
execution-policy identity; equal prices across workers do not imply equal hashes.
A deterministic dividend on expiry verifies post-event exercise and discounting
exactly once. A discrete up-and-out barrier verifies the pre-event stock is
observed even when the post-event stock lies below the barrier. Continuous
barriers and risk requests fail explicitly.

The independent kappa=0 European reference is a conditional Black integral for
`S_T=A*f_T+B*Y_T`, with sigma=0.2, nu=0.45, rho=-0.35, S0=K=100, discount=0.95,
carry factor=0.98, T=1, and a single mean cash dividend 25 at 1.4. Adaptive
quadrature on [-12,12] gives **7.653276188575835**, with reported integration
error 4.50e-12. The omitted Gaussian tail is negligible for these parameters.
This oracle does not call the library's forward, transition, RNG or payoff code.
Public RQMC uses seed 612, 2,048 points and eight scrambles, antithetic sampling,
with acceptance `abs(error) <= 6*SE+0.002`. The additive 0.002 is a stated pricing
budget, not a quadrature-error estimate; this exact limiting case has no time
splitting bias.

Four Python tests check the independent conditional Black reference (Gaussian
orders 96/128, required agreement 2e-7), immutable bindings, uncertainty metadata,
fixed-cash expiry, replay, fingerprint sensitivity and explicit invalid inputs.
Python quadrature returns 7.653276182234445 and 7.653276188820648 respectively.
The static wheel contract checks explicitly list the new classes and methods;
expectations are not inferred from the candidate stub.

## Validation status and limits

The editing environment has no Rust toolchain. Static Python/stub, schema,
reference-fixture, Markdown, source-archive and whitespace gates can run locally.
Native Rust, Clippy, Python-wheel and platform replay checks require the pinned
Rust 1.98.1 environment. Their actual revision, commands and outcomes must be
recorded in the PR/native logs; merely adding a test is not evidence it passed.
No earlier PR's tests certify this source revision.

The moment oracle checks the finite split, and kappa=0 pricing checks an exact
limiting law. There is not yet a broad continuous-time price/refinement panel
for nonzero mean reversion or a new stochastic-dividend exotic acceptance suite.
Fixed-cash/SV/HW/LSV legacy acceptance and the inherited 2F ensemble-SE issue
remain separate. No threshold, seed, platform identity or frozen replay fixture
is relaxed or relabelled here. No performance improvement is claimed.
