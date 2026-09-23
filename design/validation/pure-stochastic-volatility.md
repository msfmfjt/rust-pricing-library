# Pure SV implementation and validation

Date: 2026-09-22. Baseline: PR #82 at
`3b646c58396199f25e0de231e923404d03f522ce`.

[ADR 0012](../adr/0012-pure-stochastic-volatility.md) and the
[model specification](../../docs/models/pure-stochastic-volatility.md)
define the scope. This change introduces single-asset pure 1F/2F Bergomi
and a deterministic-rate facade shared with pure rough Bergomi.

## Numerical evidence

`crates/pricing/tests/pure_bergomi.rs` contains five integration tests:

- Nonuniform-time factor centering and externally supplied innovations
  compared with independent OU covariance and path equations, for 1F/2F,
  including zero and near-zero mean reversion.
- BS at zero vol-of-vol, 2F at zero mixing weight, and rough H=1/2 with
  eta=2*nu compared on shared draws, including stochastic rates.
- Public MC/RQMC price and AAD identity, Spot and sigma0 central differences,
  worker-count replay and full escrowed reserves including post-expiry cash.
- Deterministic facade versus exactly equivalent zero-volatility HW, and
  distinct model/parameter fingerprints.
- Unsupported base model, invalid steps, correlation mismatch and non-PSD
  joint Brownian correlations rejected.

`tests/python/test_pure_bergomi.py` contains five public-API tests:

- Price/AAD identity and 32 Spot, sigma0, discount/carry log-DF directions
  checked by full recompilation, over 1F/2F, deterministic/HW and cash/no-cash.
  Absolute central-difference budget is 3e-5 with bump 1e-6.
- Rough deterministic facade preserves the existing zero-rate-HW result and
  fingerprint; Python ownership remains immutable.
- Finite right derivative at sigma0=0 in four 1F/2F/rate configurations,
  using a deep ITM call to avoid payoff-kink ambiguity.
- Six independent two-step price references: 1F/2F at strikes 90/100/110.
  OU covariance and mean normalization are derived directly, and the last
  equity step is integrated analytically. Gaussian quadrature orders 64/96
  agree within 2e-6. Each public price uses 4,096 points in eight scrambles,
  antithetic paths and a 6*reported-SE + 2e-6 acceptance budget.
- Parameter/step validation and fingerprint sensitivity.

The two-step oracle checks the implemented finite algorithm independently;
it does not certify continuous-time discretization convergence over arbitrary
parameters or maturities. Existing refinement budgets are unchanged.

## Validation status

- New Rust integration tests: 5 passed.
- New Python tests: 5 passed; full installed-wheel Python suite: 94 passed.
- All-feature Rust workspace excluding Python: 531 passed, 44 intentionally
  ignored heavy tests scheduled separately; no failures.
- Formatting, Clippy with warnings denied, reference fixtures (67/114/61),
  schemas, dependency direction and public type-stub conformance passed.
- Rust Python binding tests: 3 passed. A freshly built CPython 3.12 wheel
  passed the full clean-environment smoke gate, including public symbol/stub
  conformance, all examples and all 94 Python tests.
- Minimal-feature pricing suite: 520 passed. Rust documentation built.
- Explicit statistical, path-dependence and early-exercise acceptance passed
  (1, 2 and 2 tests). Path dependence was repeated in release mode because the
  first command's log omitted the final summary; the repeated run confirms
  both tests passed. Extended recalibrated AAD/VegaKT acceptance: 11 passed.
- Benchmarks and replay generators ran; the final report checker rejects
  Linux replay fixtures, as specified by its macOS/Windows-only platform
  contract. No platform or fixture was relabelled. Native supported-platform
  replay certification requires CI.
- The full extended price/refinement panel was invoked. Its first test,
  `multi_asset::bs_basket_refinement_consistency`, passed. The long run was
  intentionally interrupted during the next test; the remaining pre-existing
  price/stress gates are not certified here and must complete in CI. No
  acceptance threshold, fixture or source setting was changed.
- Source archive, Markdown links and whitespace checks passed.
- Native CI results are recorded on the pull request. The local results above
  do not replace the pending supported-platform and extended acceptance gates.

Local validation uses Linux x86-64, Rust 1.98.1 and CPython 3.12.14. The local
runtime required the existing empty-object workaround: incremental disabled,
one Cargo job, and `-C lto=off -C codegen-units=1 -C llvm-args=-threads=1`.
The Python Rust test linker additionally needs a scratch linker symlink to the
runtime's actual libpython3.12.so.1.0, plus LIBRARY_PATH/LD_LIBRARY_PATH: sysconfig
reports /install/lib and the installed unversioned library symlink is stale.
These environment settings are not committed build configuration; native CI
uses the repository's normal toolchain and flags.
