# eSSVI variance-derivative mutation coverage — 2026-10-27

Base: `a5a086fc5902bd396fba185bf18cdf7cb9accb09`, following the
[eSSVI parameter tests](mutation-essvi-parameters-2026-10-27.md) in draft PR #172.
Six tests strengthen the variance and derivative contracts in
[calculation specification sections 3.4–4](../../docs/models/local-vol-vegakt-calculation-specifications.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
serialized data, dependencies, arithmetic order, RNG settings and existing
numerical/acceptance tolerances are unchanged.

## Independent scalar references

Writing `chi = rho_psi`, the scalar eSSVI variance can be expressed directly in
the interpolated parameters:

\[
w(t,k)=\frac{\theta+\chi k+
\sqrt{\theta^2+2\theta\chi k+\psi^2k^2}}{2}.
\]

The [reference generator](../../scripts/generate_essvi_variance_references.py)
evaluates this scalar formula with Python Decimal arithmetic. It does not use
the production phi/rho quotients or their analytic derivative formulas.
Centered differences supply `w_k`, `w_kk` and `w_t`. Exact affine parameter
values/slopes come from the separate
[Fraction reference generator](../../scripts/generate_essvi_parameter_references.py),
whose slice times are 2, 4 and 8 with terminal theta slope `1/32`.

Nine cases span short extrapolation, all three knots, both interpolation
intervals and long extrapolation, with both signs of log-moneyness. The two
reference calculations use 100 digits with step `1e-20`, then 140 digits with
step `1e-30`. Their maximum absolute difference over the 36 scalar/derivative
values is **6.7608888508143e-42**. This is a finite-panel cross-check, not a
formal derivative-error bound or a claim of 100-digit derivative accuracy.

At a knot, the maturity difference follows the local affine path with the
reported right-hand parameter slopes. It does not difference across a change
of slope. At the final knot that path uses the terminal theta slope and zero
psi/rho-psi slopes.

The Rust test freezes all 36 variance/derivative references and uses a new
relative tolerance of `5e-13`, with explicit finite-value checks. The audit
matches every reference literal, 18 query coordinates, 18 theta metadata
literals and nine region labels. It also verifies the 12 slice inputs, terminal
slope and all 54 existing exact affine parameter references. The Rust tests do
not invoke either generator or acquire a new runtime dependency.

## Additional contracts

- Reflect all slice correlations and negate log-moneyness. Variance, second
  log-moneyness derivative and time derivative are preserved; the first
  log-moneyness derivative changes sign. Parameter metadata is also preserved.
- Change the time unit by four dyadic factors. The variance and log-moneyness
  derivatives stay fixed, while the time derivative transforms inversely.
- Flat slices with zero terminal slope retain the same smile and an exactly
  zero time derivative at and after the first knot.
- Invalid log-moneyness queries preserve their exact input bits and take
  precedence when time is also invalid. With finite log-moneyness, invalid time
  errors propagate with their original coordinate and bits. Cases include both
  signed zeros, a negative subnormal, infinities and a payload NaN.
- Queries at either sign of maximum finite log-moneyness reject an infinite
  computed variance with the original time and log-moneyness context.
- An admissible flat surface with `theta = psi = 1e-200`, `rho_psi = 0` rejects
  a nonfinite time derivative. Its squared quotient-rule denominators underflow
  to zero. The test checks the error field and query coordinates, and classifies
  the computed value as NaN without freezing its payload across platforms.

The last case preserves fail-closed behavior: exact arithmetic would give a
finite zero time derivative, but the current arithmetic cannot evaluate it.
This is not an accuracy guarantee over all finite inputs or a numerical-policy
change. Every new test queries the public surface; none calls the private
variance helper directly.

## Matched mutation comparison

Both inventories contain the same **109 identities, metadata and complete
patches**: two operations in the public variance wrapper and 107 in
`evaluate_total_variance`.

| eSSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 10 tests | 109 | 87 | 20 | 0 | 2 | 2 |
| Expanded 16 tests | 109 | 107 | 0 | 0 | 2 | 0 |

**All 20 previously missed mutations become caught**, with all 87 existing
detections retained. Seven previously missed operations affect `w_k`; thirteen
affect `w_kk`. The old tests checked time differentiation and short-end scaling,
but many incorrect log-moneyness derivatives preserve those relations. They
therefore needed independent checks of the derivative values themselves.

For example, replacing the final `/ 2` in `w_k` with `% 2` changes the first
new panel value from approximately `-0.04515347445384588` to
`-0.09030694890769175`. The ten original tests pass; the independent-reference
test fails. All 20 newly detected operations fail that new reference test, and
one also fails the reflection test. This focused result does not imply that
the original full integration suite missed the same operations.

Both unviable operations replace a variance-returning function body with
`Ok(Default::default())`. Each fails with E0277 because
`TotalVarianceDerivatives` does not implement `Default`. They retain their
original classification and are not detections. No exclusion or waiver is added.

Both baselines pass. The normal runner's outcome validator reconciles selected
identities, completion, counts, packages, phases and statuses. Each survey has
108 complete test logs (baseline plus 107 viable operations), and two build-only
logs. Every viable log contains all ten or sixteen selected tests, zero ignored
tests and 642 filtered tests. Every caught operation has actual test failures.

The [JSON record](mutation-essvi-variance-2026-10-27.json) retains the full common
inventory and each complete patch once. Both runs' actual patch files were
compared byte-for-byte with that shared patch, and their individual hashes are
retained alongside log hashes, execution phases and failed-test names.

Reproduce the expanded diagnostic with Rust 1.98.1 and cargo-mutants 27.1.0:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/essvi.rs \
  --re='total_variance_derivatives|evaluate_total_variance' \
  --cargo-arg=--lib --cargo-test-arg=market::essvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/essvi-variance
```

The repository configuration adds `--locked` and `--all-features`. The original
survey uses a detached checkout of the base above with the identical options.
This inventory does not mutate constructors, slice admissibility checks or
parameter interpolation. The previous parameter survey is separate historical
evidence. Numeric constants, argument substitutions and metadata assignments
absent from the pinned inventory remain outside its denominator.

## Other validation and remote evidence

- **708 unit tests pass**: 653 pricing and 55 numerical, with five existing
  ignored tests and no failed/filtered tests. All sixteen focused eSSVI tests pass.
- Existing Standard SSVI and eSSVI integration reference targets both pass.
- Workspace Clippy across all targets/features with warnings denied, formatting,
  dependency direction and all 22 static/script checks pass, including 24
  mutation-runner tests. Markdown links, staged source archive and diff checks pass.
- The stacked changed-line inventory against
  `376c92624f88083b46d86e1087d2b0f03e202b91` remains 41 operations.
  This local check is list-only, not a new changed-line mutation run.

The immediate parent's [mutation workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40437811550)
completed successfully, including smoke, both changed-line shards and the
aggregate. Smoke caught 15/15. Shards selected 21/20 and caught 19/19, with
two/one unviable builds and zero misses/timeouts. Exact head/base, job metadata
and completed logs were checked; artifact ZIP contents were not independently
re-audited. Smoke also passed 24 runner tests and the installed-tool fixture
scenarios, whose deliberate misses and failed baseline verify failure propagation.

The parent's [ordinary CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40437811592)
remained in progress at observation. Its three Rust platform jobs passed;
wheel provider/custom-basis groups were still running, and the Windows
stochastic-dividend risk/refinement job was cancelled. Its log reports
`The operation was canceled` without establishing a cause. Overall ordinary
CI success is not claimed. The parent's separate
[Rough family MC Delta and LSV AAD workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40437811412)
completed successfully. Job-list retrieval covers the API's first page only.

No new full integration survey, ignored heavy acceptance panel or installed
Python-wheel run is claimed here. Earlier full-suite evidence, remote failures
on older heads, the unresolved fractional-OU guard and other recorded survivors
remain historical. New-head CI is separate from this completed local and
parent-head evidence.
