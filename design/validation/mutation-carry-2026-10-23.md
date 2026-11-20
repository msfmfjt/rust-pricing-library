# Forward and discrete-dividend mutation coverage — 2026-10-23

Base: `86738ae127c0ac2871d014ed8504b2f605e2eb0d`, following the
[discount-curve survey](mutation-curve-2026-10-23.md) in draft PR #172.
This follow-up adds 18 unit tests: six in `market::forward` and 12 in
`market::dividend`. Both production source prefixes remain byte-for-byte
unchanged. Public APIs, serialized schemas, dependencies, RNG settings and
existing numerical assertions are unchanged.

## Contracts and independent expectations

The tests exercise the forward, fixed-cash/proportional dividend and Spot-bump
contracts in [requirements section 3.2](../requirements-v1.0.md),
[architecture section 6.1](../architecture-v0.1.md) and the
[stochastic-dividend model specification](../../docs/models/stochastic-dividends.md).

The nonzero-carry reference uses `S0=128`, `D_r(t)=4^-t` and `D_q(t)=2^-t` at
the supplied pillars. The net growth is `2^t`. At time one the event pays cash
8 and proportion 1/4; at time two it pays cash 12 and proportion 1/2. Independent
exact-rational cash-flow recursion gives:

| Time | Phase | Canonical forward | Spot-contract forward |
| --- | --- | ---: | ---: |
| 0 | Initial | 128 | 128 |
| 1 | Before event | 256 | 256 |
| 1 | After event | 256 | 184 |
| 2 | Before event | 512 | 368 |
| 2 | After event | 512 | 172 |
| 3 | After all events | 1024 | 344 |

The funded cash reserve is `40/3`. Bumping initial Spot to 160 keeps this reserve
and both cash payments fixed, and produces a time-three Spot-contract forward
of 440. The JSON records the exact rational coordinates and cash-flow recursion.
Both carry curves have nonzero logs, including at the start of the interval
between events; a flat second curve would leave some sign errors unobserved.

Additional cases check:

- Curve and underlying identity, independent interpolation/extrapolation regions,
  default forward accessors and Spot replacement with and without dividends.
- Quote/event/compiled-event/timeline getters, strict cash and proportional
  domains, invalid times and event order, and structured errors for direct enum
  quotes that bypass the quote constructors.
- Empty schedules, the identity coordinate, funded-reserve boundaries and
  overflow, pre/post-event state reconstruction, and fixed-cash Spot refitting.
- The immediate floating-point neighbors of each ex-date and the contract that
  the cash coordinate is **exactly zero** after the final cash event. A decimal
  cash fixture leaves rounding residue if the documented clearing step is
  removed; this checks that exact-zero policy, without tightening price budgets.
- Canonical forward overflow before dividend carry; a positive subnormal
  canonical value whose proportional-dividend reconstruction underflows to zero;
  and zero-cash coordinates that do not need a representable carry-growth factor.
- Invalid query times, nonfinite strikes, nonpositive/nonfinite post-event Spot,
  nonfinite matching prices, default/custom tolerances and inclusive absolute
  and relative matching boundaries, including the adjacent representable value
  outside each accepted boundary.

The pinned cargo-mutants 27.1.0 visitor skips methods named `new` inside impls
and traits. Constructor/domain tests are explicit; generated coverage of those
methods is not claimed. No new mutation exclusion or timeout policy is added.

## Measured survey

Both complete runs select all 243 generated mutations in the same two files:
221 in `market/dividend.rs` and 22 in `market/forward.rs`. Both run only pricing
unit tests matching `market::`: 53 baseline tests before and 71 after. Neither
survey runs integration or numerical-crate tests for each mutation.

| State | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 243 | 126 | 83 | 0 | 34 | 2 |
| After | 243 | 206 | 3 | 0 | 34 | 2 |

All 243 identities match. **Eighty previously missed mutations become caught**;
all 126 previously caught mutations remain caught. The dividend file improves
from 114 caught / 81 missed to 192 caught / three missed, with 26 unviable builds.
The forward file improves from 12 caught / two missed to 14 caught / none missed,
with eight unviable builds. There are no timeouts. The combined diagnostic
still exits **2**, because three mutations remain missed.

The remaining guard mutations stay selected and reported as missed. Their
scope-specific redundancy arguments are:

| Location | Mutation | Assessment |
| --- | --- | --- |
| `dividend.rs:344:33` | Inner `||` to `&&` in reserve validation | Spot is finite and positive; a finite reserve is nonnegative, so its residual is finite too. A nonfinite reserve also makes the residual nonfinite. The two predicates have the same truth value for validated inputs. |
| `dividend.rs:482:37` | `&&` to `||` before an event | Successful funding starts with finite coordinates. Every preceding carry and event transition checks its updated coordinates before returning. Both final finite predicates are therefore true whenever this check is reached. |
| `dividend.rs:641:39` | `||` to `&&` in event-time validation | `DividendEvent` fields are private, and its public constructor already rejects nonfinite and negative times. Construction and Spot recompilation preserve this invariant, making both duplicate rejection predicates false. |

These arguments concern the current constructors and single mutations, not
synthetic invalid private states or simultaneous mutations. Constructor tests
remain important because the tool omits methods named `new`. There is no waiver
or exclusion, and these residuals do not become caught outcomes.

The 34 unviable mutations fail compilation: 31 need unavailable `Default`
implementations, two produce mismatched types, and one inserts an unsupported
`||` into a let-chain condition. They are not counted as detections.

The [JSON record](mutation-carry-2026-10-23.json) retains source hashes, commands,
timestamps, baseline phases, hashes of both complete inventories/reports, all
identity transitions, and failure evidence for newly detected mutations. The
runner's checker validates selected/result identities, aggregate counters,
execution phases, package selection and exit statuses. Every completed test
log is checked for the expected 53/71-test scope.

Reproduce on the corresponding source revision with a fresh output directory:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/forward.rs \
  --file=crates/pricing/src/market/dividend.rs \
  --cargo-arg=--lib --cargo-test-arg=market:: \
  --baseline=run --no-shuffle --jobs=4 \
  --timeout=20 --build-timeout=600 --output=target/mutation/carry-survey
```

Repository configuration adds `--all-features` and `--locked`. Debug information
is disabled to reduce artifacts; optimization and precision policy are unchanged.
The before run uses the clean base commit. The after run uses that same HEAD
with the test additions, pinned by the recorded file hashes.

## Validation and limits

- `cargo test --locked -p pricing -p pricing-numerics --all-features --lib`:
  **585 pricing and 55 numerical tests pass**, five existing pricing tests stay
  ignored, none fail.
- Workspace all-target/all-feature Clippy with warnings denied and formatting
  checks pass. Markdown links, the staged source archive and diff checks are
  verified for this change.

This diagnostic is narrower than the normal `full`/`changed` runner, which runs
both Rust crates and their integration/doc tests. Historical records are not
overwritten. The previous numerical survey retains 11 missed mutations, three
timeouts and seven unviable builds out of 952; the calendar survey retains five
timeouts and eight unviable builds out of 135. The discount-curve survey retains
68 caught and three unviable out of 71. No whole-library, ignored-acceptance or
installed-Python-wheel mutation coverage is claimed. Remote CI is reported
separately; PR #172 remains draft and unmerged.
