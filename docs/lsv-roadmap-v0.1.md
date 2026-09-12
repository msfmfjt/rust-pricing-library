# LSV extension: initial implementation and remaining acceptance work

Status: Experimental, not an accepted replacement for the LV/VegaKT baseline.
Date: 2026-09-12.
Base: PR #45, `43b99c2965890b50746ebe90fd90ce44cbba5ebd`.

The requested LSV extension adds stochastic forward-smile dynamics while keeping
the existing implied-volatility surface as a calibration target. The first model
is the one-factor Bergomi example in Hamdouche and Henry-Labordere, with a
particle calibration based on Guyon and Henry-Labordere. Deterministic interest,
repo and dividend conventions remain those of the existing continuous-f model.

## References

- Guyon and Henry-Labordere, [The Smile Calibration Problem Solved](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032):
  equations (1)-(3), particle calibration (20)-(21), implementation on pp. 8-9,
  and the affine-dividend appendix.
- Hamdouche and Henry-Labordere, [Vega KT for LSV Models: An AD Approach](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114):
  Example 2.1, equations (2.3)-(2.7), Theorem 4.2, Proposition 4.4 and Section 4.6.

The supplied PDFs were read. They are not redistributed in this public repository.
Our coordinate and derivative conventions, including two printed-equation
normalization questions, are recorded in the [numerical contracts](lsv-numerical-contracts-v0.1.md).

## Available now

`pricing::lsv::BergomiLsvPricingPlan::compile` accepts an existing Price-only
`PricingRequest` whose model is the LocalVolatility target, a `Bergomi1Factor`,
explicit `LsvParticleConfig`, and `ExecutionPolicy`. The Python equivalent is
`BergomiLsvPlan.compile`. This separate experimental entry point makes the
calibration stage explicit. The stable request JSON still describes the LV target;
it does not serialize LSV parameters or calibrated execution state.

The implementation provides:

- Exact joint OU/spot-increment sampling and positive log-Euler f evolution.
- Quartic-kernel particle calibration, conditional second/third/fourth moments,
  effective-sample diagnostics, and explicit fallback counts.
- Independent Pseudo-MC or randomized Sobol pricing, both-factor antithetics,
  Brownian bridge, and worker-count-independent fixed-block reduction.
- Existing European, Asian, Lookback, Digital and discrete Barrier payoff graphs
  over a positive simulation horizon. Explicit smoothing is required for the
  Local-variance risk of Digital and Barrier payoffs.
- Affine dividend observations with continuous f, pre/post jump ordering, and
  the existing non-positive post-dividend Spot error.
- Path adjoints to squared relative leverage, followed by a full discrete
  particle-calibration VJP to the effective target Local-variance nodes.
- Rust/Python result objects, read-only Python properties and an LSV plan fingerprint including
  target request, factor parameters, calibration settings and realized leverage.

The [Python example](../examples/python/bergomi_lsv.py) prices an ATM Vanilla and
returns recalibrated Local-variance sensitivities. Call `evaluate()` to obtain
Price only. With `retain_reverse_trace=True`, call
`evaluate_local_variance_risk()` to obtain the complete mean derivative of that
finite calibration-and-pricing calculation.

## Sensitivity boundary

`LsvLocalVarianceRisk.node_adjoints` is
`dPrice / d(effective relative Dupire variance node)`, in the original target
grid's row-major order. It includes the response of the conditional variance
estimator to a target-grid perturbation. It is not a market-IV bucket Vega, an
SSVI-parameter derivative, a scalar volatility Vega, or a sticky-smile Spot Greek.
Those risk requests are rejected by the experimental compiler.

The implementation differentiates the finite regularized particle algorithm
directly. It does not claim to implement the infinite-particle linear-flow
solution in Theorem 4.2, or to use the first-order approximation (4.6). This gives
a reproducible discrete reference against which a later fast approximation can
be checked. SV parameters, grids, kernel bandwidth, seed and active fallback
decisions are held fixed by this VJP.

Price standard errors are conditional on the one realized calibration surface.
For RQMC, Local-variance risk standard errors are calculated from independently
scrambled pricing replicates after each replicate's calibration VJP. Pseudo-MC
returns the mean derivative with no fabricated risk standard error. Neither
estimate includes calibration noise or systematic discretization/kernel bias.
Repeating the entire calibration with independent calibration seeds is needed
to measure calibration uncertainty.

## Implementation and acceptance sequence

| Stage | Scope | State |
| --- | --- | --- |
| S0 | Coordinates, stochastic factor, derivative contracts and references | Implemented |
| S1 | Exact OU, frozen-leverage paths and multi-observation reverse | Implemented and tested |
| S2 | Particle calibration, moment/support diagnostics, calibration VJP | Implemented and tested |
| S3 | Rust/Python pricing, MC/RQMC, dividends and replay | Implemented; broad acceptance pending |
| S4 | Calibrated effective Local-variance risk | Implemented; refinement and performance acceptance pending |
| S5 | Market-IV VegaKT and sticky-smile Delta/Gamma conventions | Not connected |
| S6 | Stable LSV wire schema, platform replay fixtures and benchmark acceptance | Pending |

S5 must map calibrated Local-variance risk through a separately validated
Vanilla-hedge / implied-volatility projection. Reusing the LV reporting bucket
type does not establish correctness of that mapping. Required gates include
Vanilla and call-spread bucket localization, the nu=0 LV limit, recalibrated
bump comparisons, maturity-step refinement, and residual/active-domain accounting.
If approximation (4.6) is added, it needs its own explicit method identifier,
non-uniform strike derivatives, measured conditional moments, and an error study
against the discrete calibration VJP. The example's approximate moments are not
silently substituted for the measured moments.

## Verification and remaining limits

The initial regression suite covers the nu=0 LV limit; exact OU covariance;
state, grid and Gaussian path adjoints; calibration feedback versus full
bump/recalibrate; end-to-end target-grid derivatives with an inserted dividend
date; MC/RQMC replay across worker counts; unsupported risk rejection; and
independent Vanilla repricing plus a martingale check. These focused checks do
not certify calibration over an arbitrary market smile or parameter region.

Local Linux validation used the pinned Rust 1.98.1 toolchain and CPython 3.12:

| Check | Result |
| --- | --- |
| Workspace tests, excluding the Python extension | 294 passed; one statistical test run separately |
| Python extension native tests | 3 passed |
| Statistical acceptance | 1 passed |
| Built-wheel metadata, stub/runtime API and clean-environment smoke check | Passed, including 43 Python tests and all three examples |
| Clippy with warnings denied, formatting and Rust API documentation | Passed |
| Reference fixtures, schemas, Markdown links and dependency direction | Passed |

The supplied ATM example gave Price 7.9588405048 against the flat-volatility
Black-Scholes target 7.9655674554, with conditional pricing standard error
0.0100233907. This is one calibration seed and one parameter set, not a smile-wide
calibration acceptance result. The native extension tests required a local
linker search-path correction for the packaged Python runtime; no repository or
CI linker configuration was changed.

Before S3-S6 acceptance, retain particle-count/bandwidth/time-grid refinement,
multiple calibration seeds, adverse smile/high-vol-of-vol cases, long maturities,
memory/runtime measurements, and native macOS/Windows replay evidence. The
current calibration sorts particles per time step and evaluates supported
kernel neighborhoods; the calibration reverse is a deterministic full sweep.
Its retained trace costs O(particles * time steps) memory. Multi-factor Bergomi,
Heston/CIR, stochastic rates, multi-asset correlation, continuous barrier
corrections and LSM state expansion remain further model/engine extensions.

This branch depends on PR #45 and the preceding stack. It neither merges that
stack nor changes its acceptance status. Rebase with refreshed CI evidence if
the base stack is merged or rewritten.
