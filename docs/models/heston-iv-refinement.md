# Heston IV grid validation and staged recalibration

This experimental layer extends [exact-IV calibration](heston-iv-calibration.md).
It compares the *same parameter vector* on explicit Fourier grids before optionally
recalibrating. It changes neither the Fourier/Riccati equations nor the IV objective.

## Grid definitions

For calibration grid `(N, M, U)` (Riccati time steps, Simpson intervals, frequency
cutoff), the report uses this fixed column order:

| Column | Grid | Changed quantity |
| --- | --- | --- |
| base | `(N, M, U)` | None |
| time | `(2N, M, U)` | Only Riccati time steps |
| frequency | `(N, 2M, U)` | Only frequency spacing, halved |
| cutoff | `(N, 2M, 2U)` | Cutoff doubled, frequency spacing unchanged |
| joint | `(2N, 4M, 2U)` | Time and cutoff doubled, frequency spacing halved |

All five columns price exactly the same model. The report contains quote-major IVs,
raw target residuals, and probe-minus-base differences. Quotes stay in input order,
including duplicates. No probe drops a quote. An invalid price, unresolved IV,
excessive cost, or invalid grid returns an explicit error, not an accepted partial
result. A plan is shared by quotes at the same maturity, separately on each grid;
validation uses prices only, not parameter Jacobians.

## Finite-grid policy and units

`HestonIvRefinementOptions` has `fit_tolerance`, `grid_tolerance`, and `max_stages`.
Both tolerances are **absolute annualized IV**, regardless of quote `iv_scale`.
One IV basis point is `0.0001`. The defaults are 5 IV bp of target error, 1 IV bp
of grid change, and at most two stages.

`fit_within_tolerance` requires every quote on **all five grids** to meet the raw
IV target tolerance. `grid_stable` requires every quote on **all four probes** to
stay within the grid-change tolerance relative to base. `accepted` requires both.
The tolerances themselves are retained in each validation report.

These flags are distinct from the optimizer's `fit_achieved`, which retains its
original **scaled residual** criterion. A solver can stop at a boundary with
`fit_achieved=false` under a 0.02-bp solver target while the fitted model passes a
separately specified 5-bp finite-grid policy. The original flag and termination
reason remain accessible; they are not overwritten.

**Accepted is not a total-error certificate.** These are finite differences between
numerical approximations, not independent continuum prices. Cancellation or shared
bias can make them small. They do not bound the omitted Fourier tail, establish
global minimization, guarantee a unique parameter vector, test holdout quotes, or
establish the accuracy of a different Monte Carlo discretization. No confidence
interval or sampling standard error is assigned to these deterministic differences.

## APIs

Rust: `problem.validate_grid(&parameters, policy)` validates an explicit point.
`problem.calibrate_refined(optimizer_options, policy)` calibrates and validates
one stage at a time. Python provides the same two methods and frozen report types.

```python
policy = rp.HestonIvRefinementOptions.create(
    fit_tolerance=5e-4, grid_tolerance=1e-5, max_stages=2,
)
result = problem.calibrate_refined(
    policy, max_iterations=60, max_evaluations=100,
    residual_tolerance=2e-6,
)
for stage in result.stages:
    print(stage.calibration.termination, stage.calibration.fit_achieved)
    print(stage.validation.configurations)
    print(stage.validation.max_abs_iv_residual)
    print(stage.validation.max_abs_iv_difference)
model = result.stages[-1].calibration.model
```

The [complete example](../../examples/python/heston_iv_refinement.py) fits an SSVI
surface with six Rough Heston parameters. Lifted Heston retains a fixed finite
kernel and permits only its five existing scalar calibration directions.

## Staging, budgets and reproducibility

A failed policy moves to the joint grid, warm-started at the exact preceding final
parameter vector. All quote inputs, targets, parameter subsets, bounds, scales and
optimizer options remain unchanged. Each stage retains its initial point, full
parent calibration result, and validation report. The input problem is immutable.
No improvement or parameter-monotonicity guarantee is made across grids: objectives
on different grids are different functions.

`max_iterations` and `max_evaluations` are **per-stage** optimizer budgets, including
the parent's final evaluation in the latter count. `optimizer_evaluations` sums
these counts; it does not include validation. `validation_plan_compilations` counts
five pricing-only plans per distinct maturity per executed stage. No work occurs
on later stages after acceptance. Exhausting the stage limit returns `accepted=false`
and Python termination `stage_limit`, not a successful fit. Unexpected numerical
failures raise an error rather than returning a partial-stage history.

The entire requested ladder is checked before optimization, even when an early
stage might pass. Each grid must meet the parent's configuration bounds, each
pricing-only plan its 8-billion work estimate, and the sum of requested calibration
and validation work must not exceed the existing trillion-unit calibration budget.
This is an operation-count guard, not an elapsed-time estimate. `max_stages` is
limited to 1 through 3; not every starting grid can support every stage count.
For example, two stages from `M=2048` would require an invalid final joint grid
with 32768 intervals and are rejected rather than silently shortened. Smaller
starting grids can support a longer ladder. The single-point `validate_grid`
method uses only the first set of probes, regardless of `max_stages`.

See the [validation record](../../design/validation/heston-iv-refinement.md) for
executed tests, the documented first precision failure and scope limitations.
