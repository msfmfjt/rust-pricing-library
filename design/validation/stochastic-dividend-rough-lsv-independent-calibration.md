# Independent finite-particle rough-LSV calibration

The [NumPy implementation](../../tests/python/rough_lsv_calibration_reference.py)
independently reconstructs the finite calibration used by stochastic-dividend
rough residual-LSV plans. It checks the calibration stage that earlier Barrier
valuation references treated as a retained input. The
[fixture](../../fixtures/stochastic-dividends/rough-lsv-calibration-reference.json)
contains explicit Gaussian inputs and independently generated calibration
outputs. Production calibration, path, price and risk functions are not called
to generate expected leverage or conditional moments.

## Inputs and numerical contract

The common input is a 64-particle, eight-step Gaussian table: Philox4x32 seed 42,
`LsvCalibration` domain, with particle-major rows and three factor-major blocks
(spot, orthogonal volatility, newest-cell residual). The
[Rust input test](../../crates/pricing/tests/rough_lsv_calibration_reference.rs)
checks every retained normal against its production coordinate within 2e-14.
This is an input-aligned algorithm comparison, not an independent RNG test.
Calibration normals are distinct from the four-factor valuation coordinates
that also contain the stochastic-dividend driver.

NumPy constructs Volterra cell averages from their analytic integrals and adds
an independent newest-cell residual. With discrete history X and variance V,
the volatility multiplier is `a = exp(eta*X/2 - eta^2*V/4)`. The calibration
particle's log residual equity uses the left-node multiplier and leverage:

```text
log(f_next/f0) = log(f/f0) - L2*a^2*dt/2 + sqrt(L2*a^2*dt)*z_spot
```

At each target log node x, the compact quartic weights are
`w = (1-u^2)^2` for `abs(u)<1`, zero otherwise, where
`u = (log(f/f0)-x)/bandwidth`. The normalization cancels. NumPy computes
weighted second, third and fourth moments of a and
`ESS = sum(w)^2/sum(w^2)` by matrix arithmetic. Rust instead evolves level states
and uses sorted particle windows with compensated summation.

A node with insufficient ESS borrows moment ratios from the nearest supported
log node, breaking equal-distance ties by lower node index. The recipient's
original ESS is retained. No supported node is an error. At time zero, or with
zero vol of vol, moments are exactly one and ESS is the particle count. Squared
leverage is target local variance divided by the conditional second moment;
spatial interpolation is linear with flat tails. These are the existing finite
algorithm's conventions, not new fallback or calibration policies.

## Controls and gates

Six cases cover:

- H=0.1, eta=0.6 and correlation -0.4 on a nonuniform grid and nonflat target.
- H=0.3, eta=1.1 and correlation +0.5 on the same grid.
- H=0.5 (Brownian limit), eta=0.8 and correlation -0.7.
- Zero vol of vol, for which the full leverage grid equals the target exactly.
- An under-supported interior node with equidistant supported neighbors. Binary
  node spacing makes the tie exact; short following steps preserve the gap.
- No supported node at the first positive time, which must reject calibration.

The Rust integration test compares every leverage value, conditional moment,
ESS, source-node index, extrapolation flag, row minimum supported ESS, fallback
count and mean particle level against NumPy outputs. Numerical tolerance is
`2e-12*(1+abs(expected))`; donor decisions and counts agree exactly. Separate
Python analytic controls check the Brownian exponential, irrelevance of its
newest-cell residual, zero-vol-of-vol moments, quartic boundary weights and
nearest-node tie behavior. Unsupported recipients must retain their own ESS.

The reference also recalibrates all 96 retained scenarios in the
[continuous quote-IV fixture](../../fixtures/stochastic-dividends/rough-continuous-market-iv-reference.json).
Original Dupire variance grids are linearly interpolated in variance onto the
nine-node execution grid. H=0.1/0.3, the original particle seed/count, bandwidth
0.35, minimum ESS 5 and funded residual equity are preserved. The independent
leverage values must match the retained Rust values within 2e-13 absolute.
Observed maximum discrepancies on the initial validation were 6.94e-17 for
H=0.1 and 1.25e-16 for H=0.3. The existing wheel tests separately recompile these
same 96 scenarios and compare independent NumPy Barrier price/Vega panels.

## Reproduction and CI

Normal verification needs only NumPy for the reference and the Rust test toolchain:

```sh
OPENBLAS_NUM_THREADS=1 python tests/python/test_rough_lsv_calibration_reference.py
cargo test --locked --release -p pricing --test rough_lsv_calibration_reference -- --nocapture
```

Linux CI runs the NumPy-only controls and 96-scenario reconstruction. Three-OS
Barrier CI runs the Rust input/output comparisons. Both retain their logs.
Default verification does not write fixtures. Wheel unittest discovery includes
the four Python controls; the Gaussian export test is ignored in ordinary runs.
To intentionally rebuild a fixture, export only the Gaussian inputs and then
run the independent calculation:

```sh
cargo test --locked --release -p pricing --test rough_lsv_calibration_reference \
  export_reference_calibration_normals -- --ignored --nocapture > /tmp/calibration-normals.log
python tests/python/rough_lsv_calibration_reference.py \
  --normal-log /tmp/calibration-normals.log --build /tmp/calibration-reference.json
```

## Limits

This establishes agreement of the stated finite-particle algorithm at fixed
inputs, including its moment/fallback branches. It does not certify infinite-
particle calibration, bandwidth or time-grid convergence, calibration sampling
uncertainty, calibration adjoints, other volatility models, physical-Spot quote
conversion, or the full VegaKT operator. Independent outer-seed calibration
replication and continuous-Barrier refinement remain separate requirements.
