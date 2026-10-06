# Mixed Bergomi representation-invariance checks

Base: PR144, `65ded25690827eac75d6f7b95639f360bbb004b8`.
This increment changes tests and this protocol only. Production pricing,
calibration, reverse formulas, APIs, sampling and numerical tolerances in existing
tests remain unchanged.

## Purpose

The shape-risk coordinates use the last component as the balancing weight.
Reordering economically identical mixture components therefore changes the
coordinate basis, not the underlying price or directional risk. Ordinary
single-coordinate bump tests do not test this transformation explicitly.

The new Python module is
`tests/python/test_mixed_bergomi_representation_invariants.py`. It is discovered
by the existing full wheel smoke suite on its source tree. Run it directly after
installing the wheel, with `tests/python` on `PYTHONPATH`, using:

```sh
python -m unittest discover -s tests/python -p 'test_mixed_bergomi_representation_invariants.py' -v
```

## Fixed cases and gates

All fixtures and gates below are declared before the first numerical run.
The existing builders retain nonflat carry, delayed-payment Asian observations,
pre-expiry fixed/proportional cash and fixed cash after expiry. Use H=.1/.5,
MC128 or RQMC4x64, valuation seed819 and the inherited bridge/antithetic settings.
LSV has128 particles, calibration seed429, bandwidth.5 and ESS3. No failed path is
dropped. These are identity tests, not accuracy prescriptions.

- **Component permutations: 48 cases.** Three components have weights
  (.2,.35,.45) and eta (.3,.8,1.1), rho=-.6. For each of six permutations, compare
  Pure/LSV prices, eta/rho/Hurst gradients, and shape gradients. For the old
  simplex gradient g, append g_last=0. Under permutation p, the new weight
  gradient must be g[p(i)]-g[p(last)]. Apply this to direct, calibration and total
  LSV contributions separately; original xi-coordinate gradients are unchanged.
  Require the LSV extrapolation-node set unchanged and compare Leverage values.
  The sum-zero physical transfer direction (1/16,-1/32,-1/32) is invariant too.
  Check that direction by full price rebuild/recalibration at widths1e-6/5e-7:
  96 derivative comparisons, without selecting a width by observed error.
- **Identical-component split: 8 cases.** Replace the last weight .45 by
  (.18,.27), both with eta1.1. Price is unchanged. Add the two split eta
  adjoints to recover the original eta adjoint. The new transfer between
  identical components is zero. Old weight-transfer and xi gradients,
  rho/Hurst gradients, Leverage and extrapolation-node sets are preserved.
- **Finite xi changes: 8 cases.** Replace original curve values
  (.04,.05,.045,.055) at times(0,.4,.9,1.4) by (.02,.08,.03,.11), or by the
  exponential .08*exp(-.3t). The target and shared calibration/valuation grid
  stay fixed. Compare each ell_new(t,k)*xi_new(t) against ell_old(t,k)*xi_old(t),
  where ell is squared Leverage. LSV price and price SE stay unchanged, while
  Pure-SV price must change by more than.001 and Leverage by more than1e-4.
  Require nontrivial direct xi risk (>1 in some coordinate) and its cancellation
  by calibration risk; do not hard-code this cancellation into production.

Discrete identity comparisons use absolute gap <=2e-9*(1+abs(reference)).
Directional finite differences use3e-5*(1+abs(reference)). Each test reports
case counts, scalar comparison counts and maxima; retain the first execution.

## Uncertainty and limits

Do not apply the simplex gradient difference formula to marginal standard
errors: changing the balancing component needs covariance. Compare SEs only for
unchanged sampling channels or genuine permutations. Likewise, adding split-eta
adjoints does not justify adding their marginal SEs as independent quantities.
The existing independent scramble/covariance controls remain in place.

Finite xi cancellation applies to the implemented high-level shared time grid,
fixed relative target and retained support/donor/interpolation branches. It is
not asserted for an independently interpolated off-grid Leverage surface, a
changed target convention or stochastic-model refitting. These tests do not
bound continuous-time, calibration-seed, smoothing, truncation or model errors,
and do not establish five-IV-basis-point pricing accuracy.
