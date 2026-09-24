# Rough Bergomi / stochastic cash-dividend validation

## Scope and fixed checks

The implementation is defined by [ADR 0019](../adr/0019-rough-stochastic-dividends.md).
These new budgets are recorded before native execution. No legacy budget or seed
is changed. Sources are [hybrid scheme](https://arxiv.org/abs/1507.03004) and the
[existing rough model](../../docs/models/rough-bergomi.md).

1. Joint `(dW_f,dW_D,dW_v,J_near)` covariance reconstructed from normal basis
   against elementary power integrals: absolute error <2e-13; H in
   {0.01,0.1,0.49,0.5}, dt in {0.0001,0.17,1.3}. Discrete-history variance and
   lognormal centering on an irregular grid: <3e-13 and <2e-13 respectively.
2. Eta=0 paths exactly equal unchanged BS cash-dividend paths after projecting
   out the two unused coordinates. H=1/2 paths agree with k=0, nu=eta/2 Bergomi
   within 2e-14. The newest-cell residual is exactly zero at this boundary.
3. Independent irregular-grid reconstruction, including older power-cell weights,
   f/Y, both cash-event sides, same-date expiry and post-expiry cash:
   state error <3e-14, physical-stock error <1e-12. Changing future normals leaves
   all earlier states exactly unchanged.
4. Independent two-step price integration in
   [rough_dividend_reference.py](../../tests/python/rough_dividend_reference.py),
   using only NumPy/math, with no production path/covariance/payoff/reserve helpers.
   At H=0.1, order 20/24 values are 7.746733042675567 / 7.7467332042848;
   at H=0.5, 7.765235020943912 / 7.765235145769844. Order gaps <2e-7.
   Each native RQMC price must satisfy `abs(price-reference) < 6*SE+0.002`
   in price units. This targets the TWO-STEP finite algorithm, not continuous time.
5. Exact price/SE worker replay and repeated evaluation; all rough inputs change
   the fingerprint, including zero-loading correlations. Bad instantaneous PSD,
   nonfinite inputs, wrong normal shape and >4096 steps reject. Singular PSD may
   price; all AAD/Gamma methods reject rough plans explicitly, without affecting
   later price evaluation. Python adds invalid H, immutable APIs and a deterministic
   fixed-cash, zero-sigma price check.

## Execution and interpretation

Run the two private coefficient tests and five public Rust tests, the three Python
API tests, both quadrature orders, and the example. Normal Rust/Python discovery,
Clippy, formatting, documentation links, source archive and explicit wheel/stub
contracts must also pass. The supported-platform CI is independent of focused
Linux evidence; report each observed result against its commit/tree.

A hybrid approximation of the rough driver, a left-frozen equity step and the
positive dividend split all contribute time-grid error. Centering preserves the
finite-grid variance mean; it does not correct the option price to continuous
time. Sampling SE excludes grid/model uncertainty. Broad rough-dividend
refinement, exotic acceptance and reverse risks remain separate tasks.
