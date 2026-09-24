# Stochastic-dividend Gamma validation

## Prespecified protocol

Baseline: PR #89 commit `7ed26f5aa48855fd29db8d9f6b4cbc402d737c99`, source tree
`ef7b985dab4af3de556884d48d60b063eb2114a1`. Tests below are new gates;
no existing tolerance, seed, schema or replay fixture is relaxed.

- Full-recompile AAD Delta bumps: BS/1F/2F, MC/RQMC, two seeds, European,
  delayed-payment Asian and smoothed pre/post-dividend barrier payoffs. All three
  ladder values agree within 2e-12 absolute. Baseline price/Delta/SEs and 1/3-worker
  numerical results are exactly equal. Execution-policy provenance changes the
  plan/risk fingerprints across worker counts, as in the baseline contract.
  Relative versus equivalent absolute bumps give the same estimates but distinct
  risk fingerprints.
- Independent one-step BS construction: build f/Y from explicit lognormal formulas,
  evaluate terminal Delta without payoff/reverse/reserve helpers, then compute
  sample variances of paired Gamma and ladder gaps. MC/RQMC with and without
  antithetics must agree within 2e-12 in mean and SE. RQMC units are scrambles.
- Fixed-cash independent Black Delta formula: nonzero cash at expiry and beyond it;
  each finite-bump Gamma satisfies `abs(actual-reference) < 6*sampling_SE + 3e-5`.
  This tests a known finite-bump target, not a vanishing-bump Monte Carlo limit.
- Explicit errors for invalid/indistinguishable bumps, nonpositive funded downside,
  and unsmoothed discontinuities. Failure does not mutate basic risk. A zero-vol
  deep-ITM case has Gamma and SE below 1e-12; perfect-correlation pricing domains
  retain Gamma even when correlation AAD rejects.
- Three Python tests: fully recompiled Delta bumps, metadata/copy/replay/identity,
  invalid conventions and unsupported domains. Wheel and stub contracts include
  the new immutable result and explicit optional keyword defaults.

## Reproduction

```bash
cargo test --locked --release -p pricing --test stochastic_dividend_gamma -- --nocapture
python -m unittest discover -s tests/python -p test_stochastic_dividend_gamma.py -v
python examples/python/stochastic_dividend_gamma.py
```

Observed native runs and source-tree identity are recorded in the pull request.
This protocol alone is not evidence that an unexecuted gate passed. Full legacy
price/risk, platform wheel/replay and continuous-time acceptance are separate.
Neither the ladder nor a small paired sampling error proves absolute Gamma
accuracy for general nonsmooth products. For a vanilla kink, reducing the bump
can leave fewer crossing paths and produce a misleading zero estimate/SE in a
finite sample. Inspect several bumps and increase sampling where needed.
