# Model and numerical method references

This page is the bibliography for the model and calculation-method specifications
under `docs/`. Each model or method document repeats the entries relevant to
its scope. A citation identifies the mathematical source of a model or method;
the repository-specific discretization, branch policy and error tolerances are
specified separately in the linked calculation specification and are not implied
by the paper.

## Core pricing models and volatility surfaces

- [Black and Scholes, *The Pricing of Options and Corporate Liabilities*](https://doi.org/10.1086/260062), *Journal of Political Economy* 81 (1973), 637–654. Continuous-volatility Black–Scholes pricing and Greeks.
- [Black, *The Pricing of Commodity Contracts*](https://doi.org/10.1016/0304-405X(76)90024-6), *Journal of Financial Economics* 3 (1976), 167–179. The forward-price Black-76 formula used for futures and bond-option references.
- [Dupire, *Pricing with a Smile*](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile), *Risk* 7 (1994), 18–20. The local-volatility construction from the call-price surface.
- [Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323), *Quantitative Finance* 14 (2014), 59–71. SVI/SSVI parameterization and sufficient static-arbitrage conditions.
- [Corbetta, Cohort, Laachir and Martini, *Robust calibration and arbitrage-free interpolation of SSVI slices*](https://arxiv.org/abs/1804.04924), arXiv:1804.04924 (2018). eSSVI slice interpolation and calendar consistency.

## Stochastic, local-stochastic and rough volatility

- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), *Risk* (2005), 67–73. Multi-factor forward-variance/Bergomi volatility factors and their correlations.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), SSRN 1885032 (2011). Particle calibration of local-stochastic volatility and the affine-dividend construction.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), SSRN 4304114 (2022). LSV path adjoints and the finite-dimensional VegaKT extension.
- [Bayer, Friz and Gatheral, *Pricing under rough volatility*](https://doi.org/10.1080/14697688.2015.1099717), *Quantitative Finance* 16 (2016), 887–904. The rough Bergomi model and rough-volatility pricing.
- [Bennedsen, Lunde and Pakkanen, *Hybrid scheme for Brownian semistationary processes*](https://arxiv.org/abs/1507.03004), arXiv:1507.03004 (2015). Near-cell/older-cell Volterra discretization used by the rough driver.

## Rates, hybrid models and dividends

- [Hull and White, *Pricing Interest-Rate-Derivative Securities*](https://doi.org/10.1093/rfs/3.4.573), *Review of Financial Studies* 3 (1990), 573–592. One-factor Hull–White short-rate model and curve-fitting shift.
- [Fries, *A Short Note on the Exact Stochastic Simulation Scheme of the Hull-White Model and Its Implementation*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2737091), SSRN 2737091. Exact joint short-rate and integrated-rate simulation.
- [Cozma, Mariapragassam and Reisinger, *Calibration of a Hybrid Local-Stochastic Volatility Stochastic Rates Model with a Control Variate Particle Method*](https://arxiv.org/abs/1701.06001), arXiv:1701.06001 (2017). Hybrid stochastic-rate/local-stochastic-volatility calibration.
- [Buehler, *Volatility Modelling with Cash Dividends and Simple Credit Risk*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1141877), SSRN 1141877. Future-dividend reserve and escrow-coordinate decomposition.
- [Henry-Labordère, *Equity modelling with local stochastic volatility and stochastic dividends*](https://www.risk.net/media/download/991346/download). Stochastic-dividend extension of the LSV calibration equation; the fixed-cash implementation is a specialization.

## Multi-asset and local correlation

- [Langnau, *Introduction into “Local Correlation Modelling”*](https://arxiv.org/abs/0909.3441), arXiv:0909.3441 (2009). State-dependent correlation calibrated to an index-volatility target.
- [Guyon, *A New Class of Local Correlation Models*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2283419), SSRN 2283419 (2013). Local-correlation families and the basket-variance projection used by the adapter.
- [Jourdain and Zhou, *Existence of a calibrated regime switching local volatility model and new fake Brownian motions*](https://arxiv.org/abs/1607.00077), arXiv:1607.00077 (2016). Conditional-expectation calibration and interacting-particle context.
- [Margrabe, *The Value of an Option to Exchange One Asset for Another*](https://doi.org/10.1111/j.1540-6261.1978.tb03397.x), *Journal of Finance* 33 (1978), 177–186. Closed-form exchange-option reference used by multi-asset tests.

## Monte Carlo and numerical differentiation

- [Glasserman, *Monte Carlo Methods in Financial Engineering*](https://doi.org/10.1007/978-0-387-21617-1), Springer (2004). Monte Carlo estimators, antithetic variates, confidence intervals and common-random-number validation.
- [Sobol', *On the distribution of points in a cube and the approximate evaluation of integrals*](https://doi.org/10.1016/0041-5553(67)90144-9), *USSR Computational Mathematics and Mathematical Physics* 7 (1967), 86–112. Sobol' low-discrepancy sequences.
- [Joe and Kuo, *Constructing Sobol Sequences with Better Two-Dimensional Projections*](https://doi.org/10.1137/070709359), *SIAM Journal on Scientific Computing* 30 (2008), 2635–2654. Direction numbers used by the RQMC implementation.
- [Salmon, Moraes, Pfau and Frey, *Parallel Random Numbers: As Easy as 1, 2, 3*](https://doi.org/10.1145/2063384.2063405), SC11 (2011). Counter-based Random123/Philox streams.
- [Broadie, Glasserman and Kou, *A Continuity Correction for Discrete Barrier Options*](https://doi.org/10.1111/1467-9965.00035), *Mathematical Finance* 7 (1997), 325–349. Discrete-monitoring and Brownian-bridge barrier corrections.
- [Longstaff and Schwartz, *Valuing American Options by Simulation: A Simple Least-Squares Approach*](https://doi.org/10.1093/rfs/14.1.113), *Review of Financial Studies* 14 (2001), 113–147. Least-squares Monte Carlo (LSM) for early exercise.
- [Giles and Glasserman, *Smoking Adjoints: Fast Evaluation of Greeks in Monte Carlo Calculations*](https://people.maths.ox.ac.uk/gilesm/files/NA-05-15.pdf), *Risk* 19 (2006), 88–92. Reverse/adjoint pathwise differentiation for Monte Carlo Greeks.
- Golub and Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press (2013), chapters 5.2–5.4. Householder QR and QR with column pivoting used for the LSM regression.
