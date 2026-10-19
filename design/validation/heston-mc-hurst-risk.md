# Rough Heston MC Hurst validation protocol

Base: PR #139, `9232c64969f936e7484c988a5ec76ae0a7c65c5b`.
Experimental, separate from production admission and the Fourier Hurst method.

Budgets below were specified before the new numerical test runs.

- Path panel: H=.03/.1/.3/.5, nu=.15/.7, 24 paths, seed91, nonuniform nodes
  0/.07/.21/.63/1; arbitrary forward and diffusion-variance cotangents at all nodes.
  Central bumps 2e-7/1e-7 in the interior; second-order backward differences at
  H=.5. All 384 comparisons must satisfy `3e-5*(1+abs(reference))`. Truncation
  counts must not change under these selected bumps; negative nodes are counted.
- High-level MC/RQMC: H=.1/.3/.5, 128 MC antithetic pairs or 4x64 RQMC points,
  price seed819, Brownian bridge, fixed cash before and beyond expiry, bumps
  1e-6/5e-7. All12 differences use the same `3e-5*(1+abs(reference))` budget.
- Independent kernel references: twelve 70-digit mpmath derivatives, including
  H=.001/.5 and nearly equal positive lags, budget2e-12*(1+abs(reference)).
- Explicit one-step Brownian-boundary check seeds terminal variance and a
  nonzero newest-cell residual. Dropping the left residual derivative fails.
- Independent finite-grid Black panel: v0=.04,kappa=.7,theta=.055,nu=0,rho=-.6,
  four uniform steps, S=K100, unit discount/carry, H=.1/.3/.5,seeds91/1973,
  RQMC8x2048 antithetic/bridge points. The independently differentiated drift
  recurrence fixes the integrated variance and the Black Hurst reference.
  Gates: price/Hurst gap<=5*correspondingSE+2e-4 and0<HurstSE<=.01.
  No continuum price-risk interpretation is made.

A separate two-pass reconstruction verifies pair-level MC and scramble-level
RQMC means and standard errors with Brownian bridge on/off. It reuses the
low-level path reverse, not the high-level estimator.

Tests additionally cover shape/nonfinite/domain errors, seeded/unseeded terminal
zero, worker1/3 reproducibility, price invariance and existing parameter-risk
invariance. Python covers Asian/delayed payment/nonflat carry/future cash,
smoothed versus hard Digital and immutable result properties. Public standard
error must be interpreted using the same sampling units as the parent MC API.

CI extends the existing Heston MC workflow; all parent gates remain intact.
References are checked, not regenerated, in CI. Release explicitly executes the
normally ignored Black panel on Linux/macOS/Windows and retains a required log.
No current-head hosted success is asserted by this protocol. Actual local run
commands, exits, failures and results accompany the delivery report.
