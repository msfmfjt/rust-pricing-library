# Rough-family LSV Spot Delta validation protocol

Base: PR #134 at `9dbc03437002bf3bb8ca33fad476f5c34e1ab890`, tree
`e7d6945ef6ce8203244d59ce3f80deb0bf64761c`. This change adds two explicit
high-level Spot conventions; it does not alter calibration/primal algorithms.
[Calculation specifications](../../docs/models/rough-family-lsv-spot-delta.md).

## Predeclared checks

Nondegenerate six-model panel retains H=.2, v0=.04 and the explicit settings
in `crates/pricing/tests/rough_family_lsv_spot_delta.rs`. Target has five
nonuniform times and five spatial nodes (zero is inside a cell), particles128,
seed429, log-bandwidth.5, minimum effective samples3. Both pseudo-MC128 units
and RQMC4x64 antithetic/bridge paths use seed819. Cash at.25 and1.5 is retained.

Compare sticky-relative-target Delta with full Spot recompile/recalibration at
bumps1e-4/5e-5. Compare frozen leverage with independent European payoff/escrow
reconstruction on a fixed public path surface, applying those same two physical
Spot bumps to its reference initial state. Reconstruct derivative standard errors
from pair-level differences or scramble means using independent two-pass sums.
Derivative/SE tolerance is3e-8*(1+abs(reference)); price reconstruction2e-13.

Check bitwise pricing and Delta/SE across workers1/3, with/without calibration
trace; old local-variance node risk before and after the new methods is unchanged.
Also cover no variance reduction, time-zero cash+proportional events, future
reserves, and explicit spatial-kink rejection versus invariant sticky lookup.
Python adds an independent Asian path/payoff reconstruction, delayed payment,
explicit payoff smoothing, invalid/discontinuous payoffs and immutable APIs.

## Independent Black-limit panel

Six constant-variance limits, variance.04, one-year ATM F=K100/D1, seeds91/1973.
Use RQMC8x2048 antithetic+bridge, particles128/seed429, bandwidth.5/ESS3,
5 target time rows at0/.15/.4/.7/1 and a flat five-node spatial target.
Analytic price7.965567455405804 and Delta.539827837277029 follow from Black.
For both conventions require abs(Delta-reference)<=5*DeltaSE+2e-4 and
abs(price-reference)<=5*priceSE+2e-4, with positive DeltaSE. No IV-bp claim.
Require the two conventions to agree in this flat deterministic-volatility limit.
The ignored numerical test is explicitly executed in release CI on three OSes.

These are finite discrete-derivative and deterministic-limit checks, not an
independent nondegenerate LSV pricing oracle or a proof of production accuracy.
Do not change seeds, references or budgets after viewing numerical outcomes.
Actual run results, source hashes and initial failures belong to the delivery
report/logs; merely defining this protocol is not a passed-test claim.
