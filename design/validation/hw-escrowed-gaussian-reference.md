# Independent Gaussian HW reference: funded strike correction

Date: 2026-09-22. Follow-up to the [S6 evidence](model-boundary-extension.md).

## Failure and cause

The extended price panel's `gaussian_hull_white_independent_price_reference`
reported a maximum absolute IV error of 38.9220 bp in CI run 282 and parent
run 278. The failed case has two-year expiry, zero rate mean reversion,
equity/rate correlation -0.4, a 3% proportional payout and fixed cash 2 at
expiry. See the [original observations](model-boundary-accuracy-followups-2026-09-21.json).

The request builder already uses the funded escrow terminal coordinate, but
this test's reporting call retained the pre-cash forward. Consequently it
inverted each simulated price using a different physical strike and forward.
The production simulation and its requested payoff are not changed by this fix.

After the expiry payout, the reserve is zero and the terminal equity is a
constant multiple of the lognormal risky component. Independently of the
production affine-map implementation, its forward for this one-event fixture is

\[
F_T=(1-\beta)S_0e^{(r-q)T}-D,
\qquad K=F_Te^x.
\]

Here `beta=0.03`, `S0=100`, `r=0.03`, `q=0.01`, `T=2` and `D=2`.
The old reporting call used `(1-beta) S0 exp((r-q)T)` for both the reference
forward and its reconstructed strike. The correction supplies the funded
forward to the existing independent Black inversion. The independent Gaussian
variance quadrature, put/call convention, seeds, path counts, six rate/correlation
combinations and all acceptance budgets remain unchanged.

Reinterpreting the saved first-case prices with Python's independent normal CDF
and bisection gives IV errors 0.02078, -0.003515 and 0.03420 bp at log-strikes
-0.12, 0 and 0.12. This calculation diagnoses the mismatch; it does not replace
running the corrected Rust test over all six combinations.

## Validation

A separate three-platform CI job runs this exact ignored test in release mode,
so a short independent-reference failure is visible before the complete heavy
price panel finishes. The full price and risk gates continue to run unchanged.
Native results for the corrected source are recorded after completion.

The separately inherited one-year 2F constituent timestep ensemble-SE failure
(4.012373 bp against 4 bp) is not caused or fixed by this reference correction.
It remains an explicit acceptance follow-up, with its seeds and budget intact.
Paid-cash stays removed and Bos–Vandermark remains deferred.
