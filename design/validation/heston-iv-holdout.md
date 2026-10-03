# Heston IV disjoint holdout protocol

## Scope and fixed design

Base is PR129, commit `02745d129f73ae421d17cde760342badf20e812f`.
This adds holdout validation without changing the prices, Riccati solves, Black
inversion, optimizer equations, model defaults or calibration target fixtures.
The existing five-grid evaluator is factored into a shared price-only helper.

The retained `iv-holdout.json` fixes its parent training-fixture SHA256, two starts,
parameter bounds and scales, quote coordinates, targets and criteria. Six fits use
the original training data: two each for ordinary Heston, an explicit three-factor
lift and SSVI-to-Rough-Heston. The first two use five scalar variables; SSVI uses six
including H. Calibration uses N128/M512/U128, at most60 iterations/100 evaluations,
residual tolerance2e-6. Each holdout freezes that fitted vector; no holdout-driven
refitting, warm starting, best-start selection or numerical-grid selection occurs.

Both starts are retained and reported, not ranked or selected by holdout error.
All fitting quotes, variable bounds and starts match the parent fixture. SSVI
training sites are T=.25/.75/1.5 and K85/100/115, F100, D.97, theta=.04*T,
rho=-.5, eta=.35, gamma=.5. Ordinary-Heston/lift training targets are the unchanged
independent parent references at T=.25/1 and K80/100/120.

Holdout panels, declared before the first calibration run:

| Target | Unique sites | Region |
| --- | ---: | --- |
| Ordinary Heston | 8 | T=.25/.5/.75/1, K90/110 |
| Explicit 3-factor lift | 8 | Same sites, distinct target model |
| SSVI interpolation | 22 | T=.4/.6/1/1.25 × K87.5/95/105/112.5; T=.25/.75/1.5 × K92.5/107.5 |
| SSVI extrapolation | 12 | T=.125/2 × K85/100/115; T=.25/.75/1.5 × K80/120 |

All use F100/D.97 and OTM option sides. Interpolation is within the training
rectangles in maturity and log-moneyness; extrapolation leaves one coordinate's
range. Every site differs from training and from the other sites in its panel.
There are100 quote rows across both starts, eight panel reports, six fits, and five
IV grid values per row. All rows are printed before the final aggregate assertion;
a numerical error aborts explicitly rather than dropping a quote.

Acceptance limits fixed before the first run:
- All five target residuals <=5 IV bp for interpolation, <=50 IV bp for extrapolation.
- All four grid differences <=0.25 IV bp for every panel.
- Ordinary-Heston/lift fitting runs must achieve the parent's scaled fit goal.
  SSVI need not achieve that stricter0.02-bp goal; its flag is reported unchanged.

These are separate declared limits. In particular, an extrapolation panel accepted
under50bp has **not** thereby passed a5bp limit. Grid checks are finite perturbations,
not a total-error estimate. The evaluator uses the original calibration problem's
base grid. max_stages=1 here; no staged optimizer sees these holdout panels.

## Independent targets and controls

Markov targets use the parent independent adaptive DOP853 affine ODE on contours
Re(z)=0 and1, with P1/P2 Gauss-Legendre inversion, not production half-moment Simpson
inversion. Orders/cutoffs160/192 and240/256 must agree in price within2e-7.
Black IV is inverted independently with erfc and Brent. The sixteen targets are
retained and rechecked without importing the extension. SSVI uses the direct
power-law formula, not the production SSVI adapter. This panel has theta=.04*T;
it does not test arbitrary interpolation of theta curves.

Five fast Rust tests check independent constant-variance IVs, frozen non-initial
parameter vectors, every probe against separate public evaluations for both model
families, unscaled residuals/order, duplicates/training overlap, 65-maturity and
4097-quote rejection, malformed quotes/parameters, strict wing-conditioning failure,
and exact calibration replay after a deliberately failed holdout. Economic-site
checks ignore side, target, discount and simultaneous F/K scaling.

Python exercises the public method signature, corresponding validation errors,
quote-order/scaling behavior, immutable reports and calibration replay. Four
reference/protocol/archive guards check target and tolerance mutations, disjoint
coordinates, missing evidence gates, dependency omissions and required source files.

## Initial results

The first six-fit/eight-panel release run passed all declared criteria without
changing starts, quotes, targets, grids or tolerances. Maxima across both starts,
all quotes and all five grid columns were:

| Panel | Maximum target error, IV bp | Maximum grid change, IV bp |
| --- | ---: | ---: |
| Ordinary Heston interpolation | 0.003167691 | 0.001400495 |
| Explicit-lift interpolation | 0.011407342 | 0.001445433 |
| SSVI interpolation | 1.357186098 | 0.025495533 |
| SSVI extrapolation | 6.081281263 | 0.057745170 |

SSVI extrapolation exceeds5bp despite the training fit's sub-bp approximation in
the parent panel. This is retained as model-fit evidence, not hidden by its separate
50bp acceptance budget. H reaches its declared lower bound in both SSVI fits; no
unconstrained optimum or uniqueness is claimed. The optimizer's strict fit flag
remains false for these SSVI cases.

## Parent CI failure and correction

PR129 workflow run37113636745 failed only in its reference/protocol job111176136943:
`check_heston_iv_calibration.py` imports mpmath, but that job installed only NumPy
and SciPy. Its three OS Rust jobs succeeded. Add `mpmath>=1.3,<2` to that job's
installation command and require the dependency in the source-archive and mutation
checks. The first successful local reference/guard rerun is not evidence of a remote
rerun. This change includes a separately retained minimal CI-fix patch.

## Limits

Synthetic holdout sites test a selected surface, not real quotes, statistical
independence, parameter identification or future market performance. The API can
exclude only this problem's fitting sites; it cannot detect external tuning or
prior holdout inspection. Selection using these results would require a fresh
untouched validation set. No global/unique optimum, total-continuum-error bound,
MC discretization improvement, AAD/VegaKT, recalibrated risk, lift-kernel fitting
or new stochastic-rate/dividend composition is claimed. All APIs remain experimental.

Execution evidence and exact source/patch provenance accompany the delivery report.
