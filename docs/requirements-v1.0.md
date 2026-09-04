# Rust Derivatives Pricing Library — Requirements v1.0

Status: Frozen MVP requirements baseline for implementation
Updated: 2026-09-03
Primary use: Model validation and quantitative research
Initial implementation language: Rust, with Python bindings

Freeze policy: normative changes require a recorded change decision, impact analysis against architecture, tests, serialized compatibility, and the implementation roadmap, followed by an explicit baseline revision. Editorial corrections that do not alter observable behavior may be merged without reopening the baseline.

## 1. Purpose

Build an extensible derivatives pricing library whose calculation core is written in Rust and whose primary interactive interface is Python. The first release targets equity and equity-linked derivatives priced by Monte Carlo methods under Black–Scholes/Black-76 and Local Volatility models.

The library shall prioritize:

1. numerical accuracy and traceability;
2. extensible product, model, and algorithm APIs;
3. deterministic reproducibility for a fixed seed and configuration;
4. efficient CPU execution and parallelism;
5. convenient use from Python for single-trade analysis.

## 2. Scope

### 2.1 In-scope products

- European vanilla options
- American vanilla options
- Digital options
- Barrier options
- Asian options
- Lookback options
- Basket options
- Worst-of options
- Autocallable products

### 2.2 In-scope models

- Black–Scholes for spot-based equity modelling
- Black-76 for forward/futures-based modelling
- Local Volatility constructed from an implied-volatility surface

### 2.3 In-scope pricing methods

- Monte Carlo using pseudo-random numbers
- Quasi-Monte Carlo
- Least-Squares Monte Carlo for early exercise

The architecture shall allow later addition of analytical, tree, finite-difference, and other Monte Carlo engines without changing product definitions.

### 2.4 Explicitly deferred

- Stochastic-volatility, Local Stochastic Volatility, and jump-diffusion models
- PDE, lattice, and general-purpose numerical integration engines
- Portfolio-scale batch infrastructure and distributed execution
- Hard latency service-level objectives
- Bitwise-identical results across different operating systems or CPU architectures
- Construction of an implied-volatility surface directly from raw market quotes

## 3. Market data requirements

The library shall represent:

- spot and forward/futures prices;
- discount-rate term structures;
- continuous dividend-yield or repo term structures;
- discrete dividends;
- calibrated SSVI/eSSVI implied-volatility surfaces;
- Local Volatility surfaces;
- multi-asset correlation matrices.

Market objects shall be immutable during a pricing call and independently replaceable so that bump-and-revalue calculations do not mutate shared state.

### 3.1 Implied-volatility and Local Volatility surfaces

The MVP shall accept calibrated Standard SSVI or eSSVI parameters. Both parameterizations shall implement a common implied-surface interface and be replaceable without changing the Dupire, simulation, product, or risk-reporting layers. Calibration from raw market quotes is outside the MVP.

Standard SSVI shall be represented as one global surface with a maturity-dependent ATM total-variance curve `theta(T)`, a surface-wide constant `rho`, and a surface-wide shape function `phi(theta)` with calibrated global parameters. Only `theta(T)` is interpolated across maturity. It shall use a non-decreasing C1 PCHIP whose analytic first derivative supplies `d theta/dT`; `rho` and the shape-function parameters shall not be interpolated by maturity.

The MVP shall provide two global `phi` families:

\[
\phi_{\mathrm{power}}(\theta)=
\frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}},
\qquad \eta>0,\;0<\gamma<1,
\]

and

\[
\phi_{\mathrm{Heston}}(\theta)=
\frac{1}{\lambda\theta}
\left(1-\frac{1-e^{-\lambda\theta}}{\lambda\theta}\right),
\qquad \lambda>0.
\]

Both implementations shall provide analytic derivatives with respect to `theta`, stable small-`theta` evaluation, and validation of the applicable Gatheral–Jacquier static-arbitrage conditions over the surface domain.

eSSVI shall accept consistent maturity slices `(theta_i, psi_i, rho_i)`, where `psi_i = theta_i * phi_i`. Between consecutive slices it shall use the arbitrage-preserving interpolation from Corbetta–Cohort–Laachir–Martini: linear interpolation of `theta`, `psi`, and `rho * psi`, with `rho` recovered as `(rho * psi) / psi`. It shall not apply generic PCHIP independently to `rho` or `psi`.

At the eSSVI short end, with `lambda=T/T_1`, the baseline extrapolation shall be `theta=lambda*theta_1`, `psi=lambda*psi_1`, and `rho=rho_1`. Beyond the final maturity, `psi` and `rho` shall remain fixed while `theta` increases linearly using a configured non-negative terminal ATM forward-variance slope. Standard SSVI shall use the corresponding linear short-end `theta` rule and terminal forward-variance extrapolation while retaining its global shape specification.

The canonical horizontal coordinate shall be log-forward-moneyness

\[
k=\log(K/F_T),
\]

and the implied surface shall expose total variance \(w(k,T)\) together with the time and log-moneyness derivatives required by Dupire.

The Local Volatility builder shall:

1. evaluate SSVI/eSSVI total variance and its required derivatives;
2. apply the Dupire transformation on a configured \((k,T)\) or corresponding \((S,t)\) grid;
3. store the resulting Local variance grid;
4. use bilinear interpolation of that Local variance grid during path simulation.

The baseline Local-variance grid coordinate shall be `x=log(S/F(T))`, aligned with the implied-surface coordinate. Outside the horizontal grid range, Local variance shall remain constant at the nearest boundary. Boundary use shall continue calculation but be counted and reported with the maximum excursion beyond the configured grid.

Bilinear interpolation shall not be differentiated as though it were a globally twice-differentiable implied-volatility surface. Dupire derivatives shall come from the SSVI/eSSVI representation, not from second derivatives of a bilinear IV surface.

The surface component shall expose separately:

- calibrated SSVI/eSSVI parameters;
- the configured reporting \((K,T)\) grid generated from the calibrated surface;
- interpolation and extrapolation policy;
- the derived Local Volatility surface;
- validation warnings or errors, including invalid or unstable Dupire results.

If the calculated Local variance is non-finite, non-positive, or outside configured bounds, the builder shall clamp it to explicit floor/cap values, continue the calculation, and emit a structured warning containing at least the location, original value, applied value, and reason. Clamp bounds must be part of the reproducible calculation configuration; silent clamping is forbidden.

Every Local Vol calculation shall explicitly provide finite Local-variance floor and cap values satisfying `0 < floor <= cap`. The MVP shall have no hidden library-wide or surface-wide default bounds.

The canonical Local-variance surface input shall contain explicit, strictly increasing maturity nodes and explicit, strictly increasing log-forward-moneyness nodes in the continuous-martingale \(f\) coordinate. A convenience helper shall accept separate left- and right-tail probabilities, evaluate the corresponding analytic \(f\)-distribution quantiles at every proposed maturity node, and form one rectangular horizontal range from the minimum left and maximum right log-forward-moneyness values across all maturities. Explicit left and right additive \(\Delta k\) padding shall be applied after this envelope is formed.

Within that common range, the helper shall generate a non-uniform horizontal grid using independently parameterized piecewise-sinh mappings to the left and right of ATM, with ATM included exactly as a node. It shall also accept the node allocation on each side and all mapping-shape parameters explicitly. Its returned nodes become ordinary explicit inputs and are stored in the normalized plan; later simulation-grid or surface changes shall not silently regenerate them.

The Local-variance floor and cap shall be mandatory values in every Local Vol calculation configuration. The MVP shall not silently supply library-wide or surface-wide defaults. Validation shall require finite values satisfying `0 < floor <= cap`.

The precise admissibility tolerances, eSSVI terminal forward-variance slope configuration, and quantitative defaults for the two tail probabilities, two paddings, side-specific node counts, and piecewise-sinh shape parameters remain to be specified. These policies must be replaceable rather than embedded in the pricing engine.

### 3.2 Dates, curves, and dividends

The public Rust and Python APIs shall accept actual calendar dates for valuation dates, expiries, exercise dates, observation dates, payment dates, curve pillars, and dividend events. At the calculation boundary, dates shall be converted to internal `f64` year fractions relative to an explicit valuation date using a configured day-count convention.

The MVP shall provide `ACT/365F` and `ACT/360`. Each curve, surface, schedule, or model input that converts dates to times shall identify its convention explicitly; `ACT/365F` may be the API default but must remain visible in the normalized configuration.

The MVP calendar shall support Saturday/Sunday weekends plus a user-supplied immutable set of holiday dates. Built-in exchange or national holiday databases are outside the MVP. The complete custom holiday set, or a stable digest plus retained source data, shall participate in replay metadata.

Internal pricing algorithms shall consume normalized times and shall not perform hidden date conversions. The normalized calculation configuration must retain the source dates, valuation date, day-count convention, and resulting year fractions for audit and replay.

The baseline deterministic discount curve shall be represented by positive discount-factor pillars and shall use log-linear interpolation:

\[
\log D(t) = (1-\lambda)\log D(t_i)+\lambda\log D(t_{i+1}).
\]

Curve validation shall reject duplicate or unsorted pillars, non-positive discount factors, and missing valuation anchors. Extrapolation before the first and after the last pillar remains to be specified explicitly.

Curve evaluation outside the pillar range shall always use flat-forward extrapolation: the boundary slope of `log D(t)` is continued beyond the first or last interpolation segment. The valuation anchor remains `D(0)=1`; negative times are invalid rather than extrapolated. Extrapolation use shall be counted in diagnostics with the affected curve and requested time range.

Schedule generation shall support `Unadjusted`, `Following`, `Modified Following`, and `Preceding` business-day adjustments using the selected weekend-plus-custom-holiday calendar. Both unadjusted and adjusted dates and the convention shall remain available in the normalized plan.

The discrete-dividend schedule shall support both:

- fixed cash dividends; and
- proportional dividends.

Dividend inputs shall be validated strictly: every fixed cash amount shall satisfy \(D_i\ge 0\), and every proportional coefficient shall satisfy \(0\le\beta_i<1\). Values outside these ranges are invalid market data in the MVP rather than generalized affine cash-flow events.

Each dividend event shall have an ex-date and amount or proportion. Pay-date metadata may also be retained, but spot-path adjustment shall occur at the ex-date. Following Guyon and Henry-Labordère (SSRN 1885032), the normalized event model shall be the affine dividend

\[
D(t_i,S^-)=\alpha_i S_0+\beta_i S^-,
\]

so that

\[
S^+=(1-\beta_i)S^- - \alpha_iS_0.
\]

A proportional input maps to \(\beta_i\); a fixed-cash input is normalized as \(\alpha_i=D_i/S_0\) for each compiled calculation. The original cash amount \(D_i\), not \(\alpha_i\), is the fixed market datum. Under the Spot bumps used for Delta or Gamma, \(D_i\) shall remain unchanged and \(\alpha_i\) shall be recomputed from the bumped initial Spot. The original quote type and value shall be retained rather than replaced by normalized coefficients. If both components occur on the same ex-date, this affine formula is equivalent to applying the proportional component before the cash component.

The path engine shall represent the event exactly rather than smearing discrete dividends into a continuous yield. If any simulated path has non-positive spot immediately after a dividend event, the entire calculation shall terminate with a typed numerical/model error. The error shall identify the dividend event, path identity, pre-jump spot, both affine coefficients, the original dividend quotes, and attempted post-jump spot. The MVP shall not floor, absorb, or silently remove such a path.

The dividend-aware Local Volatility construction shall use the paper's continuous-local-martingale transformation

\[
S_t=A(t)S_0+B(t)f_t,
\]

where deterministic \(A\) and \(B\) absorb carry and the affine dividend jumps, and \(f_t\) is continuous across ex-dates. At a dividend date they shall obey

\[
A^+=(1-\beta_i)A^- - \alpha_i,
\qquad
B^+=(1-\beta_i)B^-.
\]

Between ex-dates, the surface and Local Volatility construction shall use the dividend-aware Dupire relation, and at each ex-date the modeled call surface shall satisfy the paper's matching condition

\[
C(t_i,K)=\mathbb{E}\!\left[D_{0,t_i}\left(S^- -D(t_i,S^-)-K\right)^+\right].
\]

The compiled market shall validate that the selected SSVI/eSSVI representation and dividend transformation are mutually consistent with this condition to the configured tolerance. The continuous-martingale \(f\)-surface may use its analytic maturity derivative continuously across an ex-date. Contract values and Spot-strike mappings shall respect the event boundary, but the MVP shall not construct or mark a separate implied-volatility surface against the discontinuous Spot process.

When discrete dividends are present, calibrated SSVI/eSSVI parameters and all implied-volatility marks shall be supplied in the continuous-martingale \(f\) coordinate. Its horizontal coordinate shall be log-forward-moneyness in \(f\). Contract strikes and call values in Spot space shall be related through the deterministic affine transformation, but an implied volatility against discontinuous Spot is neither solved nor reported. Direct calibrated SSVI/eSSVI input in Spot coordinates is outside the MVP.

The Local Volatility path engine shall simulate \(f\) with Log-Euler and reconstruct \(S=A S_0+Bf\) at every contractual or diagnostic observation. It shall not evolve a separate jumped Spot state in the MVP. Positivity checks apply to every reconstructed Spot required by the plan, including immediately after each ex-date update of \(A\) and \(B\).

If an option expiry and dividend ex-date are the same calendar date and normalized event time, the dividend event shall occur first and expiry observation and settlement determination shall use the post-dividend Spot. This ordering is an MVP-wide convention and shall be recorded in the normalized product plan.

The forward used in log-forward-moneyness \(k=\log(K/F_T)\) shall be produced by a documented forward provider using the supplied spot, curves, affine dividend data, and the same \((A,B,f)\) transformation. SSVI/eSSVI and Local Volatility components shall not calculate a conflicting forward internally.

### 3.3 Correlation term structure

Multi-asset models shall accept a term structure of correlation matrices. Effective date/times shall be unique and strictly increasing after normalization. The term structure shall be right-continuous: a matrix effective at time $t_j$ applies on $[t_j,t_{j+1})$. Thus an interval ending at a change time uses the old matrix, while an interval starting at that time uses the new matrix. The simulation grid shall include every in-horizon correlation change time.

At least one matrix shall be effective at or before the valuation time; absence of such an initial matrix is an error and shall never imply Identity correlation. The latest matrix effective at or before valuation shall cover the initial simulation interval, and the final supplied matrix shall be extended piecewise-constantly through the last required simulation time. No matrix need be repeated merely to cover the terminal time. The normalized plan and diagnostics shall identify the selected matrix index and effective time for every simulation interval.

All matrices shall use one explicit, validated underlying ordering. Before simulation, each matrix shall be checked for finiteness, shape, symmetry, unit diagonal, bounded entries, and positive semidefiniteness. After canonicalization, every off-diagonal entry shall satisfy $-1\le C_{ij}\le1$ exactly; an out-of-range value is an error and shall not be clamped, even when its excess is within another configured tolerance. Singular positive-semidefinite matrices are valid; strict positive definiteness is not required. A matrix that fails validation beyond configured numerical tolerance shall produce an error. The MVP shall not add diagonal jitter, project to a nearest-correlation matrix, eigenvalue-clip, or otherwise repair an invalid matrix.

The required baseline factorization shall be an unpivoted, PSD-aware Cholesky algorithm in the declared underlying order. It shall not permute underlyings or replace the requested factorization with an eigendecomposition. A materially negative Schur-complement pivot shall be an error. A pivot classified as numerical zero may represent a singular PSD direction only when every associated remaining Schur-complement residual is also numerically zero; otherwise factorization shall fail. The implementation shall report numerical rank, raw pivot diagnostics, zero-pivot indices, and maximum residual rather than silently manufacture a positive pivot.

Every multi-asset calculation shall explicitly provide a finite, non-negative `CorrelationToleranceConfig` containing `symmetry_abs_tol`, `diagonal_abs_tol`, `psd_abs_tol`, `psd_rel_tol`, `zero_pivot_abs_tol`, and `zero_pivot_rel_tol`. There shall be no hidden global default in the normalized plan. Relative thresholds shall use a documented deterministic matrix scale and shall be combined with their absolute threshold by `max(abs_tol, rel_tol * scale)`. These tolerances classify floating-point equality and numerical zero only; they do not authorize an economic repair. All configured tolerances, effective thresholds, validation extrema, factorization method/version, and rank diagnostics shall be returned in calculation diagnostics.

After the raw matrix passes its symmetry and diagonal checks, it shall be canonicalized deterministically as $C_{ij}\leftarrow(C_{ij}+C_{ji})/2$ for each off-diagonal pair and $C_{ii}\leftarrow1$. This within-tolerance canonicalization is not an invalid-matrix repair: a raw deviation outside tolerance shall fail before canonicalization. Diagnostics shall report the maximum raw asymmetry, maximum diagonal deviation, maximum absolute element adjustment, and fingerprints of both raw and canonical matrices.

The relative-tolerance scale shall be computed from the canonical matrix as

\[
s_C=\max\!\left(1,\lVert C\rVert_\infty\right)
=\max\!\left(1,\max_i\sum_j|C_{ij}|\right),
\]

using ascending row and column order and a fixed scalar `f64` accumulation sequence. It shall not depend on parallel reduction, BLAS, or an eigenvalue calculation.

A rank-deficient Cholesky factor shall retain shape $n\times n$ in the canonical underlying order. Every numerical zero-pivot direction shall be represented by a zero column; the factor shall not be compressed to $n\times r$, pivoted, or converted to an eigenbasis. Consequently, Brownian factor count and random-dimension mapping remain $n$ for every correlation period even when numerical rank changes.

## 4. Product representation

Products shall describe contractual events and cash flows independently of the stochastic model and simulation engine.

The representation must support:

- one or multiple underlyings;
- observation and exercise schedules;
- payment dates and discounting;
- path-dependent state;
- discrete and continuous barrier monitoring;
- early exercise;
- callability and coupon conditions;
- basket and worst-of aggregation.

Public product schedules shall use dates. Engines may use a compiled product representation containing normalized year fractions and preordered events.

Product validation shall reject inconsistent schedules, invalid barriers, missing underlyings, and malformed payoff parameters before simulation begins.

### 4.1 Event and Payoff graph

Custom products shall be represented by a typed, composable Event/Payoff directed acyclic graph that can be constructed from both Rust and Python. The graph shall support at least:

- constants, arithmetic, minimum/maximum, and comparisons;
- spot and multi-asset observations at named dates;
- running average, minimum, maximum, and path-state updates;
- conditional cash flows, coupons, barriers, knock-in/knock-out state, and call events;
- basket and worst-of aggregation;
- exercise values and continuation-value hooks for LSM; and
- payment dates, currencies, and discounting references.

Before pricing, the public graph shall be validated and compiled into an immutable execution representation with resolved underlying indices, normalized event times, deterministic node ordering, and preallocated state slots. Cycles, invalid types, observations after dependent payments, ambiguous same-time event ordering, and unavailable model observables shall be rejected before simulation. The versioned Source graph is the serialization boundary. Compiled Payoff tape bytes, native indices, cache offsets, and function/dispatch representations shall not be accepted as persisted public input in the MVP. Loading a request shall validate and deterministically recompile its Source graph under an explicit graph-schema version and compiler/tape-ABI version. Replay metadata shall include fingerprints of both the canonical Source graph and the resulting logical Compiled tape.

The Compiled Payoff tape shall be a dense, topologically ordered array of closed Rust opcode-enum values dispatched by direct `match`. Per-opcode trait objects and persisted or caller-supplied function pointers shall not appear in the hot path. Source `NodeId`, compiled opcode index, value/state slot index, cache slot index, and opcode operands shall use explicit `u32` logical identifiers. Counts or assignments that would exceed `u32::MAX` shall fail compilation with a typed capacity error before allocation. A logical identifier shall be checked before conversion to platform-native `usize` at array access; unchecked truncation or serialization of `usize` is forbidden.

Topological ordering shall use deterministic Kahn sorting. The Ready set shall select the smallest persisted Source `NodeId` first, and each opcode's outgoing dependency edges shall be visited in ascending destination `NodeId` order. Source node identifiers shall be unique and stable within the serialized graph. The resulting forward tape order, slot assignment, reverse traversal order, and logical-tape fingerprint shall therefore be independent of hash-map iteration, memory address, thread scheduling, and Rust collection implementation.

Rust and Python graph builders shall assign Source `NodeId` values monotonically from zero using checked `u32` increment. An ID shall never be reused during a builder's lifetime, including after an uncommitted node is removed. A builder opened from a serialized graph shall preserve all existing IDs and begin new allocation at one greater than the greatest existing ID; overflow shall be a typed builder error. Low-level deserialization may contain gaps but shall reject duplicate IDs and shall never renumber them implicitly.

Graph compilation shall apply a versioned soft-resource-limit policy before large allocations. It shall separately limit Source nodes, Compiled opcodes, dependency edges, declared outputs, Event instances, value slots, state slots, reverse-cache slots, and estimated total workspace bytes. The policy shall expose named finite count/size limits, versioned defaults, and an explicit per-request override for each field. Counts shall be evaluated after the applicable validation/compilation stage without conflating removed Dead nodes with emitted opcodes. The normalized plan and diagnostics shall contain the policy version, defaults, requested overrides, effective limits, observed counts, workspace-estimation inputs, and estimated bytes. An override may intentionally raise a soft limit, but it shall never bypass `u32` identifier bounds, checked `usize` size arithmetic, platform allocation limits, or other hard safety validation. Exceeding an effective soft limit or a hard bound shall return distinct typed errors identifying the field and observed/effective values; the compiler shall not truncate the graph.

The MVP compiler shall not perform common-subexpression elimination, including between structurally identical pure subgraphs. Distinct Source `NodeId` values shall retain distinct identity and source-to-compiled mapping; graph authors obtain sharing only by explicitly reusing one Source node. This rule prevents an optimizer from silently changing diagnostic identity or adjoint fan-in order.

The compiler shall constant-fold a node only when its complete reachable operand subgraph consists of literal constants and pure foldable opcodes. It shall evaluate those opcodes once in the same canonical topological order, with the same `f64` primitive, operand order, branching convention, smoothing rule, and FMA policy used by runtime execution. It shall not reassociate expressions, apply algebraic identities, or merge equal folded values. Source-node provenance and the exact folded `f64` bit pattern shall remain in the source-to-compiled map and logical-tape fingerprint.

Every numeric Source literal shall be finite. If an otherwise eligible constant fold produces NaN or positive/negative Infinity at any intermediate or final node, compilation shall fail with a typed error identifying the Source `NodeId`, opcode, and operand/result bit patterns. The compiler shall not preserve a non-finite folded literal or defer the same operation to runtime. This rule shall apply during whole-Source validation even when the offending node is later found unreachable from a Product output.

The compiler shall validate the entire Source graph before output reachability removes work: identifiers, references, types, schedules, units/currencies, event ordering, literal finiteness, cycles, and constant-foldable operations in dead subgraphs remain subject to the same errors. Only after successful validation shall it remove nodes unreachable from the explicitly declared Product outputs when constructing the Compiled tape and liveness/cache layouts. Diagnostics and fingerprints shall retain the full Source graph and a deterministic list of removed Source `NodeId` values.

Canonical fingerprints shall represent every finite `f64` value by its exact IEEE-754 binary64 `to_bits()` value, written to the canonical fingerprint byte stream in fixed big-endian `u64` byte order. No decimal formatting, locale, native endianness, tolerance rounding, or arithmetic normalization shall enter the fingerprint. In particular, `+0.0` and `-0.0` shall remain distinct. Non-finite inputs shall be rejected by the applicable domain validation before fingerprint construction.

The fingerprint byte stream shall use a versioned self-delimiting framing independent of the user-facing serialization format. Every encoded value shall have a stable type tag followed by a fixed-width unsigned payload-length prefix and payload; all integers and lengths shall use fixed-width big-endian encoding. Struct fields shall follow schema order with stable field identifiers, sequences shall preserve their canonical order and encode their count, and maps shall be sorted by canonical encoded key bytes. Missing optional fields and explicitly present values shall have distinct encodings. Concatenation without framing and native Rust memory representations are forbidden.

Strings in the canonical stream shall be encoded as their exact UTF-8 bytes with their byte length. The encoder shall perform no Unicode normalization, case folding, trimming, locale conversion, or path normalization. Consequently, canonically equivalent Unicode spellings and differently cased identifiers remain different inputs and produce different fingerprints. A Python string that cannot be converted to valid UTF-8 for the Rust API shall be rejected; individual domain validators may additionally reject empty strings or forbidden control characters but shall not silently rewrite accepted text.

Plan, Source-graph, logical-tape, market, product, and numerical-policy fingerprints shall use BLAKE3 with the full 256-bit output. Each fingerprint input shall begin with a fixed artifact-specific domain tag and its schema/ABI version so that identical payload bytes in different domains cannot alias by construction. The raw 32-byte digest is canonical; user-facing text shall use lowercase 64-digit hexadecimal prefixed by the algorithm identifier `blake3-256:`. Fingerprints identify configuration and replay state but are not digital signatures or authenticity proofs.

Within one path lane, contributions to an adjoint slot shall be accumulated by scalar `f64` addition while traversing live opcodes in decreasing tape order and each opcode's operands in their declared order. Repeated references to the same operand shall likewise be added in operand-position order. This fan-in shall not use atomics, parallel reduction, compensation, reassociation, FMA contraction, or a collection whose iteration order can vary. Neumaier compensation remains required for the later cross-path result reduction, not for per-path graph-slot fan-in.

The compiled graph shall expose the differentiable operations needed by AAD. Arbitrary per-path Python callbacks are not part of the MVP because they would break Rust-side execution, parallelism, and the controlled differentiation graph.

### 4.2 Currency scope

Each MVP valuation shall use exactly one reporting/payment currency. All product cash flows, discount curves, collateral/carry inputs, and monetary payoff constants supplied to one pricing request shall be denominated in that currency. Currency mismatches shall be rejected during graph or market-context validation.

Deterministic FX conversion, stochastic FX modelling, quanto adjustment, and cross-currency discounting are outside the MVP. The graph may retain an explicit currency identifier so these capabilities can be added without changing cash-flow identity.

### 4.3 Asian and Lookback contracts

The MVP Asian product shall be an arithmetic average-price option with Call and Put variants. Its contractual payoff at expiry shall be

\[
\max(A-K,0) \quad\text{or}\quad \max(K-A,0),
\]

where \(A=\sum_i w_i S_i\) is formed from an explicit schedule of observation dates and weights. Every weight shall be finite and non-negative, and the supplied weights shall sum to one within a versioned validation tolerance. The engine shall reject a schedule outside that tolerance rather than silently renormalize it. The contract shall separately accept known historical fixings attached to scheduled observation dates, allowing the same product type to represent both forward-starting and partially fixed trades. Strike, expiry, payment date, underlying, currency, option side, schedule, weights, and known fixings shall be explicit fields. Arithmetic average-strike and geometric-average variants are outside the MVP product set, although the Event/Payoff graph must remain capable of later extensions.

An Asian observation strictly before the valuation date shall require a known fixing, and an observation strictly after it shall be an unknown model observation. Every observation on the valuation date shall explicitly declare `Known` or `Unknown`; no date-only heuristic shall choose the status. A `Known` observation requires one finite fixing value, while an `Unknown` observation shall not carry a fixing value. The normalized plan and result metadata shall retain this classification.

The MVP Lookback product shall be a fixed-strike option with Call and Put variants. The Call payoff shall use the contractual observed maximum and the Put payoff the contractual observed minimum:

\[
\max(M-K,0), \qquad \max(K-m,0).
\]

Strike, expiry, payment date, underlying, currency, option side, discrete monitoring schedule, and any required historical extremum shall be explicit fields. If one or more declared monitoring dates precede the valuation date, a fixed-strike Lookback Call shall require the historical running maximum and a Put shall require the historical running minimum over those past declared observations. This single scalar is the canonical historical input; the MVP shall not require or accept the full past fixing series in the core product object. The historical extremum must be finite and positive.

The MVP Lookback shall observe extrema only at its declared monitoring dates; continuous-monitoring correction and between-date extremum reconstruction are outside the MVP. Initial Spot, valuation-date Spot, and trade-start Spot shall not be added implicitly: each participates only when its date is an explicit monitoring date with the applicable known/unknown observation status. Floating-strike and partial-lookback payoff forms are also outside the MVP product set.

Asian and Lookback schedules shall compile to the same normalized event ordering used by the generic graph. In particular, an observation sharing a date with a dividend ex-date observes the post-dividend Spot under the library-wide event convention. Contract validation shall reject duplicate or unordered schedule entries after normalization, non-finite fixing values or weights, fixings not associated with the contract schedule, and payment before payoff determination.

Known Asian fixings and a Lookback's historical running extremum are contractual state, not live market data. They shall remain numerically fixed under Spot, volatility, curve, dividend, and correlation bumps and shall carry zero adjoint. Both AAD and bump-and-revalue shall apply this rule, and risk metadata shall identify the fixed historical state used by the calculation.

### 4.4 Basket and Worst-of underlyings

A standard Basket builder shall aggregate scaled Spot components as

\[
B(t)=\sum_{i=1}^{n} w_i\frac{S_i(t)}{c_i},
\]

where each underlying identifier, weight \(w_i\), and positive scale \(c_i\) is explicit. Choosing \(c_i=1\) represents a Raw-Spot component, while choosing a contractual reference or initial level represents a normalized Performance component. Mixed scales are permitted when intentionally supplied. Weights need only be finite: positive, zero, and negative weights and arbitrary weight sums are valid. The standard builder shall not normalize, rescale, or otherwise reinterpret them.

A standard Worst-of builder shall compute

\[
W(t)=\min_i \frac{S_i(t)}{R_i},
\]

using one explicit, finite, strictly positive contractual reference level \(R_i\) per underlying. It shall not use Raw-Spot minima or infer reference levels from the current market snapshot. Underlying order shall be explicit and shall resolve to the same order used by the correlation term structure and model state.

The MVP Basket option shall provide fixed-strike Call and Put payoffs on the Basket value,

\[
\max(B(T)-K,0), \qquad \max(K-B(T),0),
\]

and the MVP Worst-of option shall provide fixed-strike Call and Put payoffs on the Worst-of Performance,

\[
\max(W(T)-K,0), \qquad \max(K-W(T),0).
\]

For each, observation date, strike, option side, expiry, payment date, currency, and component definitions shall be explicit. More complex coupons, barriers, and path-dependent uses of the same Basket or Worst-of observable shall be composed through the Event/Payoff graph or the Autocallable builder.

All Basket scales and Worst-of reference levels belong to the product definition. They shall remain fixed under Spot and every other market bump and shall carry zero adjoint; neither Sticky-moneyness conventions nor Spot-relative bump conventions may alter them. Contract validation shall reject duplicate underlying identities within one standard builder, missing scales or references, non-finite weights, non-positive scales or references, and currency/model inconsistencies before simulation. Equivalent or more general signed combinations remain constructible directly through the Event/Payoff graph.

### 4.5 Autocallable contracts

The standard MVP Autocallable shall use Worst-of Performance \(W(t)=\min_i S_i(t)/R_i\) as its call-condition observable. Each call observation shall explicitly provide its observation date, call barrier, call redemption amount or notional rule, call coupon amount, and payment date. Call barriers are independent finite positive levels; they may be constant, step down, step up, or follow a non-monotone schedule. The library shall not infer or enforce monotonicity.

The standard builder shall support all of the following coupon components:

- a date-specific coupon paid when the trade autocalls at that observation;
- a conditional periodic coupon with no memory; and
- a conditional periodic coupon with memory.

Every conditional coupon observation shall explicitly provide its observation date, Worst-of coupon barrier, coupon amount, and payment date. The product shall explicitly select `NoMemory` or `Memory`; it shall not infer memory from repeated amounts or schedules. A no-memory failed coupon is permanently lost. In Memory mode, each failed period's explicit nominal coupon amount shall be added to a path-state balance without interest or other accrual. When a later coupon condition succeeds, the engine shall pay that balance plus the current period's nominal coupon and reset the balance to zero.

At a date carrying both coupon and call observations, the engine shall evaluate the shared post-dividend Worst-of value once, determine the conditional coupon, update or release its Memory balance, and create its coupon cash flow before evaluating the call condition. If the call condition then succeeds, the engine shall also create the date-specific call coupon and redemption cash flow and terminate all later product events. This ordering is fixed for the standard builder and retained in the compiled event plan.

If the trade has not autocalled, its maturity redemption shall be conditional on an explicit final Worst-of barrier \(H_F\), observed once at maturity. Under exact contractual logic, touch belongs to the protected branch and the redemption is

\[
R_T=
\begin{cases}
N, & W(T)\ge H_F,\\
N\,W(T), & W(T)<H_F,
\end{cases}
\]

where \(N\) is the explicit Notional. There is no pathwise knock-in state and no monitoring of the final barrier before maturity. The primary smoothed Price/AAD calculation shall replace the final predicate with the common compact C2 indicator while leaving both branch amounts explicit.

The product shall independently specify the treatment of an unreleased Memory balance on Autocall termination and on Maturity termination. Each setting shall be either `Release` or `Forfeit`; no default or coupling between the two events is permitted. `Release` creates a cash flow for the full nominal balance on the applicable termination payment date, while `Forfeit` clears it without payment. This termination policy does not change the separately configured current conditional coupon or call coupon.

A partially observed but still active Autocallable shall be initialized from an explicit state snapshot containing: `Active` lifecycle state, the finite non-negative unpaid Memory-coupon balance, and every already determined but unpaid cash flow with amount, currency, and payment date. Past call and coupon observations shall not be replayed from full historical paths. Snapshot cash-flow amounts and the Memory balance are fixed contractual state under every market bump and carry zero adjoint; pending cash flows nevertheless retain their ordinary discount-curve sensitivity through their future payment dates.

Call and coupon observations strictly before the valuation date must already be reflected in the snapshot and shall not remain executable events. Observations strictly after it are unknown model observations. Each call or coupon observation on the valuation date shall explicitly declare `Known` or `Unknown`; date-only logic shall not infer the status. A known result is fixed contractual state, while an unknown result is evaluated by the model under the normalized time-zero convention.

Exact call and coupon predicates are inclusive: Worst-of Performance equal to the applicable barrier satisfies the condition, \(W\ge H\). In smoothed mode the common symmetric quintic indicator therefore has value one half at equality, and results shall identify the smooth rather than contractual boundary semantics.

All call and coupon conditions shall use the configured compact C2 smoothing in the primary smoothed Price/AAD calculation and the same half-width convention as other graph indicators. The standard Autocallable shall compile into ordinary observation, state, conditional cash-flow, and termination nodes so that exact/smoothed modes, AAD, and bump validation are shared with custom graphs.

## 5. Simulation requirements

### 5.1 Path generation

The simulation layer shall support:

- single-asset and correlated multi-asset paths;
- a user-supplied seed;
- configurable independent sampling-unit count and time grid;
- pseudo-random and quasi-random sequences;
- deterministic results for the same seed and complete calculation configuration;
- CPU parallel execution;
- variance-reduction components that can be enabled independently.

Pseudo-Monte Carlo shall use **Philox4x32-10** with the following versioned word-level ABI. The master seed is an unsigned `u64` and shall be split directly into the two Philox key words as $k_0=\operatorname{low32}(seed)$ and $k_1=\operatorname{high32}(seed)$. This definition is by unsigned bit extraction and is independent of host byte order.

For zero-based path index $p\in u64$, zero-based global logical dimension $d$, and explicit domain identifier $q\in u32$, the four counter words shall be

\[
(c_0,c_1,c_2,c_3)=
(\operatorname{low32}(p),\operatorname{high32}(p),\lfloor d/4\rfloor,q).
\]

One Philox invocation therefore produces the four consecutive dimensions $4c_2,\ldots,4c_2+3$ for the same path, and dimension $d$ consumes output lane $d\bmod 4$. The compiled plan shall reject $d\ge 2^{34}$, so that the dimension-block index fits in `u32`. `SimulationPlan` shall deterministically linearize time, factor, Brownian-bridge, and any other model-driver coordinates into this global dimension before execution; product structure shall not be hashed or packed implicitly into the counter.

The `u32` domain is an explicit substream namespace with registry version 1: `Valuation = 0`, `LsmTrain = 1`, `RqmcScramble = 2`, and `Diagnostics = 3`. Base valuation, AAD replay, and common-random-number bump valuations shall use `Valuation`; LSM fitting shall use `LsmTrain`; construction of randomized-QMC LMS matrices and digital shifts shall use `RqmcScramble`; standalone numerical diagnostics shall use `Diagnostics`. The registry version and domains actually used shall be included in replay metadata. Values not defined by the selected registry version shall be rejected rather than silently interpreted or hashed.

Philox rounds and word arithmetic shall use defined wrapping `u32` operations, making integer outputs and lane selection independent of platform, endianness, worker scheduling, block scheduling, and traversal order. Known-answer tests shall cover seed splitting, counter packing, all four lanes, every registry-v1 domain, boundary path values, and the largest valid dimension.

Each Philox output word $x\in\{0,\ldots,2^{32}-1\}$ shall be mapped to an open-interval uniform by

\[
u=(\operatorname{f64}(x)+0.5)2^{-32}.
\]

The cast, addition, and multiplication shall occur in that order. This one-word midpoint mapping produces exactly $2^{-33}\le u\le 1-2^{-33}$, never consumes a second lane, and shall have boundary known-answer tests.

The QMC engine shall embed the complete **Joe–Kuo 6.21201** direction-number set and support dimensions 1 through 21,201 inclusive. A compiled request whose effective dimension exceeds 21,201 shall fail before execution; dimensions shall not wrap, repeat, or fall back to another generator. The table version and an integrity checksum shall be part of build and replay metadata.

Each randomized replicate shall apply **linear matrix scrambling (LMS) plus digital shift**. Every `(scramble, dimension)` pair shall receive its own independently generated 32-by-32 binary LMS matrix and its own independently generated 32-bit digital-shift word. Neither object may be shared across dimensions or scrambles.

LMS bit vectors shall use MSB-first indexing: bit index 0 is the `u32` most-significant bit. The binary matrix shall be unit lower-triangular in this indexing, with diagonal entries fixed to one, entries above the diagonal fixed to zero, and independently randomized entries below the diagonal. Thus output bit row $r$ may depend only on input bits $0,\ldots,r$, and the resulting matrix is nonsingular by construction. The matrix-vector product and shift addition shall be over $GF(2)$, with addition implemented as XOR. Matrix orientation, bit numbering, and operation order are versioned parts of the replay contract.

RQMC shall accept an explicit `u64` master scramble seed distinct from the pseudo-Monte Carlo master seed. It shall form the Philox key by the same unsigned split `(low32, high32)` and generate scramble data under `RandomDomain::RqmcScramble = 2`. For zero-based scramble index $a$, zero-based Sobol dimension $j$, and slot $s$, it shall use Philox path $p=a$ and global dimension $d=32j+s$. Each dimension owns exactly 32 Philox words: slot 0 is its digital shift; for rows $r=1,\ldots,31$, slot $r$ supplies the candidate random bits below that row's diagonal. Row 0 is the deterministic first identity row and consumes no separate random content. Unused candidate-word bits shall be masked away before the diagonal bit is inserted.

During `SimulationPlan` compilation, the engine shall generate and materialize the LMS row masks and shifts for every requested scramble and every effective Sobol dimension, and for no unused dimensions. The resulting table shall be immutable during execution; lazy first-use generation and per-point regeneration are not allowed. Checked size arithmetic and allocation shall occur before generation, and allocation failure shall be a structured compilation error. Replay metadata shall contain the independent scramble seed, coordinate-ABI version, effective dimension, scramble count, and a deterministic checksum of the materialized table.

Each scramble shall produce one replicate estimate, and the final price shall be the average of the replicate estimates. The reported QMC standard error shall be estimated from variation between independent scrambled replicates, not from treating points within one Sobol net as IID samples. The default number of independent scrambles shall be **16**, while remaining explicitly configurable.

The QMC configuration shall expose and record:

- Sobol points per replicate;
- number of independent scrambles;
- master scramble seed;
- scrambling method;
- effective dimension and dimension ordering;
- whether Brownian-bridge construction is enabled.

The number of Sobol points per scramble shall be an exact positive power of two, $N=2^m$; compilation shall reject any other value. A public helper may return the smallest representable power of two not less than a requested target, but shall not silently replace the explicit calculation input. Each scramble shall use the complete zero-based point-index set $0,1,\ldots,N-1$, including index zero. The pre-scramble all-zero point is valid: LMS leaves it zero and the configured digital shift supplies its randomized word.

Both pseudo-Monte Carlo and RQMC shall transform open-interval uniforms to standard normals using **Wichura AS241 inverse CDF**. A scrambled Sobol `u32` shall use the same one-word midpoint mapping $u=(x+0.5)2^{-32}$ as Philox, including the same normative cast/add/multiply order and boundary behavior.

The AS241 implementation shall bundle the original published double-precision coefficient tables and branch constants, each materialized as a fixed IEEE-754 `f64` value. Polynomial evaluation shall use a versioned Horner order with explicit multiply followed by add; fused multiply-add, algebraic reassociation, compiler fast-math, and delegation to an external inverse-normal implementation are prohibited in the reproducible kernel. The transform version, coefficient bits, branch boundaries, evaluation order, and extreme-tail behavior shall be covered by bit-pattern known-answer tests so the transform participates in the same-platform bitwise-reproducibility contract.

The MVP shall support the following variance-reduction methods:

- antithetic variates for compatible pseudo-random and randomized-QMC configurations; and
- Brownian-bridge path construction for mapping low Sobol dimensions to the most important Brownian time components.

For pseudo-Monte Carlo antithetics, the base path shall evaluate AS241 once and the mate shall use the exact floating-point negation $-Z$ of each resulting standard normal. The mate shall consume no Philox path, counter, dimension, lane, or additional AS241 evaluation. Base and mate shall be evaluated as one independent sampling unit and reduced through their arithmetic-average Price and risk contributions.

The pseudo-Monte Carlo count parameter shall always mean the number $N$ of independent sampling units. With antithetics disabled this evaluates $N$ trajectories; with antithetics enabled it evaluates exactly $2N$ trajectories while retaining effective independent sample size $N$. The base trajectory's Philox path word shall equal the zero-based sampling-unit index, and the mate shall have no separate random path identity. Results shall report the requested sampling-unit count, antithetic multiplicity, actual trajectory-evaluation count, and effective sample size separately. Overflow in the derived trajectory count shall be rejected during plan compilation.

For randomized QMC antithetics, each Sobol point shall remain one within-scramble sampling unit. Its midpoint-mapped uniform shall be evaluated by AS241 once, and its mate shall use exact floating-point negation $-Z$ for every normal coordinate. The mate shall consume no additional Sobol point or dimension and shall not be constructed by complementing the scrambled word or reevaluating AS241. `points_per_scramble = N` therefore means $N$ Sobol points and $2N$ trajectory evaluations per scramble when antithetics are enabled. Pair contributions shall be averaged before the deterministic within-scramble reduction; uncertainty shall still be estimated only across complete independent scrambles.

For an RQMC Brownian bridge on normalized times $0=t_0<t_1<\cdots<t_n=T$, $W(0)=0$ shall be fixed without consuming a dimension and the first bridge normal shall construct $W(T)=\sqrt{T}Z$. Thereafter, every unfilled time $t_k$ bracketed by already constructed times $t_l<t_k<t_r$ is a candidate with conditional variance

\[
v_k=\frac{(t_k-t_l)(t_r-t_k)}{t_r-t_l}.
\]

The next normal shall be assigned to the candidate with maximum conditional variance across all current intervals. Exact ties shall select the earlier time, then the lower normalized node index. Its conditional mean and variance shall use the analytical Brownian-bridge formulas. The complete node order, bracketing indices, coefficients, arithmetic order, and tie-break result shall be materialized in `SimulationPlan`; runtime path scheduling shall not affect them.

With $F$ independent Brownian factors, Sobol dimensions shall be ordered bridge-node-major and factor-minor: global dimension $d=bF+q$, where $b$ is the bridge construction rank and $q=0,\ldots,F-1$ is the canonical factor index. Thus the first $F$ dimensions construct the terminal Brownian value for every factor before any lower-priority bridge node. Auxiliary non-Brownian random drivers, if supported by a model extension, shall be assigned only after the complete Brownian block under a separately versioned layout.

The bridge shall first construct $F$ independent Brownian paths. For each simulation interval, their increments shall be divided by the interval square root and then multiplied by that interval's validated correlation factor to obtain correlated standard-normal shocks. A time-dependent correlation matrix shall therefore act on interval increments after bridging, not on bridge input normals. The reverse kernel shall apply the exact transpose operations in reverse order.

QMC Brownian-bridge construction is distinct from Brownian-bridge barrier-crossing correction. Enabling one shall not implicitly enable the other.

### 5.2 Time stepping and barrier monitoring

Black–Scholes and Black-76 path evolution shall use exact conditional transitions over each simulation interval for deterministic curves and continuous carrying costs.

The baseline Local Volatility scheme shall be Log-Euler in the continuous-martingale state \(f\):

\[
\log f_{i+1}=\log f_i-
\frac{1}{2}\sigma_{\mathrm{loc}}^2(t_i,f_i)\Delta t_i
+\sigma_{\mathrm{loc}}(t_i,f_i)\sqrt{\Delta t_i}\,Z_i.
\]

Deterministic carry and affine dividend jumps are represented by the \((A,B)\) transformation rather than a drift or jump in \(f\). The evaluation point of Local Volatility and treatment of reconstructed Spot observations shall be explicit parts of the scheme definition. A scheme identifier and all step controls shall be included in result metadata.

The simulation grid shall be constructed from the sorted union of all required contractual and market event times, including observation, exercise, barrier-monitoring, payment-relevant, and dividend ex-dates. Every Local Vol calculation shall explicitly supply a finite positive maximum step width \(\Delta t_{\max}\); there is no hidden default. An event interval of length \(h\) shall be divided into \(n=\lceil h/\Delta t_{\max}\rceil\) equal sub-steps, ensuring every actual step is no wider than the requested maximum. Required event times shall never be shifted merely to fit a uniform global grid.

Barrier products shall support two explicit monitoring modes:

- discrete monitoring at configured observation times; and
- continuous-monitoring correction between simulated endpoints using a Brownian-bridge crossing approximation.

The continuous correction shall identify the crossing-probability formula and interval-variance approximation used. Under Local Volatility it shall be documented as a discretization-dependent approximation, and it shall be possible to compare it with the uncorrected discrete result. QMC Brownian-bridge path construction and barrier-crossing correction shall remain independently configurable.

For a Spot barrier \(H_S(t)\), continuous-monitoring correction shall operate in log continuous-martingale coordinates against

\[
H_f(t)=\frac{H_S(t)-A(t)S_0}{B(t)}.
\]

Compilation shall reject any monitored interval on which the transformed barrier is non-positive or undefined. On a diffusion sub-step, the bridge's effective integrated variance shall be the trapezoidal endpoint approximation

\[
v_i=\frac{1}{2}\left[\sigma_{\mathrm{loc}}^2(t_i,f_i)+
\sigma_{\mathrm{loc}}^2(t_{i+1},f_{i+1})\right]\Delta t_i.
\]

The correction shall use the transformed barrier values at both interval endpoints and shall identify its within-step barrier interpolation convention. This approximation shall participate in the configured \(\Delta t_{\max}\) convergence tests.

Within each diffusion sub-step, \(\log H_f(t)\) shall be interpolated linearly between its endpoint values. Conditional on two safe endpoints, the engine shall compute the Brownian-bridge crossing probability and propagate the complementary survival probability as a path weight. It shall not sample an additional Bernoulli crossing variable in the MVP. Products and diagnostics shall therefore distinguish conditional survival weights from realized discrete or dividend-jump hit states.

For numerical stability, products of interval survival probabilities shall be accumulated with stable logarithmic operations and converted to a weight only at defined consumption points. The implementation shall use stable `log1p`/`expm1` branches near probabilities zero and one and shall record underflow or saturation diagnostics.

The AAD reverse rule shall differentiate the continuous-barrier survival weight through both log-state endpoints, both Local-variance endpoint evaluations, the trapezoidal interval variance, and both transformed-barrier endpoints. It shall not freeze the correction probability during the primary AAD calculation. The matching bump-and-revalue validation shall rebuild the same transformed barriers and interval variances.

Barrier contact is inclusive: equality of the monitored Spot and barrier is a hit. The MVP shall support one active barrier per barrier component, covering up, down, knock-in, and knock-out variants. Double barriers and window barriers with multiple simultaneous levels are outside the initial barrier component, although time-dependent single-barrier schedules remain representable.

The MVP shall support no rebate and a fixed-cash rebate paid at expiry. Immediate-at-hit rebates are outside the MVP because the conditional-survival estimator does not generate a first-passage time. For expiry-paid structures, knock-out continuation cash flows use the survival weight and the hit rebate uses its complement; knock-in activation uses the complementary hit weight. All branches shall use the same deterministic endpoint/dividend hits and conditional bridge survival factors, and shall be discounted from the contractual expiry payment date.

At a discrete-dividend ex-date, the engine shall evaluate the reconstructed pre- and post-dividend Spots. The contractual, unsmoothed rule treats an affine dividend jump that touches or passes through an active barrier as a deterministic barrier hit; it shall not be folded into a diffusion bridge probability. For the primary smoothed Price/AAD calculation, endpoint and dividend-jump hit indicators shall instead use the same compact C2 quintic smoothing policy defined in Section 7.3. This changes only the hit indicator: the dividend jump remains deterministic and consumes no random variate. The result diagnostics shall distinguish endpoint hits, bridge-inferred hits, and dividend-jump hits and identify whether exact or smoothed indicators were used.

### 5.3 Least-Squares Monte Carlo

LSM shall support American vanilla options initially and later reusable early-exercise/callability logic.

An American vanilla contract shall provide an explicit, strictly increasing schedule of permissible exercise dates. The final expiry date shall be present in that schedule. A calendar-aware helper shall generate a schedule containing every business day between supplied start and expiry dates using the configured Weekend-plus-Custom-holidays calendar and business-day convention; the helper output is ordinary explicit input and no lazy calendar lookup occurs during pricing.

If an exercise date coincides with a discrete-dividend ex-date, the dividend event shall be applied first and immediate exercise value shall be computed from the reconstructed post-dividend Spot. This is consistent with the MVP expiry ordering and is fixed for the standard American builder. The compiled plan shall record the collision and event order.

At every non-terminal exercise date, exercise shall occur only when immediate exercise value is strictly greater than the fitted continuation value. Equality selects continuation. The same strict comparison and numerical comparison policy shall be used while training the policy and while applying it to independent valuation paths. At final expiry the terminal intrinsic payoff is applied directly rather than compared with a continuation estimate.

The baseline continuation-value approximation shall use all monomials in the configured state variables whose total degree does not exceed the configured maximum degree, including cross-variable interaction terms. The constant monomial shall be included. Column enumeration shall be deterministic from the declared state-variable order and a versioned graded ordering, and shall be identical in Rust and Python APIs.

At each non-terminal exercise date, every non-constant state feature shall be standardized using the arithmetic mean and population standard deviation computed only from that date's \(n\) training in-the-money paths:

\[
\mu=\frac1n\sum_{p=1}^{n}x_p,\qquad
\sigma=\sqrt{\frac1n\sum_{p=1}^{n}(x_p-\mu)^2},\qquad
z_p=\frac{x_p-\mu}{\sigma}.
\]

Those fitted statistics shall be stored with the policy and applied unchanged to out-of-sample valuation paths. No valuation-path observation may influence feature scaling. A zero or numerically negligible population standard deviation shall cause every dependent non-constant basis column to enter the deterministic rank-exclusion path and emit a structured warning; the constant column remains unscaled.

The baseline regression solver shall be column-pivoted QR. Each calculation shall explicitly supply finite non-negative absolute and relative rank tolerances \(\tau_{\mathrm{abs}}\) and \(\tau_{\mathrm{rel}}\). With pivoted diagonal entries ordered by magnitude, column \(j\) shall be retained only when

\[
|R_{jj}|>\max\!\left(\tau_{\mathrm{abs}},
\tau_{\mathrm{rel}}|R_{11}|\right).
\]

The implementation shall report both tolerances, numerical rank, pivot ordering, retained and excluded basis columns, residual diagnostics, and ill-conditioning warnings at every exercise date. When the rank test classifies a pivoted column as dependent, that column and all subsequent below-tolerance columns shall be deterministically excluded, their coefficients represented as zero in the original basis ordering, and a structured warning emitted. Pivot ties shall use original basis-column order. The MVP shall not fall back to Ridge regression or another solver implicitly.

Regression shall use only paths that are in the money at the relevant exercise date. The in-the-money definition shall be the strictly positive immediate exercise value, subject to a documented numerical tolerance.

If a non-terminal exercise date has zero training in-the-money paths, regression shall be skipped for that date and the fitted policy shall contain an explicit `ContinueAll` decision. All valuation paths shall therefore continue at that date even if the valuation sample happens to contain in-the-money paths. A structured warning shall report the zero count and skipped regression; a later date's regression shall never be reused.

Training and valuation shall use independent path sets. With a common master seed they shall be separated by the fixed `LsmTrain = 1` and `Valuation = 0` Philox domains, respectively, and the calculation configuration shall provide separate sampling-unit counts. Any antithetic multiplicity shall follow the common pseudo-Monte Carlo count convention in both phases. Regression coefficients and the resulting exercise policy shall be fitted only on training paths and applied out of sample to valuation paths.

The implementation shall make the following configurable and report them in diagnostics:

- basis functions;
- regression method;
- exercise dates;
- in-the-money tolerance and retained path count at every exercise date;
- training and valuation sampling-unit counts, actual trajectory counts, and random-domain identifiers;
- regression rank or conditioning diagnostics.

For LSM sensitivities, the base valuation shall first determine regression coefficients and an exercise policy. The primary AAD calculation shall then hold the fitted policy fixed while differentiating the selected valuation-path evolution and cash flows. It shall not differentiate through QR or refit the continuation regression.

The result shall label these sensitivities as fixed-exercise-strategy Greeks and include the base strategy identifier or digest required to reproduce them. They shall not be presented as the total derivative of a fully reoptimized LSM estimator.

The independent validation method shall be common-random-number bump-and-revalue with the same trained exercise policy, no regression refit, and the same realized stopping index for each corresponding valuation path. The bump therefore revalues the cash flow selected by the base run rather than moving or re-evaluating the exercise boundary.

Both AAD and bump results shall be labelled as frozen-stopping-index sensitivities. The stored stopping indices shall be associated with deterministic path identities so that the base, AAD, and bumped calculations use exactly the same pathwise cash-flow selection.

## 6. Valuation outputs

Each pricing result shall include at least:

- present value;
- Monte Carlo standard error;
- confidence interval;
- seed and effective simulation configuration;
- warnings generated by market-data, model, or numerical validation.

For pseudo-Monte Carlo, standard error shall be based on the estimator's independent sampling units, accounting for antithetic pairing. For randomized QMC, standard error and confidence intervals shall be based on independent scrambled replicate estimates. The result shall identify which error estimator was used and its effective sample size.

Price and every requested scalar Greek shall carry its own standard error. Their requested covariance statistics shall use mergeable centered moments rather than raw `sum_of_squares - sum^2` formulas. For pseudo-MC, one unpaired path or one averaged antithetic pair is the independent sampling unit. For RQMC, one complete independently scrambled replicate estimate is the independent sampling unit; within-replicate Sobol points shall not be treated as independent observations.

For VegaKT, the standard result shall include a variance or standard error for every reporting bucket and its covariance with Price. A complete bucket-by-bucket covariance matrix has quadratic storage and shall be computed only when explicitly requested. The result shall state which covariance fields were requested and populated; absence of the full matrix shall not be presented as zero covariance.

Where applicable, the engine shall also expose exercise, barrier, cash-flow, or convergence diagnostics through structured optional result objects.

## 7. Sensitivity requirements

### 7.1 Required sensitivities

- Delta
- Gamma
- Vega
- VegaKT: a decomposition of Local Vega onto a configured implied-volatility strike/maturity reporting grid

Canonical Scalar Vega shall be defined as the sum of all in-domain canonical \(f\)-IV VegaKT buckets using the same decomposition normalization:

\[
\operatorname{Vega}=\sum_{i,j}\operatorname{VegaKT}_{i,j}.
\]

VegaKT shall use the decomposition convention of *Vega KT for the Local Volatility Model: An AD Approach* rather than a pseudo-inverse transformation of SSVI parameter sensitivities. The reporting grid is generated from the calibrated SSVI/eSSVI surface and is not the library's independent calibration input.

Because the reporting grid generally has more points than the SSVI parameter vector, the reported VegaKT shall not be described as a unique mathematical gradient with respect to independently bumpable SSVI-generated node values. It is a paper-defined market-Vega decomposition on the chosen \((K,T)\) basis. SSVI parameter Greeks are a separate possible output and are not required for the MVP.

The returned VegaKT result shall preserve the configured strike/maturity coordinates, quote units, and deterministic bucket ordering. With affine dividends, the decomposition shall be expressed only against marked \(f\)-IV nodes. For interpretation, each node may also carry the corresponding Spot-contract strike

\[
K_S(T)=A(T)S_0+B(T)K_f
\]

as reference metadata. This is a coordinate annotation only: it shall not trigger Spot-IV inversion, a Black-Vega-ratio transformation, or a second set of sensitivity values.

VegaKT maturity nodes shall be supplied explicitly. An automatic helper may propose standard or event-aware maturity nodes, but its output shall be materialized as the same explicit, serializable input before compilation. Strike coordinates shall use log-forward-moneyness. The implied volatilities sampled from SSVI/eSSVI at these nodes shall define a separate reporting basis reconstructed by bilinear interpolation in maturity and log-forward-moneyness. This reporting interpolation is used only by the finite-node VegaKT projection; it shall not replace SSVI/eSSVI for Dupire derivatives or pricing.

The finite-node interpolation basis shall form a partition of unity in its covered domain. Consequently, the sum of in-domain buckets represents the first-order parallel shift of all reporting-node implied volatilities. Any sensitivity excluded from the projection domain shall be returned separately as a residual and shall not be silently included in Scalar Vega.

### 7.2 Calculation methods

The library shall provide:

- reverse-mode automatic differentiation/AAD as the primary sensitivity mechanism;
- bump-and-revalue as an independent reference and validation mechanism.

The MVP pathwise AAD implementation shall combine dedicated matched primal/reverse Simulation kernels with a compiled Payoff opcode tape. It shall not record scalar arithmetic from every Simulation step on a general-purpose dynamic tape. The immutable Payoff tape shall be executed in topological order and reversed in the corresponding deterministic reverse order; each supported opcode shall provide a matched primal rule, reverse rule, and gradient tests. The Payoff reverse pass shall seed adjoints of observed path states and cash flows, after which the Simulation reverse kernel shall propagate them through the path evolution and model inputs.

Within an AAD execution block, path states and adjoints shall use a state-major structure-of-arrays layout. Each logical state slot owns a contiguous array indexed by the fixed local path index; primal and adjoint buffers shall be distinct. Every SoA allocation shall begin on a 64-byte boundary, and each state-slot stride shall be rounded up to a multiple of eight `f64` elements. Logical tile length shall remain distinct from padded physical capacity. Before each tile use, all non-live padded primal and adjoint lanes shall be initialized to positive zero. Every scalar and SIMD kernel shall be bounded or masked by logical length, and padded lanes shall consume no random coordinates, execute no product event, trigger no error/warning, and contribute to no Price, Greek, statistic, or diagnostic reduction. Block capacity, state-slot ordering, buffer-layout version, physical stride, and padding shall be properties of the compiled plan or kernel ABI and shall not depend on Rayon worker assignment. Thread-local buffers shall be allocated or resized outside the path/step hot loops and reused across blocks.

Path replay shall use fixed-interval checkpointing augmented by mandatory state-changing-event-batch boundaries. A versioned numerical policy shall supply the default checkpoint interval $K$, and each calculation may explicitly override it. The initial and terminal nodes and every normalized event time containing at least one state-changing event shall be checkpoint boundaries, and no replay segment may contain more than $K$ Simulation intervals. Dividends, path-state updates, barrier-state changes, averaging/extremum updates, coupon/call state changes, and exercise/stopping events are state-changing for this purpose. All events at one normalized time shall execute in the library's canonical event order; the full checkpoint shall contain only the state after the complete same-time batch. Each individual event shall retain its own minimum reverse cache as selected by liveness analysis. Reverse execution shall visit the batch's events in exact reverse canonical order, recovering successive pre-event adjoints without storing a full state after every event. The implementation shall not duplicate the complete pre-event state unless it is itself live. The resolved positive value, policy version, ordered checkpoint boundaries, canonical batch membership/order, and event identities shall be stored in the normalized plan and replay metadata. Runtime memory availability, thread count, or timing measurements shall not silently change $K$ after compilation. The exact initial default may be selected by documented benchmark, but changing it shall require a numerical-policy version change.

For each compiled risk request, the Payoff-tape compiler shall perform deterministic reverse-liveness analysis from the requested outputs and allocate cache slots only for primal values, branch/smoothing data, and state snapshots actually required by reachable reverse rules. Dead opcodes and primal values that no reachable reverse rule reads shall consume no reverse cache. The liveness algorithm version, live-opcode bitmap or equivalent fingerprint, typed cache-slot order, and cache size shall be part of the compiled plan. Cache allocation shall be static for a fixed plan and shall not change according to path outcomes.

Logical deterministic Reduction blocks and AAD execution tiles shall be distinct nested levels. Each fixed Reduction block shall be partitioned into contiguous, ascending-path AAD tiles of a fixed compiled capacity, with at most one shorter terminal tile. An AAD tile shall never cross a Reduction-block boundary. A versioned policy shall provide the default positive tile capacity and each calculation may override it with a positive capacity no larger than the Reduction-block size; invalid overrides shall be rejected rather than silently truncated. Compilation shall resolve the capacity before allocating workspaces. Tile scheduling and worker assignment shall not alter the Reduction-block boundaries, sampling-unit order, compensated accumulation sequence, or balanced merge tree. The policy version, default, override, resolved capacity, and layout version shall be replay metadata; changing tile capacity shall not change the defined reduction result.

Delta shall be calculated by AAD for supported differentiable paths and payoff graphs. Gamma shall use a central finite difference of AAD Delta:

\[
\Gamma(S_0)\approx
\frac{\Delta_{\mathrm{AAD}}(S_0+h)-\Delta_{\mathrm{AAD}}(S_0-h)}{2h}.
\]

The two Delta calculations shall use common random numbers and otherwise identical numerical configuration. The absolute/relative Spot bump convention, \(h\), surface-risk convention under the Spot bump, and observed convergence across alternative bump sizes shall be available in result metadata.

For Local Volatility pricing, VegaKT shall follow the algorithmic decomposition described in *Vega KT for the Local Volatility Model: An AD Approach* (Adrien et al., SSRN 4107770):

\[
\text{Monte Carlo payoff}
\xrightarrow{\mathrm{AD}}
\text{Local Vega}
\longrightarrow
\text{paper-defined market VegaKT decomposition}.
\]

This decomposition is a functional requirement; the implementation must not require one independent full repricing for every implied-volatility node. The Local Vega representation and the transformation from Local Vega to implied-volatility strike/maturity buckets shall be separate, testable components.

The exact discrete Local Vega definition, decomposition operator, normalization, and mapping to the reporting grid shall be documented alongside the implementation and checked against the reference paper. If an implementation uses an algebraically equivalent formulation, equivalence shall be demonstrated by numerical tests.

The MVP Local Vega estimator shall regularize the Dirac distribution in the paper's pathwise formula with a compact piecewise-linear hat kernel whose width is the applicable Local-Vega grid interval. The kernel projection shall be deterministic and shall expose its grid, boundary treatment, and normalization in calculation metadata.

By default, the Local-Vega projection grid shall reuse the explicit Local-variance grid. A calculation may instead provide a separate explicit Local-Vega grid without changing the path adjoint or VegaKT reporting interfaces. Automatic grid helpers may propose either grid, but compiled inputs shall contain the resulting explicit nodes.

The MVP shall recover the next-time Local Gamma from Local Vega with the paper's first-order approximation (equation (11) of SSRN 4107770), rather than solving the discrete convolution equation. The approximation and its time-step order shall be identified in result metadata and tested by time-step refinement.

Where the call density in the denominator of equation (11) is below a configured active-domain threshold, the affected region shall be excluded from the Local-Gamma projection rather than denominator-clamped. The threshold shall be an explicit, dimensionless ratio to the maximum call density at the applicable maturity and shall be required for each VegaKT calculation. The implementation shall report the ratio, resulting excluded domain, excluded probability mass where available, and signed residual sensitivity. This residual shall remain separately observable from the reported VegaKT buckets.

The call density used by equation (11) shall be calculated analytically from the calibrated SSVI/eSSVI representation in continuous-martingale coordinates. It shall not be estimated from the pricing paths. The VegaKT operator and quote buckets shall remain in forward-normalized \(f\) coordinates; corresponding Spot-contract strikes may be attached through the deterministic affine map as non-risk metadata.

The transition integral in the paper's discrete operator shall use cell-integrated transition-kernel weights on the possibly non-uniform grid. It shall not use transition-density values sampled only at grid nodes. Local Gamma shall be reconstructed piecewise-linearly between interior nodes. The two outer cells shall extend to zero and positive infinity in strike space and shall use the nearest edge value as a constant, avoiding an unbounded linear extrapolation while preserving all transition probability mass. The cell-boundary convention and analytic or numerical evaluation of the cell integrals shall be versioned numerical-policy metadata.

At each maturity, the active domain selected by the relative call-density threshold shall be the maximal connected grid interval that contains the forward. Qualifying cells disconnected from that interval shall remain excluded and shall contribute to the reported residual diagnostics. If the forward itself fails the configured density test, the calculation shall return a structured VegaKT error rather than construct an empty or arbitrary domain.

Vega projected beyond the configured strike-node range shall be accumulated into the nearest edge bucket and shall emit a structured warning containing the affected range and magnitude. This edge aggregation is distinct from the low-density active-domain exclusion above.

For validation, bump-and-revalue shall use perturbations consistent with the decomposition basis prescribed by the reference algorithm and reprice with common random numbers. It shall not silently reinterpret dependent SSVI-generated grid values as independent calibrated parameters. The bump size, central/forward scheme, perturbation basis, surface rebuild policy, and any regularization shall be recorded in the result metadata.

AAD and bump-and-revalue shall use compatible bump conventions and return structured metadata identifying the differentiation method and numerical settings.

Non-smooth payoffs and discontinuous events shall not silently return unreliable pathwise derivatives. The result must identify the estimator or smoothing/convention used.

### 7.3 Non-smooth payoff sensitivities

For digitals, barriers, conditional coupons, autocalls, maxima/minima, and other non-smooth graph nodes, the primary AAD estimator shall use explicitly configured smooth surrogate operations. Smoothing shall be implemented at graph-node level.

The MVP Digital option set shall include cash-or-nothing and asset-or-nothing Calls and Puts. Cash payouts and asset quantity, strike, expiry observation, payment date, and currency shall be explicit contract fields. Digital equality shall use the same Call/Put exercise-boundary convention recorded by the product.

An indicator on signed condition value \(x\) shall use a compact symmetric transition based on the quintic smoothstep

\[
q(u)=6u^5-15u^4+10u^3,\qquad 0\le u\le1.
\]

It shall be exactly zero below the configured transition band, exactly one above it, and map the band affinely to \(u\in[0,1]\). Value, first derivative, and second derivative shall join continuously at both band edges. Call and Put indicators shall use mirrored versions of the same kernel.

Smoothed Maximum and Minimum shall be constructed from a compact smooth positive-part function whose derivative is the same quintic indicator kernel. They shall equal exact `max`/`min` outside the configured band and preserve symmetry, translation equivariance, and `min(a,b)+max(a,b)=a+b` under smoothing. The primal and reverse implementations shall share one versioned formula.

When smoothing is enabled, the primary reported value shall be the Smoothed Price and the reported AAD Greeks shall be derivatives of the same smooth payoff. The result shall not combine an unsmoothed contractual price with Greeks of a different payoff under one valuation label.

Smoothing width shall be specified as a fixed positive half-width \(h\) in the payoff node's native value units. The complete transition band is \([-h,h]\) in signed-distance coordinates and is mapped by \(u=(x+h)/(2h)\) to \([0,1]\). Thus the configured number is not the full band width; the full width is \(2h\). The width may differ between graph nodes. Relative-width and automatically inferred bandwidth conventions are outside the MVP unless added later as separate policies.

Barrier endpoint and dividend-jump hit predicates shall use this same quintic kernel in their signed hit-distance when smoothing is enabled. For an up barrier, positive distance denotes the monitored Spot being at or above the barrier; for a down barrier, positive distance denotes it being at or below the barrier. A dividend-jump crossing score shall be formed from the pre- and post-jump signed distances using the library's C2 smoothed Maximum, so the complete smoothed predicate, including aggregation of the two endpoints, has an explicit reverse rule. At zero signed distance the smoothed hit weight is \(1/2\); the contractual exact convention remains touch-is-hit. The smoothing width and its Spot units shall be explicit per barrier component.

The API shall provide an optional Width ladder helper. It accepts a user-supplied ordered list of positive half-widths and performs a complete, separately labelled valuation for each width using the same seed, random-coordinate assignment, simulation plan, and market inputs. It shall report Price and requested Greeks at every width together with adjacent-width differences. It shall not adapt the width, select a preferred result, extrapolate to zero width, or combine ladder estimates implicitly. The primary result remains the valuation at the separately declared primary half-width.

Every sensitivity result involving smoothing shall report:

- affected graph nodes and discontinuity types;
- smoothing kernel;
- smoothing half-width, full transition width, and units;
- confirmation that the reported price and Greeks use the same smooth payoff;
- bump-and-revalue comparison settings.

The library shall not silently apply smoothing or label a smoothed derivative as an exact derivative of a discontinuous finite-sample payoff.

### 7.4 Spot-risk and display conventions

Every Local Volatility Delta or Gamma request shall select one explicit Smile-dynamics convention. The MVP shall support:

- Sticky log-forward-moneyness;
- Sticky absolute strike; and
- Sticky delta.

The selected convention determines how the implied-volatility surface and derived Local Volatility surface are reconstructed after a Spot bump. It shall be applied consistently to both bumped AAD Delta calculations used for Gamma and recorded in the result. No global implicit Smile convention is permitted.

Every Greek result shall expose both its raw mathematical derivative and a market-scaled value. For a Spot level \(S_0\), the baseline scaling conventions are:

\[
\Delta_{1\%}=\Delta\,(0.01S_0),
\qquad
\Gamma_{1\%^2}=\Gamma\,(0.01S_0)^2,
\]

with the second-order P&L contribution for a one-percent Spot move reported separately as \(\tfrac12\Gamma_{1\%^2}\). For volatility risk,

\[
\operatorname{Vega}_{1\mathrm{vp}}=0.01\operatorname{Vega}_{\mathrm{raw}},
\qquad
\operatorname{VegaKT}_{i,j,1\mathrm{vp}}
=0.01\operatorname{VegaKT}_{i,j,\mathrm{raw}}.
\]

Raw units, scaling factors, reference Spot, volatility-point convention, and currency shall be present in structured result metadata. Multi-asset Spot risks shall be keyed by underlying identity and scaled using the corresponding reference Spot.

## 8. Rust API requirements

The Rust API shall separate the following concepts:

- market data;
- product definition;
- stochastic model/process;
- random sequence generator;
- path generator;
- pricing engine;
- sensitivity engine;
- numerical configuration;
- result and diagnostics.

Core interfaces shall allow new products, models, random sequences, variance-reduction techniques, and pricing engines to be added without modifying unrelated components.

Normal validation failures shall be represented by typed errors rather than panics. Public types shall document units, time conventions, ordering, ownership, and thread-safety expectations.

### 8.1 Request and result serialization

The MVP shall serialize public `PricingRequest` and `PricingResult` documents as versioned UTF-8 JSON. Each top-level document shall carry an explicit document-kind discriminator and integer schema version. Compiled plans, Payoff tapes, native indices, pointers, and runtime workspaces shall not be serialized. JSON encoding is a human-facing interchange format and shall remain distinct from the canonical binary stream used for BLAKE3 fingerprints.

`schema_version` shall be a positive, monotonically increasing `u32`; zero is reserved and invalid. Version numbers shall never be reused, reordered, or interpreted as floating-point values. The public wire format uses one global current Schema version across document kinds, while the Library crate's semantic major version defines the guaranteed backward-readable compatibility boundary. Writers emit only the current Schema version.

Deserialization shall reject unknown fields recursively in every versioned object, as well as duplicate object-member names, comments, trailing non-whitespace data, invalid UTF-8, and a document-kind mismatch. An error shall identify the schema version and the applicable JSON path. A field belonging to a future schema shall not be silently ignored. Serialization shall emit fields in documented schema order and map entries in their domain-defined deterministic order so fixtures and diffs are stable, although the fingerprint shall be calculated from normalized typed data rather than raw JSON bytes.

Every finite `f64` shall be emitted as a JSON number using a versioned shortest-round-trip binary64-to-decimal algorithm: reparsing the token as `f64` must recover the identical IEEE-754 bits. Negative zero shall be emitted explicitly as `-0.0` and shall remain distinct from positive zero. NaN and positive/negative Infinity are not JSON values in this schema and shall be errors, never quoted sentinel strings or `null`. The serializer's exponent spelling and other lexical choices shall be covered by frozen byte-for-byte fixtures; insignificant differences in external JSON spelling shall not affect the normalized-data fingerprint.

Every enum shall use an internally tagged JSON object with the reserved field `"type"` and a stable documented string variant tag. Payload fields shall be siblings of `"type"`; even unit variants shall use an object rather than a bare string. Writers shall emit `"type"` first, and readers shall reject a missing, duplicate, non-string, or unknown tag. Variant and field names shall be part of the schema version and shall not be inferred from Rust debug names or in-memory discriminants.

All schema-defined JSON field names, enum variant tags, and `document_kind` values shall be lowercase ASCII `snake_case`, beginning with an ASCII letter and continuing only with ASCII letters, digits, or underscores. Rust/Python attribute aliases, class names, capitalization, Unicode lookalikes, and hyphenated spellings shall not be accepted as alternate wire names. Renaming any public wire name requires a schema-version change and, where supported, an explicit migration.

An absent optional value shall be represented only by omitting its field. Writers shall omit `None`; readers shall reject JSON `null` for optional and required fields, and shall not treat `null` as omission, zero, an empty collection, or a default. A present optional field shall be validated exactly as its underlying type. If a future product requires a semantic null state, it shall use an explicit tagged variant rather than overloading JSON `null`.

Every integer-typed field, including IDs, counts, shape dimensions, seeds, enum-independent version numbers, and indices, shall accept only a JSON integer token with no decimal point or exponent. Parsing shall operate directly on decimal digits rather than through `f64`, and the value must fit the field's declared signed or unsigned Rust width exactly. Unsigned fields shall reject negative tokens, including `-0`; quoted decimal strings and numerically integral forms such as `1.0` or `1e0` shall be errors. Writers shall emit ordinary base-10 integer tokens with no leading plus sign, exponent, decimal point, or unnecessary leading zero.

A map whose schema key type is a JSON string shall be encoded as a JSON object. A map with any non-string key type shall instead be encoded as an array of entry objects, each containing exactly `"key"` and `"value"`; a key shall never be coerced to a string solely to fit JSON object syntax. Readers shall reject duplicate logical keys in either representation. Each map schema shall define a deterministic entry order for serialization; object-member order remains semantically insignificant on input.

Map serialization order shall be canonical and independent of insertion order or hash-table iteration. String keys shall be ordered by unsigned lexicographic comparison of their exact UTF-8 bytes. Non-string keys shall be ordered by unsigned lexicographic comparison of their domain key's canonical fingerprint byte encoding, before any surrounding map-entry framing. Canonical key encodings must be injective within the declared key type; equal comparison bytes for distinct normalized keys shall be an error rather than resolved by an unstable tie-breaker.

The standard serializer shall emit compact deterministic JSON with no insignificant spaces or line breaks inside the document. A separate pretty-print helper shall be available for human inspection, while preserving the same schema field order, map-entry order, numeric spelling, and parsed value as compact output. Whitespace choices shall not affect typed-data fingerprints, and the pretty form shall not constitute a separate wire schema.

JSON strings shall emit valid Unicode scalar values directly as UTF-8. Writers shall escape only quotation mark, reverse solidus, and control characters U+0000 through U+001F; solidus and non-ASCII characters shall not be escaped merely for ASCII compatibility. The escape spelling shall be deterministic: the standard short escapes shall be used for backspace, tab, LF, form feed, and carriage return, with the remaining control characters written as lowercase four-hex-digit `\u00xx` escapes.

The pretty-print helper shall use exactly two ASCII spaces per nesting level and shall never emit tab characters. Its structural line endings and final newline shall follow the same file-output rules as compact serialization. Indentation width is not caller-configurable in the versioned standard helper.

Readers shall accept non-canonical but otherwise valid JSON lexical forms, including insignificant whitespace, arbitrary object-member order, and equivalent valid string escapes, provided that all schema-specific rules remain satisfied. Accepted input shall be decoded into typed domain data and then validated, migrated where applicable, and fingerprinted from its normalized typed representation. Input bytes are therefore not preserved as an identity, and reserialization shall produce the library's current canonical output rather than reproduce the caller's spelling.

JSON string decoding shall preserve the exact sequence of Unicode scalar values and shall not apply NFC, NFD, case folding, locale transformation, trimming, or other Unicode normalization. Literal UTF-8 and valid JSON escape sequences that decode to the same scalar sequence shall yield the same typed string and fingerprint. Invalid UTF-8, malformed escapes, and unpaired UTF-16 surrogate escapes shall be rejected.

The JSON parser shall enforce versioned default resource limits with a per-read explicit override bounded by immutable absolute hard caps. Limits shall independently cover total input bytes, nesting depth, string bytes, number-token length, array elements, object members, and total parsed values before domain construction. An override may tighten a limit or raise it only up to the corresponding hard cap; it shall be recorded in replay metadata when a parsed request is used for valuation. Exceeding any limit shall stop parsing with a typed error identifying the limit and observed quantity where safely available.

Every supported schema version and top-level document kind shall ship with a corresponding JSON Schema document declaring JSON Schema Draft 2020-12. The bundled schemas shall specify required fields, types, bounds expressible in JSON Schema, tagged-union structure, array shapes where statically expressible, and strict object membership. They are a public interoperability contract and shall be frozen and regression-tested with the release; constraints requiring financial or cross-field semantics shall remain in the documented typed-domain validator rather than be approximated incorrectly in JSON Schema.

Version-specific wire DTO definitions shall be the source for JSON Schema generation; the current in-memory pricing types shall not implicitly redefine an older wire contract. Schema generation shall be deterministic, and the generated documents shall be committed as Golden artifacts. Any byte-level Schema diff shall require explicit review together with compatibility fixtures and, when the public wire contract changes, an appropriate schema-version change and migration. CI shall fail on an uncommitted or unreviewed generated diff.

Every JSON validation issue shall identify its instance location with an RFC 6901 JSON Pointer. The root pointer is the empty string, object tokens escape `~` as `~0` and `/` as `~1`, and array positions use zero-based decimal indices. Errors shall also identify the schema version, document kind, validation phase, and stable machine-readable code, so a pointer created before migration cannot be confused with a path in the migrated document.

Schema and typed-domain validation shall collect multiple independent issues in deterministic order up to a configured maximum. Syntax failures and parser resource-limit failures may terminate immediately when safe continuation is impossible. The collected-error limit shall have a versioned default, allow an explicit override only within an absolute hard cap, and return a truncation indicator when further issues may exist. Given identical input, schema version, limits, and library version, the ordered issue list and truncation state shall be identical.

Validation shall execute in this order: JSON syntax and parser resource limits; strict validation against the declared-version Schema; each registered deterministic migration step; strict validation against the Schema produced by that step, ending with the current Schema; and current typed-domain semantic validation. A failed phase shall prevent later phases that depend on a valid result. Documents already at the current version skip migration but still undergo both current-Schema and domain validation.

The versioned default for `max_validation_errors` shall be 64. A caller may request a smaller positive limit or a larger limit not exceeding the absolute validation-error hard cap. The returned collection shall never contain more than the effective limit; fatal parser failures may return one issue independently of the recoverable collection budget.

JSON files shall use UTF-8 without a byte-order mark, LF (`0x0A`) line endings, and exactly one LF after the top-level JSON value. Writers shall always produce that final newline; readers shall reject a UTF-8 BOM but may accept otherwise valid JSON whitespace after the top-level value subject to the existing trailing-data rule. Platform-native newline conversion is forbidden in Rust and Python file helpers.

Dense vectors, matrices, surfaces, correlation factors, and VegaKT arrays shall use an object containing `"shape"` and flat `"data"`. Shape dimensions shall be non-negative integers, and data shall be in C/row-major order with the last index varying fastest. The checked product of all dimensions shall equal the data length exactly and fit the applicable platform and resource limits; ragged, nested, column-major, or implicit-shape encodings shall be rejected. Axis meanings and coordinate arrays shall remain explicit sibling fields in the containing domain object rather than being inferred from shape.

The library shall maintain an explicit registry of supported deterministic migrations between older schema versions. Reading an older document shall first validate it strictly against its declared old schema, then apply the registered migration chain one version at a time, validating after every step. Each migration shall be pure, deterministic, tested with frozen fixtures, and prohibited from consulting wall-clock time, environment variables, machine state, or external services. The normalized plan and result metadata shall record the original version, current version, ordered migration identifiers, and pre/post-migration fingerprints. A newer-than-supported version or a missing migration link shall be a typed error; best-effort interpretation is forbidden.

Within a library major release line, every previously released Schema version for a supported document kind shall remain readable through the deterministic migration registry. The release shall publish a compatibility matrix identifying the library major, current Schema version, and all accepted source Schema versions. A new library major may retire old migrations only as an explicitly documented breaking change; patch or minor library releases shall not reduce the readable set within their major line.

Migration support shall be forward-only from an accepted older Schema version to the current Schema version. The core library shall not expose downgrade serialization or reverse migrations, and shall not promise direct conversion between arbitrary historical versions. All successful public serialization after loading or migration shall use the current Schema.

A migration must preserve the complete financial and execution semantics required to reconstruct the current typed document. If required information is absent, ambiguous, or cannot be represented without choosing a new economic meaning, migration shall fail with a typed loss-of-semantics error. It shall not invent a default, silently discard information, or continue with only a warning. Pure representational canonicalization is permitted when its semantic equivalence is specified and covered by fixtures.

Python shall expose all recoverable validation issues through one `ValidationError` whose `issues` attribute is an ordered list matching Rust's deterministic issue order. Each item shall expose the stable code, RFC 6901 instance path, validation phase, Schema version, document kind, and message, plus migration source/target versions where applicable. No partially constructed request, result, or migrated object shall be returned with the exception.

Each Python `ValidationIssue` shall be immutable after construction and expose read-only typed attributes plus `to_dict()`. The conversion shall return a new plain-Python dictionary using the public ASCII `snake_case` field names, preserve deterministic field order, and omit absent optional fields rather than emit `None`. Mutating the returned dictionary shall not mutate the issue or exception.

Validation codes shall be stable lowercase ASCII `snake_case` identifiers and are the programmatic compatibility contract within a library major. Human-readable messages shall be concise English diagnostics but are not stable comparison keys. Bundled Schema `$id` values shall use a stable, network-independent URN namespace containing the Schema version and document kind; changing an existing `$id` requires the same review as any other public Schema diff.

## 9. Python API requirements

Python shall be the primary interactive research interface. Bindings should be built using PyO3 and packaged with maturin unless implementation constraints justify another choice.

The Python API shall:

- expose product, market, model, pricing configuration, and result objects;
- accept NumPy-compatible vectors, matrices, and surface grids;
- return structured results rather than positional tuples;
- release the Python GIL during long-running Rust calculations;
- translate Rust errors into meaningful Python exceptions;
- preserve the same pricing semantics and seeded results as the Rust API;
- provide type hints and concise API documentation.

Scalar Python date inputs shall accept `datetime.date` and strict ISO-8601 calendar-date strings (`YYYY-MM-DD`). Datetime values with time-of-day or timezone semantics and NumPy `datetime64` inputs are outside the MVP date API. Both accepted representations shall normalize to the same Rust date value before validation and fingerprinting.

Direct dependencies on pandas or Polars are not required in the core API; users can convert their data to NumPy arrays or Python-native structures.

### 9.1 Packaging and target platforms

The MVP shall be distributed privately rather than published to PyPI or crates.io. Deliverables shall include:

- private Python wheel artifacts containing the Rust extension; and
- a private consumable Rust crate artifact or source package with locked dependency metadata.

The required target platforms are:

- Apple Silicon macOS (`aarch64-apple-darwin`); and
- 64-bit Windows using the MSVC toolchain (`x86_64-pc-windows-msvc`).

Build and test automation shall produce platform-specific wheels, run the Rust and Python consistency suites on both targets, and record the Rust compiler, Python ABI, target triple, enabled features, and dependency lock state. Linux and Intel macOS are outside the MVP support commitment.

## 10. Numerical and reproducibility requirements

- Production calculations shall use `f64` unless an algorithm explicitly documents otherwise.
- A result must be bitwise reproducible on the same supported platform when the seed, market data, product, model, algorithm, numerical feature set, build profile, thread configuration, and library version are unchanged.
- Floating-point reduction order and parallel scheduling policy must be controlled or recorded where they affect reproducibility.
- All dates, times, rates, volatilities, correlations, and cash amounts shall have documented units and conventions.
- Invalid inputs, NaNs, non-positive variances, non-positive-semidefinite correlations, and unstable Local Volatility values shall be detected and reported.
- Algorithms shall define tolerances explicitly rather than depend on undocumented global constants.

CPU parallel work shall run on a calculation-owned Rayon thread pool. Price, requested scalar Greeks, and VegaKT bucket sums shall use Neumaier compensation within fixed logical path blocks. Block partials shall be combined by a fixed balanced binary tree keyed only by logical block ID, with a versioned rule for merging each partial's sum and compensation. Block boundaries and the tree shall not depend on Rayon worker assignment or completion order. Mergeable centered moments and covariances shall use the Chan–Golub–LeVeque combination formulas with the same fixed tree rather than raw sums of squares. RQMC replicate variance shall use a deterministic two-pass calculation across replicate estimates.

Bitwise reproducibility across operating systems or CPU architectures is not required in v1.0. Cross-platform comparisons shall use documented numerical tolerances. Philox integer outputs and logical random-coordinate mapping shall nevertheless be platform-independent. Reproducibility across different thread counts remains a design goal rather than an MVP guarantee until deterministic-reduction conformance tests establish it.

## 11. Performance requirements

There is no hard response-time target for the initial release; accuracy is the priority. Nevertheless, the design shall:

- avoid per-path heap allocation in hot loops;
- support multi-core CPU parallelism;
- allow vectorized/batched path operations where useful;
- avoid unnecessary copies across the Python/Rust boundary;
- expose sufficient timing and workload metadata for benchmarking.

The supported MVP build shall use a pure-Rust linear-algebra backend and shall not link external BLAS/LAPACK. Linear algebra invoked inside Rayon workers shall not start an independent internal thread pool. A future optional BLAS backend may be added behind an internal abstraction but is outside the MVP reproducibility guarantee.

Performance optimizations must not change estimator semantics or silently weaken reproducibility.

## 12. Validation and testing requirements

The library shall include:

- unit tests for curves, interpolation, payoffs, random sequences, and path evolution;
- Black–Scholes/Black-76 analytical benchmarks for European vanilla options;
- Monte Carlo convergence tests with confidence-interval-based acceptance;
- randomized-QMC tests across independent scrambles, including error-estimator coverage checks;
- counter-based RNG tests proving deterministic path addressing under parallel scheduling;
- antithetic-pairing and Brownian-bridge dimension-ordering tests;
- AAD-versus-bump comparisons for smooth products;
- Gamma tests comparing central-bumped AAD Delta with analytical Black–Scholes Gamma and bump-size convergence;
- Spot-risk tests for every supported Smile-dynamics convention;
- raw-versus-market-scaled Greek identity tests, including Scalar Vega versus the VegaKT sum;
- VegaKT bump-and-revalue checks for individual and aggregate decomposition-basis perturbations;
- Local Volatility tests using constant-volatility and known-surface limits;
- Log-Euler weak-convergence and time-step refinement tests;
- Black–Scholes exact-transition tests under deterministic term structures;
- barrier tests against analytical Black–Scholes benchmarks where available, separately for discrete and continuous-correction modes;
- correlation and multi-asset statistical tests;
- piecewise-constant correlation change-time and factorization tests;
- Event/Payoff graph validation, compilation, deterministic-ordering, and built-in-product equivalence tests;
- smoothing-width convergence tests and AAD-versus-bump comparisons for discontinuous payoffs;
- LSM regression and exercise-policy tests;
- LSM in-sample versus independent out-of-sample bias tests;
- fixed-policy AAD versus fixed-policy common-random-number bump tests;
- polynomial-feature construction and pivoted-QR rank-diagnostic tests;
- fixed-exercise-strategy AAD tests against matching frozen-strategy bump-and-revalue;
- path-identity and frozen-stopping-index consistency tests across base, AAD, and bumped LSM runs;
- deterministic seeded regression tests;
- invalid-input and numerical-failure tests;
- non-positive post-dividend Spot failure tests with deterministic path diagnostics;
- single-currency graph and market-context validation tests;
- Python/Rust API consistency tests.

Statistical tests shall use tolerances derived from sampling error and avoid fragile assertions against one exact Monte Carlo value.

## 13. Initial vertical slice

The first end-to-end implementation shall be a European vanilla option.

It shall cover:

1. construction of market inputs;
2. Black–Scholes path simulation;
3. pseudo-random and quasi-random pricing;
4. price, standard error, and confidence interval;
5. Delta, Gamma, and Vega using AAD where supported;
6. bump-and-revalue comparison;
7. Rust and Python entry points;
8. comparison with the analytical Black–Scholes price and Greeks;
9. deterministic replay for a fixed complete configuration.

The subsequent vertical slice shall add calibrated SSVI/eSSVI input, Dupire Local Volatility construction, and VegaKT decomposition before expanding to all path-dependent and multi-asset products.

### 13.1 VegaKT acceptance criteria

The Local Volatility VegaKT slice shall be accepted when:

1. Local Vega is calculated by Monte Carlo and algorithmic differentiation;
2. the Local Vega is transformed into the paper-defined VegaKT decomposition on the configured SSVI/eSSVI reporting grid;
3. output coordinates, quote units, and bucket ordering exactly match the reporting-grid definition;
4. selected basis perturbations and aggregate VegaKT measures agree with common-random-number bump-and-revalue within explicitly defined Monte Carlo and finite-difference tolerances;
5. the implementation reports the estimator, SSVI/eSSVI convention, reporting basis, Dupire method, Local-variance interpolation, clamp events, and bump conventions needed to reproduce the result;
6. runtime and peak-memory benchmarks compare the AD/decomposition algorithm with its bump-and-revalue validation benchmark.

Acceptance additionally requires conservation checks for the hat-kernel projection, cell-integrated transition probability, and reporting-basis partition of unity; explicit reconciliation of bucket sum plus residual sensitivity; and time-step convergence evidence for the equation (11) Local-Gamma approximation.

## 14. Proposed delivery stages

The European Black–Scholes stage is committed through `european-bs-roadmap-v0.1.md`. Later stages remain sequencing recommendations until their own implementation roadmaps are accepted.

1. **Foundation:** numerical types, market objects, product/model/engine interfaces, result diagnostics, Python packaging.
2. **European Black–Scholes slice:** MC/QMC, analytical benchmark, AAD and bump Greeks.
3. **Local Volatility slice:** calibrated SSVI/eSSVI input, analytic/AD Dupire derivatives, Local variance grid with bilinear interpolation, Local Vega by MC/AD, and paper-defined VegaKT decomposition.
4. **Path dependence:** digital, barrier, Asian, and lookback products with monitoring corrections.
5. **Early exercise:** American vanilla and reusable LSM framework.
6. **Multi-asset:** baskets, worst-of products, correlation handling.
7. **Autocallables:** event-driven observations, coupons, calls, barriers, and multi-asset underlyings.

## 15. Architecture and numerical-specification decisions

The following choices do not change the agreed product scope or externally observable requirements, but must be fixed and documented during architecture and detailed numerical design:

- settlement-lag representation used by schedule generation;
- precise Standard SSVI and eSSVI formulas and admissibility constraints;
- SSVI/eSSVI admissibility tolerances, eSSVI terminal-slope configuration, and quantitative Local-grid helper parameters;
- numerical defaults for VegaKT maturity-grid helper generation and the active-domain density threshold;
- exact Dupire discretization and stabilization policy;
- Local-variance floor/cap values and warning thresholds;
- Local Volatility time-step convergence protocol and recommended helper values;
- Local Volatility Brownian-bridge interval-variance approximation;
- extension policy for double/window barriers and hit-time-paid rebates;
- exact numerical defaults for checkpoint interval and AAD tile capacity;
- precise non-uniform-grid hat-kernel normalization and analytic formulas or controlled numerical method for transition-cell integrals;
- precise smoothing-width interpretation and fixed widths by discontinuity type;
- supported Rust toolchain and Python versions/ABI;
- serialization format for calculation configurations and results;
- private artifact repository and internal licensing terms.

## 16. MVP acceptance summary

The MVP is acceptable when a Python user can define a European equity option, supply curves and calibrated SSVI/eSSVI parameters, select Black–Scholes or Local Volatility simulation, obtain a price and uncertainty estimate, calculate standard Greeks and the paper-defined VegaKT decomposition, reproduce a run from its complete configuration and seed, and independently validate key sensitivities using bump-and-revalue.

## 17. Reference algorithms and future extension

The MVP Local Volatility VegaKT reference is:

- Joachim Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*, SSRN 4107770 (2022), https://doi.org/10.2139/ssrn.4107770.

The following paper is a future Local Stochastic Volatility extension and is not part of the MVP model scope:

- Mohamed Hamdouche and Pierre Henry-Labordère, *Vega KT for LSV Models: An AD Approach*, SSRN 4304114 (2022), https://doi.org/10.2139/ssrn.4304114.

The affine discrete-dividend model, continuous-local-martingale transformation, dividend-date call-price matching condition, and dividend-aware Dupire construction follow:

- Julien Guyon and Pierre Henry-Labordère, *The Smile Calibration Problem Solved*, SSRN 1885032 (2011), https://doi.org/10.2139/ssrn.1885032.

The architecture shall therefore avoid coupling VegaKT exclusively to a Dupire Local Volatility process. A future implementation must be able to substitute an LSV-specific Local/Leverage Vega producer and calibration sensitivity mapping while reusing the market VegaKT representation, diagnostics, and validation framework.
