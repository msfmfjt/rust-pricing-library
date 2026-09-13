# Integrate the completed baseline and stochastic-volatility PR stack

Date: 2026-09-13. Status: integration decision.

The user requested merging all open PRs. Main commit
`792a410dffe580aa5287172c977473cac20abf1b` contains the completed Local Volatility,
Path Dependence and Early Exercise baseline as a single commit, while PRs
#32–45 and #48–52 retain the original development history. Its tree is exactly
`0a863b7315dc03004436231be9dcca912eb16dd1`, also the tree of the completed
`e0-early-exercise-roadmap` branch. Use that retained history for three-way
integration and record both the published main and original history as parents.

Keep the completed baseline's continuous Barrier/LSM engines, schema-v3
migrations, acceptance/replay fixtures and gates. Add the LSV/Hull–White/rough
adapters, exports, documentation and examples. Resolve the observation adapter
against the baseline's optional observation dates and its per-node, bumped-Spot
affine coordinates; preserve both path implementations.

The experimental adapters still implement fixed contractual observations.
They explicitly reject American exercise, continuous Barrier monitoring and
smoothing-width ladders until those model-specific policies are implemented.
The general BS/Local Volatility facade continues to accept those requests.
Model capability must not silently expand merely because a shared request
builder now accepts a product or a risk option.

Keep the separate LSV calibration RNG namespace. Python diagnostic formatting
can identify it; schema v3 has no representation for it. Serialization of an
externally mutated LSM result containing that domain returns an error, rather
than relabelling it or changing the accepted schema.

Retain every existing CI gate and run all seven Python examples together.
Focused tests cover the adapter exclusions and the schema-domain boundary;
the full baseline and hybrid test suites verify the shared observation changes.
The experimental calibration and risk acceptance limits recorded in ADRs
0001–0005 remain in force after the repository merge.
