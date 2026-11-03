# ADR 0033: Modular wheel gates and installed API signatures

Date: 2026-10-21. Status: implemented.

## Context and decision

The release wheel gate had grown to 5,564 lines. Each API extension could require
edits to separate export, member, decorator, signature and result maps. Runtime
validation checked exported names and descriptors, but could accept a native
callable whose parameter names, keyword rules or defaults differed from its stub.

Keep `scripts/smoke_test_wheel.py` as the stable CLI, example list and isolated test
runner. Move archive/metadata/RECORD/SBOM checks, installation, AST helpers, static
contracts and installed-runtime checks into `scripts/wheel_validation`. Consolidate
the existing independent expectations by domain, with each class's member
constraints together. Register domains explicitly and reject duplicate class
contracts and unknown fields. Expectations must remain independently reviewed
data, never generated from the shipped stub or installed implementation.

After static verification, derive a caller-facing manifest from the stub and compare
every declared function, constructor, static method, instance method and special
method with `inspect.signature` in the installed wheel's interpreter. Constructors
use the class signature. Normalize only the implicit `self` receiver; preserve all
other names, order, positional-only and keyword-only boundaries, required/default
distinctions and default value types. Reject missing or uninspectable callables,
variadic signatures and opaque/nonfinite defaults. Do not execute pricing methods,
constructors, factories or property getters to inspect signatures.

All CLI phases retain the same package gate. Model workflow path filters include
the new package so changes to shared validation keep triggering model CI. The
source-archive gate requires the package, tests and this decision and checks its
validation responsibilities at their new paths.

## Compatibility and numerical impact

- The CLI flags, test partition, automatic discovery, inventory checks, module
  isolation, examples and failure propagation are unchanged.
- Existing frozen contracts, archive checks and runtime descriptor checks remain.
  This adds full callable signature comparison; it does not claim full annotation
  freezing for every legacy method. PyO3 has no runtime type annotations to compare.
- `ValidationIssue.__eq__` and `__ne__` stubs now declare `value` as positional-only,
  matching CPython/PyO3's existing descriptors. Keyword calls were already rejected.
- `Engine.randomized_quasi_monte_carlo` advertises `scramble_count=16` using an
  explicit text signature. Its native default still uses
  `RqmcConfig::DEFAULT_SCRAMBLE_COUNT`. A request serialization test checks that
  omitted arguments equal explicit defaults, including variance reduction.
- No pricing formula, model validation, seed, reduction order, schema, canonical
  writer, fingerprint domain, dependency or compiled-plan cache changes.

## Validation and consequences

Migration checks compare all 173 class contracts and module expectations with the
old independent gate data, and compare ASTs of 44 moved functions. Unit mutation
tests exercise names, argument order, positional/keyword boundaries, required
arguments, bool/int/float default distinctions, opaque signatures, binding kinds,
missing/extra members and special methods. An installed-wheel suite verifies all
339 current callables and checks native RQMC defaults and comparison calls.

Signature reflection validates advertised call shapes, not all argument values,
native dispatch or financial behavior. Independent static annotation contracts,
runtime behavior tests and existing numerical/replay acceptance gates remain
necessary. Future changes to constants used by explicit text signatures must
update and pass both reflection and behavior tests.
