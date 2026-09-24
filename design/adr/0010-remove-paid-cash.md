# ADR 0010: Remove paid-cash support

Date: 2026-09-21. Status: accepted by user direction.
Supersedes the IV-conversion decision in
[ADR 0009](0009-escrowed-simulation-and-iv-conversion.md).

## Decision

Remove the remaining paid-cash quote convention and its bidirectional IV
converter. All equity simulations and calibration inputs continue to use the
[escrowed dividend convention](../../docs/models/hull-white-cash-dividends.md).
Bos–Vandermark remains deferred; this change adds no replacement model or
approximation.

## Impact

- **API:** remove Rust `DividendIvCoordinate`, `DividendIvConversion`, both
  converted-result types, the conversion error variant, and
  `AffineDividendTransform::paid_cash_coordinate`. Remove the Python
  `DividendIvConversion` class and `DividendIvCoordinate` type alias.
- **Inputs and serialized data:** calibration IVs must already use the escrowed
  coordinate. There is no retained paid-cash compatibility converter. Request
  schemas, payout schedules, and the escrowed Python selector remain unchanged.
- **Numerics and risk:** the escrowed simulation, calibration, AAD and VegaKT
  formulas are unchanged. The deterministic dividend helper always starts from
  the funded initial coordinate. Removed quote-conversion Jacobians are no
  longer part of the API; quote preparation is the caller's responsibility.
- **Reproducibility:** no simulation fingerprint, random-coordinate allocation,
  event ordering, or price/risk convention changes from the escrowed baseline.
- **Tests and packaging:** remove tests dedicated to the deleted converter and
  update Python facade and source-archive expectations. Existing escrowed
  dividend, pricing and risk regression tests remain the validation gates.
- **Documentation:** remove active paid-cash model and conversion references.
  Retain earlier ADRs and captured validation data as historical records under
  their original conventions, with supersession made explicit.

This decision is independent of the deferred broader model-boundary refactor
and performance work.
