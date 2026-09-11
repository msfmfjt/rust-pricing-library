# Early Exercise reference fixtures

`reference-cases-v0.1.json` freezes the independent mathematical reference for
the `early_exercise_v1` decision, polynomial-basis, feature-scaling, and
column-pivoted Householder QR policies.

All numeric inputs and expected values are decimal strings. The checker uses
Python `Decimal` at 80-digit precision and does not import the production Rust
or Python package. Production binary64 tests separately freeze exact operation
order and bit patterns.
