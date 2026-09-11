# Path Dependence Reference Fixtures

`reference-cases-v0.1.json` freezes independent high-precision values for the
`path_dependence_v1` smoothing and barrier numerical policy.

Validate it from the repository root with:

```shell
python3 scripts/check_path_dependence_reference_fixture.py
```

The checker uses Python Decimal formulas and does not import the production
library. Production unit tests must compare Rust results with this artifact;
they must not regenerate or rewrite it.

