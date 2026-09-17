# Documentation

This directory contains the library, product and model reference documentation.

The [model and numerical-method bibliography](references.md) records the
published sources for the models and calculation methods described below.

| Start here | Contents |
| --- | --- |
| [Library guide](library/README.md) | Rust and Python entry points, diagnostics, serialization, migration and benchmarking |
| [Product reference](products/README.md) | Supported financial products and their contract pages and examples |
| [Model reference](models/README.md) | Black–Scholes/Black-76, Local Volatility, Bergomi LSV, Hull–White, rough volatility and multi-asset model contracts |

Contributor checks are in [CONTRIBUTING.md](../CONTRIBUTING.md).

Numerical contracts and diagnostics stay with the library or model they describe:
they specify observable behavior, conventions, supported inputs and limitations.
Development plans and historical validation records belong in `design/`.
