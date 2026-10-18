# Library portability follow-up, 2026-10-04

This follows the [macOS full-library check](library-full-check-2026-10-03.md).
The [result inventory](library-cross-platform-2026-10-04.json) retains commands,
source/image/wheel identities and the exact scope of the Linux run.

## Fixed validation input

The tested input is the previously validated, uncommitted 652-file source
archive. It was copied into a local container through a read-only mount;
no source was uploaded to GitHub. The running checkout subsequently received
concurrent Local Volatility work. That later work and this turn's new audit
utility are not part of the frozen Linux numerical test input.

- Source archive SHA-256: `d9f2f626b6e4d5bbe3113acb52c16d2ba4cf446f6438f10aa87d6b2f8cc3e486`.
- Official image: `rust:1.98.1-slim-trixie`, digest
  `sha256:4cd829461bd5c4d511c32e269da9cb8929223b666519d8004e35fc8d1d771ab7`.
- Native target: `aarch64-unknown-linux-gnu`; Debian Trixie / glibc 2.41.
- Rust 1.98.1, CPython 3.13.5; container limited to four CPUs and 4 GiB RAM.
- Local logs and artifacts: `benchmark-results/cross-platform-20261004/`.

## Results

| Gate | Result |
| --- | --- |
| Release-mode all-features Rust workspace, excluding Python crate | 711 passed; 50 ignored |
| Rust Python crate | 3 passed |
| All targets, warnings-denied Clippy and formatting | Passed |
| Release-wheel metadata, API and clean-environment smoke | 199 run, two optional-dependency skips |
| Same wheel with validation/constraints extras | 199 passed, zero skips |
| Companion package with that native wheel | 193 passed, zero skips |
| Eight independent Local Vol/path/early-exercise reference scripts | Passed |
| Schemas and snapshot Markdown links | Passed |
| European BS, Local Vol, path dependence and early exercise replay | Two independent runs per family, all byte-for-byte equal |

The 50 extended/ignored tests from the macOS record were not repeated on Linux.
This is Linux ARM64 evidence, not native Linux x86-64 or Windows evidence.
The Linux repeatability comparison is against a second execution of the same
snapshot, not a frozen supported-platform fixture. The supported-platform exact
fixture gate was not weakened and no Linux golden was added.

The built file is `rust_pricing-0.1.0-cp313-cp313-manylinux_2_34_aarch64.whl`,
with SHA-256 `55a4110a8a71467f9ca66b0d48e8797a3c14d5d24ff9e77ae293682e42434753`. It is retained under the local evidence directory's
`wheels/` subdirectory; no release or package-index upload was performed.

The earlier experimental Local Volatility paper-VegaKT gate remains unaccepted
(54/154 failures in the retained macOS run). No mathematical algorithm, numerical
acceptance budget or diagnostic-only guard was changed in this follow-up.

## Windows capture preparation

The public GitHub repository's inspected CI run belonged to
`feat/stochastic-dividend-hard-barrier-risk` at
`6b778f740d22e4a811d151a2a3b7ed3b3473cf87`; that commit did not contain the Bass
source. Its unrelated successful jobs are not evidence for this snapshot.
No new branch or CI run was published by this follow-up. A native Windows run
of the reviewed snapshot and a numerical review of its stale Local Vol fixture
remain required. Sending the unpublished snapshot to the public repository
requires the user's publication decision.

The new [candidate auditor](../../scripts/audit_local_vol_replay.py) is a
read-only review aid. It validates replay structure/platform identity and limits
differences to the 46 exact paths from the retained macOS correction audit.
It rejects changes to price, Delta/Gamma/scalar Vega, inputs, fingerprints or
other unreviewed fields. It also verifies the diagnostic labels, both
scalar/bucket/residual reconciliations and agreement of the edge diagnostics.
It refuses output paths that would overwrite a capture or repository fixture.

Eight [mutation tests](../../scripts/test_audit_local_vol_replay.py) pass on the
host, including altered prices/inputs, malformed labels, corrupted buckets,
projection discrepancies and mismatched platforms. CI configuration now includes these tests. The
existing native macOS correction still reports 46 changes and
`requires_native_numerical_review`; an unchanged capture reports `unchanged`.
Neither status certifies provenance or paper-model numerical acceptance, and
the auditor never promotes a golden file.

On native Windows with the pinned Rust toolchain, after the normal Rust/wheel
checks, capture the candidate twice:

```powershell
New-Item -ItemType Directory -Force benchmark-results | Out-Null
cargo run --locked --release -p pricing --example replay_local_vol -- benchmark-results/local-vol-native-1.json
cargo run --locked --release -p pricing --example replay_local_vol -- benchmark-results/local-vol-native-2.json
python -c "from pathlib import Path; assert Path('benchmark-results/local-vol-native-1.json').read_bytes() == Path('benchmark-results/local-vol-native-2.json').read_bytes()"
python scripts/audit_local_vol_replay.py benchmark-results/local-vol-native-1.json --output benchmark-results/local-vol-candidate-audit.json
python scripts/check_replay_fixture.py benchmark-results/local-vol-native-1.json
```

The final exact-fixture command must continue to fail while the Windows golden
still represents the old allocation. Retain both captures, their hashes,
`rustc -vV`, the source revision/manifest and the candidate audit. Run the
independent 48-check allocation probe against the same native wheel before
reviewing any golden replacement. Preserve the prior Windows fixture as
historical evidence when a reviewed replacement is eventually accepted.
