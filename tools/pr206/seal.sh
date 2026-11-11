#!/bin/bash
set -euo pipefail
out="$RUNNER_TEMP/individual-history"
publication="$GITHUB_WORKSPACE/evidence"
mkdir -p "$publication"
test -d "$out"
if test -f "$out/summary.json"; then cp "$out/summary.json" "$publication/"; fi
# The artifact is public. Keep full process metadata in an encrypted archive.
# Only the workspace holds the decryption key. Original binaries contain no heap payload.
cd "$out"
tar -cf - . | zstd -q -1 -T1 | openssl cms -encrypt -binary -stream -aes-256-cbc -outform DER -out "$publication/individual-events.cms" "$GITHUB_WORKSPACE/diagnostics/tools/pr206/recipient.pem"
cd "$publication"
shasum -a 256 individual-events.cms > SHA256
