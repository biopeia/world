#!/usr/bin/env bash
set -euo pipefail

# Runner for TLC model checking on WorldCommitRecovery.tla
# Requirements: Java 11+
SPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SPEC_DIR}/.." && pwd)"

JAR_PATH="${TLA2TOOLS_JAR:-/tmp/tla2tools.jar}"
PINNED_SHA="15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac"

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Downloading tla2tools.jar v1.8.0 to ${JAR_PATH}..."
    curl -L -s -f -o "$JAR_PATH" https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
fi

ACTUAL_SHA="$(sha256sum "$JAR_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$PINNED_SHA" ]]; then
    echo "ERROR: SHA256 mismatch for ${JAR_PATH}: expected ${PINNED_SHA}, got ${ACTUAL_SHA}" >&2
    exit 1
fi

echo "Running TLC on specs/WorldCommitRecovery.tla with ${JAR_PATH} (SHA: ${PINNED_SHA:0:16}...)..."
java -XX:+UseParallelGC -cp "$JAR_PATH" tlc2.TLC -workers 1 "$@" "${SPEC_DIR}/WorldCommitRecovery.tla"
