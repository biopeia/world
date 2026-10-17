#!/usr/bin/env bash
set -euo pipefail

# Runner for TLC model checking on WorldCommitRecovery.tla
# Requirements: Java 11+
SPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SPEC_DIR}/.." && pwd)"

JAR_PATH="${TLA2TOOLS_JAR:-/tmp/tla2tools.jar}"

if [[ ! -f "$JAR_PATH" ]]; then
    echo "Downloading tla2tools.jar v1.8.0 to ${JAR_PATH}..."
    curl -L -s -f -o "$JAR_PATH" https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
fi

echo "Running TLC on specs/WorldCommitRecovery.tla with ${JAR_PATH}..."
java -XX:+UseParallelGC -cp "$JAR_PATH" tlc2.TLC -workers 1 "$@" "${SPEC_DIR}/WorldCommitRecovery.tla"
