#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test: builds fake_gpu, builds the verification binary, runs it via LD_PRELOAD,
# and shows the generated report.

BUILD_DIR=${BUILD_DIR:-build}
REPORT=${REPORT:-fake_gpu_report.json}

cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

gcc verification/verify_preload.c -o "$BUILD_DIR/verify_preload" -ldl

LD_PRELOAD="$PWD/$BUILD_DIR/libfake_gpu.so" "$BUILD_DIR/verify_preload"

echo "\nSmoke test finished. Report preview (if present):"
if [[ -f "$REPORT" ]]; then
    head -n 40 "$REPORT"
else
    echo "Report not found at $REPORT"
fi
