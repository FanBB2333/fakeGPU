#!/usr/bin/env bash
set -euo pipefail

# Test Python GPU detection with fake GPU library

# Activate conda environment
CONDA_ENV=${CONDA_ENV:-patent}
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

BUILD_DIR=${BUILD_DIR:-build}
FAKE_GPU_LIB="$PWD/$BUILD_DIR/libfake_gpu.so"

if [[ ! -f "$FAKE_GPU_LIB" ]]; then
    echo "Error: $FAKE_GPU_LIB not found. Build first with: cmake --build build"
    exit 1
fi

echo "========================================"
echo "Testing with pynvml (NVML Python API)"
echo "========================================"
echo ""

# Check if pynvml is installed
if ! python -c "import pynvml" 2>/dev/null; then
    echo "Installing nvidia-ml-py3..."
    pip install nvidia-ml-py3 --quiet
fi

FAKE_GPU_LIB="$FAKE_GPU_LIB" LD_PRELOAD="$FAKE_GPU_LIB" python verification/test_gpu.py

echo ""
echo "========================================"
echo "Report generated:"
echo "========================================"
if [[ -f fake_gpu_report.json ]]; then
    cat fake_gpu_report.json
fi

echo ""
echo "========================================"
echo "Testing with PyTorch (optional)"
echo "========================================"
echo ""

if python -c "import torch" 2>/dev/null; then
    echo "PyTorch detected, running test..."
    FAKE_GPU_LIB="$FAKE_GPU_LIB" LD_PRELOAD="$FAKE_GPU_LIB" python verification/test_pytorch.py
else
    echo "PyTorch not installed. Skipping PyTorch test."
    echo "To test with PyTorch, install it first: pip install torch"
fi
