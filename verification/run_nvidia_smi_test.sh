#!/usr/bin/env bash
set -euo pipefail

# Test nvidia-smi with fake GPU library

BUILD_DIR=${BUILD_DIR:-build}
FAKE_GPU_LIB="$PWD/$BUILD_DIR/libfake_gpu.so"

if [[ ! -f "$FAKE_GPU_LIB" ]]; then
    echo "Error: $FAKE_GPU_LIB not found. Build first with: cmake --build build"
    exit 1
fi

echo "========================================"
echo "Testing nvidia-smi with Fake GPU"
echo "========================================"
echo ""

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found in PATH"
    echo "This is expected if you don't have NVIDIA drivers installed"
    echo ""
    echo "To test, you would run:"
    echo "  LD_PRELOAD=$FAKE_GPU_LIB nvidia-smi"
    exit 0
fi

# Create symlink for libnvidia-ml.so.1 if it doesn't exist
if [[ ! -L "$BUILD_DIR/libnvidia-ml.so.1" ]]; then
    echo "Creating symlink: $BUILD_DIR/libnvidia-ml.so.1 -> libfake_gpu.so"
    ln -sf libfake_gpu.so "$BUILD_DIR/libnvidia-ml.so.1"
fi

echo "Note: nvidia-smi on systems with real NVIDIA drivers will use the system's"
echo "libnvidia-ml.so library, which has higher priority than LD_LIBRARY_PATH."
echo ""
echo "Method 1: Using LD_PRELOAD (may not work if nvidia-smi uses dlopen)"
echo "----------------------------------------"
LD_PRELOAD="$FAKE_GPU_LIB" nvidia-smi 2>&1 | head -20 || true
echo ""

echo "Method 2: Using LD_LIBRARY_PATH (system library usually has priority)"
echo "----------------------------------------"
LD_LIBRARY_PATH="$PWD/$BUILD_DIR:$LD_LIBRARY_PATH" nvidia-smi 2>&1 | head -20 || true
echo ""

echo "========================================"
echo "Analysis"
echo "========================================"
echo ""
echo "If you see real GPU information above, it means:"
echo "1. nvidia-smi uses dlopen() to load libnvidia-ml.so at runtime"
echo "2. The system library (/lib/x86_64-linux-gnu/libnvidia-ml.so.1) has priority"
echo "3. LD_PRELOAD cannot intercept dlopen() without additional dlopen interception"
echo ""
echo "To fully test fake GPU with nvidia-smi, you would need to:"
echo "1. Test on a system without real NVIDIA drivers, OR"
echo "2. Temporarily rename/move the system's libnvidia-ml.so.1, OR"
echo "3. Implement dlopen/dlsym interception (complex and error-prone)"
echo ""
echo "Our fake GPU library works correctly with direct linking (see smoke test)"
echo "and with LD_PRELOAD for programs that don't use dlopen."
echo ""
echo "========================================"
echo "Demonstration: NVML Direct Test"
echo "========================================"
echo ""
echo "This test program links directly to our fake GPU library and demonstrates"
echo "that the NVML API implementation works correctly (same API used by nvidia-smi):"
echo ""

# Compile and run the direct NVML test if source exists
if [[ -f "verification/test_nvml_direct.c" ]]; then
    if gcc -o "$BUILD_DIR/test_nvml_direct" verification/test_nvml_direct.c "$FAKE_GPU_LIB" 2>/dev/null; then
        LD_LIBRARY_PATH="$PWD/$BUILD_DIR" "$BUILD_DIR/test_nvml_direct"
    else
        echo "Failed to compile test_nvml_direct.c"
    fi
else
    echo "test_nvml_direct.c not found"
fi
echo ""
