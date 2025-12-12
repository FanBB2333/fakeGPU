#!/usr/bin/env python3
"""
Simple Python script to test fake GPU detection using pynvml
"""
import sys
import os
from ctypes import CDLL

# Force load our fake GPU library before importing pynvml
fake_gpu_lib = os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so')
if os.path.exists(fake_gpu_lib):
    print(f"[Python] Loading fake GPU library: {fake_gpu_lib}")
    CDLL(fake_gpu_lib, mode=os.RTLD_GLOBAL)
else:
    print(f"WARNING: Fake GPU library not found at {fake_gpu_lib}")

try:
    import pynvml
except ImportError as e:
    print(f"ERROR: pynvml not installed or failed to import: {e}")
    print("Install with: pip install nvidia-ml-py3")
    sys.exit(1)

try:
    
    print("=== Testing GPU Detection with pynvml ===")
    
    # Initialize NVML
    print("[Python] Calling pynvml.nvmlInit()...")
    pynvml.nvmlInit()
    
    # Get device count
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[Python] Detected {device_count} GPU(s)")
    
    # Get info for each device
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"\n[GPU {i}]")
        print(f"  Name: {name}")
        print(f"  UUID: {uuid}")
        print(f"  Total Memory: {memory_info.total / (1024**3):.2f} GB")
        print(f"  Used Memory:  {memory_info.used / (1024**3):.2f} GB")
        print(f"  Free Memory:  {memory_info.free / (1024**3):.2f} GB")
    
    # Shutdown
    pynvml.nvmlShutdown()
    print("\n[Python] Test completed successfully!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
