#include "cuda_driver_defs.hpp"
#include "../core/global_state.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace fake_gpu;

// Track current context (simplified - just track device)
static int current_context_device = 0;
static bool driver_initialized = false;

extern "C" {

CUresult cuInit(unsigned int Flags) {
    printf("[FakeCUDA-Driver] cuInit called with flags=%u\n", Flags);
    GlobalState::instance().initialize();
    driver_initialized = true;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    // Report CUDA 12.0 (12000)
    *driverVersion = 12000;
    printf("[FakeCUDA-Driver] cuDriverGetVersion returning 12000\n");
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();
    *count = GlobalState::instance().get_device_count();
    printf("[FakeCUDA-Driver] cuDeviceGetCount returning %d\n", *count);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (ordinal < 0 || ordinal >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    *device = ordinal;
    printf("[FakeCUDA-Driver] cuDeviceGet(%d) returning device %d\n", ordinal, *device);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    strncpy(name, device.name.c_str(), len - 1);
    name[len - 1] = '\0';
    printf("[FakeCUDA-Driver] cuDeviceGetName(%d) returning '%s'\n", dev, name);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Return fake but reasonable values
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *pi = 1024;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *pi = 2147483647;
            break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            *pi = 32;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *pi = 8;  // Ampere
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = 108;  // A100 has 108 SMs
            break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
            *pi = 65536;
            break;
        default:
            *pi = 0;
            break;
    }

    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    *bytes = device.total_memory;
    printf("[FakeCUDA-Driver] cuDeviceTotalMem(%d) returning %zu bytes\n", dev, *bytes);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(char *uuid, CUdevice dev) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    Device& device = GlobalState::instance().get_device(dev);
    strncpy(uuid, device.uuid.c_str(), 64);
    return CUDA_SUCCESS;
}

// Context management (simplified - we don't really need contexts)
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    printf("[FakeCUDA-Driver] cuCtxCreate for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    printf("[FakeCUDA-Driver] cuCtxDestroy(%p)\n", ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (ctx == nullptr) {
        current_context_device = 0;
    } else {
        current_context_device = (int)(uintptr_t)ctx - 1;
    }
    printf("[FakeCUDA-Driver] cuCtxSetCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    printf("[FakeCUDA-Driver] cuCtxSynchronize (no-op)\n");
    return CUDA_SUCCESS;
}

// Memory management
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;

    void* ptr = malloc(bytesize);
    if (!ptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    int device = current_context_device;
    if (!GlobalState::instance().register_allocation(ptr, bytesize, device)) {
        free(ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    printf("[FakeCUDA-Driver] cuMemAlloc allocated %zu bytes at 0x%llx on device %d\n",
           bytesize, *dptr, device);
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    void* ptr = (void*)dptr;
    size_t size;
    int device;

    if (GlobalState::instance().release_allocation(ptr, size, device)) {
        free(ptr);
        printf("[FakeCUDA-Driver] cuMemFree(0x%llx) released %zu bytes from device %d\n",
               dptr, size, device);
        return CUDA_SUCCESS;
    }

    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dstHost, (void*)srcDevice, ByteCount);
    printf("[FakeCUDA-Driver] cuMemcpyDtoH copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void*)dstDevice, srcHost, ByteCount);
    printf("[FakeCUDA-Driver] cuMemcpyHtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstDevice || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    printf("[FakeCUDA-Driver] cuMemcpyDtoD copied %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Return a fake context pointer (just use device number + 1 to avoid NULL)
    *pctx = (CUcontext)(uintptr_t)(dev + 1);
    current_context_device = dev;
    printf("[FakeCUDA-Driver] cuDevicePrimaryCtxRetain for device %d, context=%p\n", dev, *pctx);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    GlobalState::instance().initialize();

    int count = GlobalState::instance().get_device_count();
    if (dev < 0 || dev >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    printf("[FakeCUDA-Driver] cuDevicePrimaryCtxRelease for device %d\n", dev);
    return CUDA_SUCCESS;
}

// Context stack management
CUresult cuCtxPushCurrent(CUcontext ctx) {
    if (ctx == nullptr) {
        current_context_device = 0;
    } else {
        current_context_device = (int)(uintptr_t)ctx - 1;
    }
    printf("[FakeCUDA-Driver] cuCtxPushCurrent(%p) -> device %d\n", ctx, current_context_device);
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)(uintptr_t)(current_context_device + 1);
    printf("[FakeCUDA-Driver] cuCtxPopCurrent returning context %p\n", *pctx);
    return CUDA_SUCCESS;
}

// Error handling
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    static const char* error_strings[] = {
        "CUDA_SUCCESS",
        "CUDA_ERROR_INVALID_VALUE",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CUDA_ERROR_NOT_INITIALIZED",
        "CUDA_ERROR_DEINITIALIZED"
    };

    if (pStr) {
        if (error < 5) {
            *pStr = error_strings[error];
        } else {
            *pStr = "CUDA_ERROR_UNKNOWN";
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    return cuGetErrorString(error, pStr);
}

} // extern "C"
