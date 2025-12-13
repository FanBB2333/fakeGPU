#include "cuda_defs.hpp"
#include "../core/global_state.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace fake_gpu;

extern "C" {

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidValue;
    // ensure initialized
    GlobalState::instance().initialize();
    *count = GlobalState::instance().get_device_count();
    printf("[FakeCUDA] cudaGetDeviceCount returning %d\n", *count);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    GlobalState::instance().initialize();
    int count = GlobalState::instance().get_device_count();
    if (device < 0 || device >= count) {
        printf("[FakeCUDA] cudaSetDevice invalid index %d\n", device);
        return cudaErrorInvalidDevice;
    }
    GlobalState::instance().set_current_device(device);
    printf("[FakeCUDA] cudaSetDevice(%d)\n", device);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    GlobalState::instance().initialize();
    int device = GlobalState::instance().get_current_device();
    Device& dev = GlobalState::instance().get_device(device);
    if (dev.index < 0) return cudaErrorInvalidDevice;

    // Allocate real RAM
    void* ptr = malloc(size);
    if (!ptr) {
        printf("[FakeCUDA] cudaMalloc failed to allocate %zu bytes\n", size);
        return cudaErrorMemoryAllocation;
    }

    if (!GlobalState::instance().register_allocation(ptr, size, device)) {
        printf("[FakeCUDA] cudaMalloc denied %zu bytes on device %d (over commitment)\n", size, device);
        free(ptr);
        return cudaErrorMemoryAllocation;
    }

    *devPtr = ptr;
    printf("[FakeCUDA] cudaMalloc allocated %zu bytes at %p on device %d\n", size, ptr, device);
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    printf("[FakeCUDA] cudaFree(%p)\n", devPtr);
    if (!devPtr) return cudaSuccess;

    size_t size = 0;
    int device = -1;
    bool tracked = GlobalState::instance().release_allocation(devPtr, size, device);
    if (!tracked) {
        printf("[FakeCUDA] cudaFree warning: pointer not tracked\n");
    } else {
        printf("[FakeCUDA] cudaFree released %zu bytes from device %d\n", size, device);
    }
    free(devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    printf("[FakeCUDA] cudaMemcpy count=%zu kind=%d\n", count, kind);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    GlobalState::instance().initialize();
    
    Device& dev = GlobalState::instance().get_device(device);
    if (dev.index < 0) return cudaErrorInvalidDevice;
    
    // Fill basic props
    memset(prop, 0, sizeof(cudaDeviceProp));
    snprintf(prop->name, 256, "%s", dev.name.c_str());
    prop->totalGlobalMem = dev.total_memory;
    prop->major = 8; // Ampere
    prop->minor = 0;
    
    return cudaSuccess;
}

// Stub for kernel launch - simplest signature
// For real/complex apps, they use cudaLaunchKernel (C++ API) which wraps internal C generic calls
// or use the driver API. This is basic runtime stub.
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    printf("[FakeCUDA] cudaLaunchKernel (stub) called! Grid(%d,%d,%d) Block(%d,%d,%d)\n", 
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    
    // Here lies the main logic: we do NOTHING but log it.
    // If we wanted to fuzz output, we'd need to know which args are pointers.
    
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidValue;
    *device = GlobalState::instance().get_current_device();
    printf("[FakeCUDA] cudaGetDevice returning %d\n", *device);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    printf("[FakeCUDA] cudaMemcpyAsync count=%zu kind=%d stream=%p\n", count, kind, stream);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    printf("[FakeCUDA] cudaMemset ptr=%p value=%d count=%zu\n", devPtr, value, count);
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    printf("[FakeCUDA] cudaMemsetAsync ptr=%p value=%d count=%zu stream=%p\n", devPtr, value, count, stream);
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    printf("[FakeCUDA] cudaDeviceSynchronize (no-op)\n");
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    printf("[FakeCUDA] cudaStreamSynchronize stream=%p (no-op)\n", stream);
    return cudaSuccess;
}

// Error tracking
static __thread cudaError_t last_error = cudaSuccess;

cudaError_t cudaGetLastError(void) {
    cudaError_t err = last_error;
    last_error = cudaSuccess;  // Clear error
    printf("[FakeCUDA] cudaGetLastError returning %d\n", err);
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    printf("[FakeCUDA] cudaPeekAtLastError returning %d\n", last_error);
    return last_error;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorInvalidDevice:
            return "invalid device ordinal";
        case cudaErrorNoDevice:
            return "no CUDA-capable device is detected";
        default:
            return "unknown error";
    }
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";
        default:
            return "cudaErrorUnknown";
    }
}

} // extern C
