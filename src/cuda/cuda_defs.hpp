#pragma once
#include <cstddef>
#include <cstdint>

// Struct definitions (must be visible to C++, and compatible layout for C)
typedef struct uint3 {
    unsigned int x, y, z;
} uint3;

typedef struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 grid) : x(grid.x), y(grid.y), z(grid.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
} dim3;

typedef struct cudaDeviceProp_st {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
    // ... add more if needed
} cudaDeviceProp;

// API Declarations
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidValue = 11,
    cudaErrorInvalidDevice = 10,
    cudaErrorNoDevice = 100,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef struct CUstream_st *cudaStream_t;

// Function signatures
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif
