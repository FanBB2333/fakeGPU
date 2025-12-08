#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Declare expected symbols (normally from nvml.h / cuda_runtime.h)
typedef enum { NVML_SUCCESS = 0 } nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct { unsigned long long total, free, used; } nvmlMemory_t;

typedef enum { cudaSuccess = 0 } cudaError_t;

// We declare them weak or just assume they will be resolved by DYLD_INSERT_LIBRARIES
// "extern" implies they exist at link time. On Mac with -undefined dynamic_lookup this works.
extern nvmlReturn_t nvmlInit();
extern nvmlReturn_t nvmlDeviceGetCount(unsigned int *count);
extern nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
extern nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);

extern cudaError_t cudaGetDeviceCount(int *count);
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaFree(void *devPtr);

int main() {
    printf("--- Verification Start ---\n");

    // 1. NVML Test
    printf("[Test] calling nvmlInit...\n");
    if (nvmlInit() != NVML_SUCCESS) {
         printf("nvmlInit failed (or symbol not found)\n");
         return 1;
    }
    
    unsigned int nvml_count = 0;
    nvmlDeviceGetCount(&nvml_count);
    printf("[Test] NVML Device Count: %d\n", nvml_count);
    
    if (nvml_count > 0) {
        nvmlDevice_t dev;
        nvmlDeviceGetHandleByIndex(0, &dev);
        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(dev, &mem);
        printf("[Test] Device 0 Memory: %llu bytes\n", mem.total);
    }

    // 2. CUDA Test
    int cuda_count = 0;
    cudaGetDeviceCount(&cuda_count);
    printf("[Test] CUDA Device Count: %d\n", cuda_count);
    
    void* ptr = NULL;
    size_t size = 1024 * 1024 * 64; // 64 MB
    printf("[Test] Allocating 64MB via cudaMalloc...\n");
    if (cudaMalloc(&ptr, size) == cudaSuccess) {
        printf("[Test] Allocation successful at %p. Writing data...\n", ptr);
        memset(ptr, 0xAB, size); // Touch memory
        cudaFree(ptr);
        printf("[Test] Freed.\n");
    } else {
        printf("[Test] cudaMalloc failed\n");
    }

    printf("--- Verification End ---\n");
    return 0;
}
