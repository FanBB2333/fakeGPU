#include <stdio.h>
#include <stddef.h>

// CUDA Driver API declarations
typedef enum {
    CUDA_SUCCESS = 0
} CUresult;

typedef int CUdevice;

typedef enum {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

int main() {
    printf("=== Testing CUDA Driver API ===\n");

    // Initialize
    CUresult res = cuInit(0);
    printf("cuInit: %d\n", res);

    // Get device count
    int count = 0;
    res = cuDeviceGetCount(&count);
    printf("cuDeviceGetCount: %d devices\n", count);

    // Get device
    for (int i = 0; i < count && i < 3; i++) {
        CUdevice dev;
        res = cuDeviceGet(&dev, i);
        printf("\nDevice %d:\n", i);

        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        printf("  Name: %s\n", name);

        size_t mem;
        cuDeviceTotalMem(&mem, dev);
        printf("  Memory: %zu bytes (%.2f GB)\n", mem, mem / (1024.0 * 1024.0 * 1024.0));

        int major, minor;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        printf("  Compute: %d.%d\n", major, minor);
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
