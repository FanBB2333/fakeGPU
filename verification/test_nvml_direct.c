#include <stdio.h>
#include <stddef.h>

// NVML API declarations
typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_INVALID_ARGUMENT = 2
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
const char* nvmlErrorString(nvmlReturn_t result);

int main() {
    printf("=== Testing NVML API (like nvidia-smi) ===\n\n");

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("NVML initialized successfully\n\n");

    // Get device count
    unsigned int deviceCount = 0;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device count: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("Found %u GPU device(s)\n\n", deviceCount);

    // Query each device
    for (unsigned int i = 0; i < deviceCount && i < 3; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            printf("Failed to get device %u handle\n", i);
            continue;
        }

        printf("GPU %u:\n", i);

        // Get device name
        char name[256];
        result = nvmlDeviceGetName(device, name, sizeof(name));
        if (result == NVML_SUCCESS) {
            printf("  Name: %s\n", name);
        }

        // Get device UUID
        char uuid[80];
        result = nvmlDeviceGetUUID(device, uuid, sizeof(uuid));
        if (result == NVML_SUCCESS) {
            printf("  UUID: %s\n", uuid);
        }

        // Get memory info
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result == NVML_SUCCESS) {
            printf("  Total Memory: %.2f GB\n", memory.total / (1024.0 * 1024.0 * 1024.0));
            printf("  Used Memory:  %.2f GB\n", memory.used / (1024.0 * 1024.0 * 1024.0));
            printf("  Free Memory:  %.2f GB\n", memory.free / (1024.0 * 1024.0 * 1024.0));
        }

        printf("\n");
    }

    // Shutdown
    nvmlShutdown();
    printf("=== Test completed successfully ===\n");

    return 0;
}
