#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>

// Forward declarations of our fake functions
extern "C" {
    // NVML functions
    void* nvmlInit();
    void* nvmlShutdown();
    void* nvmlDeviceGetCount();
    void* nvmlDeviceGetHandleByIndex();
    void* nvmlDeviceGetName();
    void* nvmlDeviceGetUUID();
    void* nvmlDeviceGetMemoryInfo();
    void* nvmlDeviceGetPciInfo();

    // CUDA Runtime functions
    void* cudaGetDeviceCount();
    void* cudaSetDevice();
    void* cudaMalloc();
    void* cudaFree();
    void* cudaMemcpy();
    void* cudaGetDeviceProperties();
    void* cudaLaunchKernel();
    void* cudaGetErrorString();

    // CUDA Driver functions
    void* cuInit();
    void* cuDriverGetVersion();
    void* cuDeviceGetCount();
    void* cuDeviceGet();
    void* cuDeviceGetName();
    void* cuDeviceGetAttribute();
    void* cuDeviceTotalMem();
    void* cuDeviceGetUuid();
    void* cuCtxCreate();
    void* cuCtxDestroy();
    void* cuCtxSetCurrent();
    void* cuCtxGetCurrent();
    void* cuCtxSynchronize();
    void* cuMemAlloc();
    void* cuMemFree();
    void* cuMemcpyDtoH();
    void* cuMemcpyHtoD();
    void* cuMemcpyDtoD();
    void* cuGetErrorString();
    void* cuGetErrorName();
}

// Store the real dlopen and dlsym
typedef void* (*dlopen_fn)(const char*, int);
typedef void* (*dlsym_fn)(void*, const char*);
typedef void* (*dlvsym_fn)(void*, const char*, const char*);
typedef int (*dlclose_fn)(void*);

static dlopen_fn real_dlopen = nullptr;
static dlsym_fn real_dlsym = nullptr;
static dlvsym_fn real_dlvsym = nullptr;
static dlclose_fn real_dlclose = nullptr;

// Fake handle to return for intercepted libraries
static void* FAKE_HANDLE = (void*)0xFADE0001;

// Flag to prevent recursion during initialization
static __thread bool in_init = false;

// Initialize real functions using a safe method
__attribute__((constructor))
static void init_real_functions() {
    if (in_init) return;
    in_init = true;

    // Use RTLD_NEXT to get the real functions
    // This is safe because we're in a constructor, not in an intercepted call
    real_dlopen = (dlopen_fn)dlsym(RTLD_NEXT, "dlopen");
    real_dlsym = (dlsym_fn)dlsym(RTLD_NEXT, "dlsym");
    real_dlvsym = (dlvsym_fn)dlsym(RTLD_NEXT, "dlvsym");
    real_dlclose = (dlclose_fn)dlsym(RTLD_NEXT, "dlclose");

    in_init = false;
}

// Check if a library name should be intercepted
static bool should_intercept(const char* filename) {
    if (!filename) return false;

    std::string name(filename);

    // Check for NVIDIA libraries
    if (name.find("libnvidia-ml.so") != std::string::npos) {
        printf("[DL-Intercept] Intercepting libnvidia-ml.so\n");
        return true;
    }
    if (name.find("libcuda.so") != std::string::npos) {
        printf("[DL-Intercept] Intercepting libcuda.so\n");
        return true;
    }
    if (name.find("libcudart.so") != std::string::npos) {
        printf("[DL-Intercept] Intercepting libcudart.so\n");
        return true;
    }

    return false;
}

// Map symbol names to our fake functions
static void* get_fake_symbol(const char* symbol) {
    if (!symbol) return nullptr;

    // NVML symbols
    if (strcmp(symbol, "nvmlInit") == 0 || strcmp(symbol, "nvmlInit_v2") == 0) {
        return (void*)nvmlInit;
    }
    if (strcmp(symbol, "nvmlShutdown") == 0) {
        return (void*)nvmlShutdown;
    }
    if (strcmp(symbol, "nvmlDeviceGetCount") == 0 || strcmp(symbol, "nvmlDeviceGetCount_v2") == 0) {
        return (void*)nvmlDeviceGetCount;
    }
    if (strcmp(symbol, "nvmlDeviceGetHandleByIndex") == 0 || strcmp(symbol, "nvmlDeviceGetHandleByIndex_v2") == 0) {
        return (void*)nvmlDeviceGetHandleByIndex;
    }
    if (strcmp(symbol, "nvmlDeviceGetName") == 0) {
        return (void*)nvmlDeviceGetName;
    }
    if (strcmp(symbol, "nvmlDeviceGetUUID") == 0) {
        return (void*)nvmlDeviceGetUUID;
    }
    if (strcmp(symbol, "nvmlDeviceGetMemoryInfo") == 0) {
        return (void*)nvmlDeviceGetMemoryInfo;
    }
    if (strcmp(symbol, "nvmlDeviceGetPciInfo") == 0 || strcmp(symbol, "nvmlDeviceGetPciInfo_v3") == 0) {
        return (void*)nvmlDeviceGetPciInfo;
    }

    // CUDA Runtime symbols
    if (strcmp(symbol, "cudaGetDeviceCount") == 0) {
        return (void*)cudaGetDeviceCount;
    }
    if (strcmp(symbol, "cudaSetDevice") == 0) {
        return (void*)cudaSetDevice;
    }
    if (strcmp(symbol, "cudaMalloc") == 0) {
        return (void*)cudaMalloc;
    }
    if (strcmp(symbol, "cudaFree") == 0) {
        return (void*)cudaFree;
    }
    if (strcmp(symbol, "cudaMemcpy") == 0) {
        return (void*)cudaMemcpy;
    }
    if (strcmp(symbol, "cudaGetDeviceProperties") == 0) {
        return (void*)cudaGetDeviceProperties;
    }
    if (strcmp(symbol, "cudaLaunchKernel") == 0) {
        return (void*)cudaLaunchKernel;
    }
    if (strcmp(symbol, "cudaGetErrorString") == 0) {
        return (void*)cudaGetErrorString;
    }

    // CUDA Driver symbols
    if (strcmp(symbol, "cuInit") == 0) {
        return (void*)cuInit;
    }
    if (strcmp(symbol, "cuDriverGetVersion") == 0) {
        return (void*)cuDriverGetVersion;
    }
    if (strcmp(symbol, "cuDeviceGetCount") == 0) {
        return (void*)cuDeviceGetCount;
    }
    if (strcmp(symbol, "cuDeviceGet") == 0) {
        return (void*)cuDeviceGet;
    }
    if (strcmp(symbol, "cuDeviceGetName") == 0) {
        return (void*)cuDeviceGetName;
    }
    if (strcmp(symbol, "cuDeviceGetAttribute") == 0) {
        return (void*)cuDeviceGetAttribute;
    }
    if (strcmp(symbol, "cuDeviceTotalMem") == 0 || strcmp(symbol, "cuDeviceTotalMem_v2") == 0) {
        return (void*)cuDeviceTotalMem;
    }
    if (strcmp(symbol, "cuDeviceGetUuid") == 0) {
        return (void*)cuDeviceGetUuid;
    }
    if (strcmp(symbol, "cuCtxCreate") == 0 || strcmp(symbol, "cuCtxCreate_v2") == 0) {
        return (void*)cuCtxCreate;
    }
    if (strcmp(symbol, "cuCtxDestroy") == 0 || strcmp(symbol, "cuCtxDestroy_v2") == 0) {
        return (void*)cuCtxDestroy;
    }
    if (strcmp(symbol, "cuCtxSetCurrent") == 0) {
        return (void*)cuCtxSetCurrent;
    }
    if (strcmp(symbol, "cuCtxGetCurrent") == 0) {
        return (void*)cuCtxGetCurrent;
    }
    if (strcmp(symbol, "cuCtxSynchronize") == 0) {
        return (void*)cuCtxSynchronize;
    }
    if (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) {
        return (void*)cuMemAlloc;
    }
    if (strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0) {
        return (void*)cuMemFree;
    }
    if (strcmp(symbol, "cuMemcpyDtoH") == 0 || strcmp(symbol, "cuMemcpyDtoH_v2") == 0) {
        return (void*)cuMemcpyDtoH;
    }
    if (strcmp(symbol, "cuMemcpyHtoD") == 0 || strcmp(symbol, "cuMemcpyHtoD_v2") == 0) {
        return (void*)cuMemcpyHtoD;
    }
    if (strcmp(symbol, "cuMemcpyDtoD") == 0 || strcmp(symbol, "cuMemcpyDtoD_v2") == 0) {
        return (void*)cuMemcpyDtoD;
    }
    if (strcmp(symbol, "cuGetErrorString") == 0) {
        return (void*)cuGetErrorString;
    }
    if (strcmp(symbol, "cuGetErrorName") == 0) {
        return (void*)cuGetErrorName;
    }

    return nullptr;
}

extern "C" {

// Intercept dlopen
void* dlopen(const char* filename, int flag) {
    // Prevent recursion
    if (in_init || !real_dlopen) {
        init_real_functions();
    }

    if (should_intercept(filename)) {
        // Return our fake handle
        return FAKE_HANDLE;
    }

    // Call real dlopen for other libraries
    return real_dlopen(filename, flag);
}

// Intercept dlsym
void* dlsym(void* handle, const char* symbol) {
    // Prevent recursion
    if (in_init || !real_dlsym) {
        // During initialization, we can't intercept
        return nullptr;
    }

    // If this is our fake handle, return our fake symbols
    if (handle == FAKE_HANDLE) {
        void* fake_sym = get_fake_symbol(symbol);
        if (fake_sym) {
            printf("[DL-Intercept] dlsym(%p, '%s') -> fake function\n", handle, symbol);
            return fake_sym;
        }
        printf("[DL-Intercept] dlsym(%p, '%s') -> symbol not found\n", handle, symbol);
        return nullptr;
    }

    // For RTLD_DEFAULT or RTLD_NEXT, check if it's a GPU symbol
    if (handle == RTLD_DEFAULT || handle == RTLD_NEXT) {
        void* fake_sym = get_fake_symbol(symbol);
        if (fake_sym) {
            printf("[DL-Intercept] dlsym(RTLD_*, '%s') -> fake function\n", symbol);
            return fake_sym;
        }
    }

    // Call real dlsym for other cases
    return real_dlsym(handle, symbol);
}

// Intercept dlvsym (versioned symbol lookup)
void* dlvsym(void* handle, const char* symbol, const char* version) {
    // Prevent recursion
    if (in_init || !real_dlvsym) {
        init_real_functions();
    }

    // If this is our fake handle, return our fake symbols
    if (handle == FAKE_HANDLE) {
        void* fake_sym = get_fake_symbol(symbol);
        if (fake_sym) {
            printf("[DL-Intercept] dlvsym(%p, '%s', '%s') -> fake function\n", handle, symbol, version);
            return fake_sym;
        }
        return nullptr;
    }

    // For RTLD_DEFAULT or RTLD_NEXT, check if it's a GPU symbol
    if (handle == RTLD_DEFAULT || handle == RTLD_NEXT) {
        void* fake_sym = get_fake_symbol(symbol);
        if (fake_sym) {
            printf("[DL-Intercept] dlvsym(RTLD_*, '%s', '%s') -> fake function\n", symbol, version);
            return fake_sym;
        }
    }

    // Call real dlvsym for other cases
    return real_dlvsym(handle, symbol, version);
}

// Intercept dlclose
int dlclose(void* handle) {
    // Prevent recursion
    if (in_init || !real_dlclose) {
        init_real_functions();
    }

    // If this is our fake handle, just return success
    if (handle == FAKE_HANDLE) {
        printf("[DL-Intercept] dlclose(%p) -> fake handle\n", handle);
        return 0;
    }

    // Call real dlclose for other handles
    return real_dlclose(handle);
}

} // extern "C"
