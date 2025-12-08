#include <stdio.h>

// Forward declaration if we were linking against real CUDA, 
// but here we just manually declare it to test dynamic symbol resolution.
int cudaGetDeviceCount(int *count);

int main() {
    printf("--- Start of Target ---\n");
    
    // 1. Test standard libc interception
    puts("Hello World");
    
    // 2. Test "CUDA" interception
    // Note: Since we don't link against libcudart, this symbol won't be found
    // at compile time unless we use dlsym or are lax.
    // However, if we run with DYLD_INSERT_LIBRARIES, the symbol MIGHT be available 
    // if the library provides it and we are dynamically linked?
    // Actually, normally 'target' would be linked against 'libcudart'.
    // Since we don't have libcudart on mac, we can't link against it easily.
    // So this part is tricky to simulate without a "stub" libcudart.
    // For now, let's just stick to 'puts'.
    
    printf("--- End of Target ---\n");
    return 0;
}
