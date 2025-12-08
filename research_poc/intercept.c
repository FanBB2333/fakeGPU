#include <stdio.h>
#include <unistd.h>
#include <string.h>

// Replacement function
int my_puts(const char *s) {
    const char *prefix = "[FakeGPU:Interposed] ";
    write(1, prefix, strlen(prefix));
    write(1, s, strlen(s));
    write(1, "\n", 1);
    return 0;
}

// Macro to create the interpose section
#define DYLD_INTERPOSE(_replacement,_replacee) \
   __attribute__((used)) static struct{ const void* replacement; const void* replacee; } _interpose_##_replacee \
            __attribute__ ((section ("__DATA,__interpose"))) = { (const void*)(unsigned long)&_replacement, (const void*)(unsigned long)&_replacee };

// Declaration of the original function we want to replace
extern int puts(const char *);

// Apply the interposition
DYLD_INTERPOSE(my_puts, puts)
