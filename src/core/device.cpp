#include "device.hpp"
#include <cstdio>

namespace fake_gpu {

Device::Device(int idx) : index(idx), used_memory(0) {
    name = "Fake NVIDIA A100-SXM4-80GB";
    char uuid_buf[64];
    snprintf(uuid_buf, sizeof(uuid_buf), "GPU-%08x-%04x-%04x-%04x-%012x", 
             idx, 0xabcd, 0xef01, 0x2345, 0x6789abcdef00 + idx);
    uuid = std::string(uuid_buf);
    
    // 80 GB
    total_memory = 80ULL * 1024 * 1024 * 1024;
    
    char pci_buf[32];
    snprintf(pci_buf, sizeof(pci_buf), "00000000:%02x:00.0", idx + 1);
    pci_bus_id = std::string(pci_buf);
}

} // namespace fake_gpu
