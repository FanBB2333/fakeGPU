#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "../core/global_state.hpp"

namespace fake_gpu {

class ResourceMonitor {
public:
    static ResourceMonitor& instance() {
        static ResourceMonitor s_instance;
        return s_instance;
    }

    ResourceMonitor() {
        printf("[Monitor] Initialized\n");
    }

    ~ResourceMonitor() {
        dump_report();
    }

    void dump_report() {
        std::string report_path = "fake_gpu_report.json";
        GlobalState& gs = GlobalState::instance();
        int count = gs.get_device_count();
        printf("[Monitor] Dumping report to %s. GlobalState Addr: %p, Device Count: %d\n", report_path.c_str(), &gs, count);
        
        std::ofstream out(report_path);
        out << "{\n";
        out << "  \"devices\": [\n";
        
        for (int i = 0; i < count; ++i) {
            Device& dev = gs.get_device(i);
            out << "    {\n";
            out << "      \"name\": \"" << dev.name << "\",\n";
            out << "      \"uuid\": \"" << dev.uuid << "\",\n";
            out << "      \"total_memory\": " << dev.total_memory << ",\n";
            out << "      \"used_memory_peak\": " << dev.used_memory_peak << ",\n";
            out << "      \"used_memory_current\": " << dev.used_memory << "\n";
            out << "    }" << (i < count - 1 ? "," : "") << "\n";
        }
        
        out << "  ]\n";
        out << "}\n";
    }
};

// Static instance to force construction/destruction
static ResourceMonitor& s_monitor = ResourceMonitor::instance();

} // namespace fake_gpu
