#include "global_state.hpp"

namespace fake_gpu {

GlobalState& GlobalState::instance() {
    static GlobalState* s_instance = new GlobalState();
    return *s_instance;
}

GlobalState::GlobalState() {
}

void GlobalState::initialize() {
    std::lock_guard<std::mutex> lock(mutex);
    printf("[GlobalState-%p] initialize called. Current devices: %lu\n", this, devices.size());
    if (initialized) return;

    // Create 8 fake devices
    for (int i = 0; i < 8; ++i) {
        devices.emplace_back(i);
    }
    initialized = true;
    printf("[GlobalState-%p] Valid devices count after init: %lu\n", this, devices.size());
}

int GlobalState::get_device_count() const {
    return devices.size();
}

Device& GlobalState::get_device(int index) {
    // In a real scenario, check bounds
    if (index < 0 || index >= devices.size()) {
        static Device failure_dev(-1);
        return failure_dev;
    }
    return devices[index];
}

} // namespace fake_gpu
