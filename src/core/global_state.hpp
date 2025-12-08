#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include "device.hpp"

namespace fake_gpu {

class GlobalState {
public:
    static GlobalState& instance();

    void initialize();
    int get_device_count() const;
    Device& get_device(int index);

private:
    GlobalState();
    ~GlobalState() = default;

    bool initialized = false;
    std::vector<Device> devices;
    std::mutex mutex;
};

} // namespace fake_gpu
