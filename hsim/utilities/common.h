#ifndef COMMON_H
#define COMMON_H

#include <tuple>
#include <string>
#include <array>
#include <unordered_map>
#include <vector>

/* (layer, (id, size, address)) */
using BaseMap = std::unordered_multimap<std::string, std::tuple<unsigned int, unsigned int, uint64_t>>;

using SyncMap = BaseMap;
using WaitMap = BaseMap;

enum DeviceType {
    HOST = 0,
    NPU,
    PIM,
    DRAM,
    SSD,
    MCU,
    NONE
};

struct SSDSummary {
    bool valid{false};
    uint64_t req_count{0};
    uint64_t req_bytes{0};
    uint64_t done_count{0};
    uint64_t done_bytes{0};
    std::string elapsed;
};

struct SSDAccessInfo {
    uint32_t size_bytes{0};
    std::string latency;
};

struct ArbiterSrcSummary {
    uint64_t reqs{0};
    double avg_wait_cycles{0.0};
    uint64_t max_wait_cycles{0};
};

struct ArbiterSummary {
    bool valid{false};
    uint64_t fw_enq{0};
    uint64_t fw_deq{0};
    uint64_t dram_enq{0};
    uint64_t dram_deq{0};
    uint64_t contention_events{0};
    uint64_t fw_q_max{0};
    double fw_q_avg{0.0};
    uint64_t fw_q_nonempty_cycles{0};
    std::array<ArbiterSrcSummary, NONE + 1> src{};
};

#endif
