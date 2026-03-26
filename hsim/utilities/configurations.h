#ifndef CONFIGURATIONS_H
#define CONFIGURATIONS_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <json.h>

#include <utilities/common.h>

typedef enum _LOG_LEVEL {
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_OFF
} LOG_LEVEL;

struct MemInfo {
    uint64_t shared_offset;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t wb_offset;
    uint64_t buf_offset;
};

struct NetworkInfo {
    std::string name;
    MemInfo meminfo;
    std::string out_path;
    std::string prefix;
    bool precompiled;
};

struct CompileInfo {
    bool quantized;
    std::string midap_compiler;
    unsigned packet_size;
    unsigned midap_level;
    uint32_t fmem_bank_num;
    uint32_t fmem_bank_size;
    uint32_t cim_num;
    uint32_t wmem_size;
    uint32_t ewmem_size;
    NetworkInfo net_info;
    std::string additional_flags;
};

struct DRAMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct PIMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct DelayCount {
    unsigned int delay;
    unsigned int count;
};

struct PIMCmdProfile {
    std::unordered_map<std::string, DelayCount> cmd_map;
    std::vector<std::string> ordered_cmd_name;
};

class Configurations {
public:
    Configurations();

    Json::Value parse_json(const std::string& file_name) const;

    void init_dram();
    void init_system();
    void init_compiler();
    void init_sync_wait_map();
    void init_configurations();
    void compile_network();
    void init_host_profile();
    void init_pim_profile(const std::string& pim_type);

    bool pim_enabled() const { return enable_pim; }
    bool dram_enabled() const { return enable_dram; }

    uint32_t get_scenario() const { return scenario_id; }
    uint32_t get_packet_size() const { return compileinfo.packet_size; }
    uint32_t get_midap_level() const { return compileinfo.midap_level; }
    uint32_t get_dram_req_size() const { return dram_req_size; }
    uint64_t get_dram_capacity() const { return draminfo.capacity; }
    uint32_t get_dram_channels() const { return draminfo.channels; }
    uint32_t get_pim_channels() const { return piminfo.channels; }

    MemInfo get_meminfo() const { return compileinfo.net_info.meminfo; }
    NetworkInfo get_netinfo() const { return compileinfo.net_info; }
    SyncMap get_host_sync_map() const { return host_sync_map; }
    WaitMap get_host_wait_map() const { return host_wait_map; }
    SyncMap get_pim_sync_map() const { return pim_sync_map; }
    WaitMap get_pim_wait_map() const { return pim_wait_map; }

    double get_dram_freq() const { return draminfo.freq; }
    double get_pim_freq() const { return piminfo.freq; }

    double get_host_freq() const { return host_frequency; }

    std::string get_cycle_log_file() const { return cycle_log_file; }

    std::string get_pim_config() const { return piminfo.config; }
    std::string get_pim_type() const { return piminfo.type; }
    std::string get_pim_preset() const { return piminfo.preset; }

    std::string get_dram_config() const { return draminfo.config; }
    std::string get_dram_type() const { return draminfo.type; }
    std::string get_dram_preset() const { return draminfo.preset; }
    std::string get_dram_backend() const { return draminfo.backend; }

    std::string get_memory_structure() const { return memory_structure; }

    std::string get_net_name() const { return compileinfo.net_info.name; }
    std::string get_compile_dir() const { return compileinfo.net_info.out_path; }
    std::string get_compile_prefix() const { return compileinfo.net_info.prefix; }
    std::string get_target_model() const { return target_model; }

    const std::unordered_map<unsigned int, bool>& get_pim_sync_id_map() const { return pim_sync_id_map; }
    const std::unordered_map<unsigned int, bool>& get_pim_wait_id_map() const { return pim_wait_id_map; }
    const std::unordered_map<std::string, unsigned int>& get_layer_tile_map() const { return layer_tile_map; }
    const std::unordered_map<std::string, unsigned int>& get_host_profile() const { return host_profile; }
    const std::unordered_map<std::string, PIMCmdProfile>& get_pim_profile() const { return pim_profile; }
    const std::tuple<unsigned int, unsigned int>& get_mode_change_delay() const { return mode_change_delay; }

    void load_sync_wait_info(const std::string& file_name, BaseMap& info_map);
    void print_configurations();

    unsigned int get_attn_head_num() const { return num_attn_heads; }

private:
    void validate_system_config(const Json::Value& root) const;
    void validate_memory_config(const Json::Value& root) const;
    void validate_compiler_config(const Json::Value& root) const;

private:
    bool enable_pim{false};
    bool enable_dram{false};

    uint32_t scenario_id{0};
    uint32_t dram_req_size{0};
    std::string memory_structure;
    std::string cycle_log_file;
    CompileInfo compileinfo{};
    DRAMInfo draminfo{};
    PIMInfo piminfo{};

    unsigned int dram_to_pim_delay{0};
    unsigned int pim_to_dram_delay{0};
    unsigned int input_sequence_length{0};
    unsigned int output_sequence_length{0};
    unsigned int decoder_block{0};
    unsigned int num_attn_heads{0};
    double host_frequency{0.0};

    std::string target_model;

    std::map<std::string, int> dram_list;
    std::unordered_map<std::string, unsigned int> layer_tile_map;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;
    std::unordered_map<unsigned int, bool> pim_sync_id_map;
    std::unordered_map<unsigned int, bool> pim_wait_id_map;
    std::tuple<unsigned int, unsigned int> mode_change_delay;

    SyncMap host_sync_map;
    WaitMap host_wait_map;
    SyncMap pim_sync_map;
    WaitMap pim_wait_map;
};

#endif // CONFIGURATIONS_H
