#include "configurations.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
void require_key(const Json::Value& obj, const char* key, Json::ValueType type, const char* where) {
    if (!obj.isMember(key) || obj[key].type() != type) {
        throw std::runtime_error(std::string("Missing or invalid key '") + key + "' in " + where);
    }
}

void require_uint_key(const Json::Value& obj, const char* key, const char* where) {
    if (!obj.isMember(key)) {
        throw std::runtime_error(std::string("Missing key '") + key + "' in " + where);
    }
    const Json::Value& value = obj[key];
    if (!value.isIntegral() || value.asLargestInt() < 0) {
        throw std::runtime_error(std::string("Missing or invalid key '") + key + "' in " + where);
    }
}
} // namespace

Configurations::Configurations() {}

void Configurations::init_configurations()
{
    try {
        init_system();
        init_dram();
        init_compiler();
        compile_network();
        init_sync_wait_map();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Configuration initialization failed: ") + e.what());
    }
}

void Configurations::print_configurations()
{
    std::cout << "\n========== Simulation Configuration ==========\n";
    std::cout << "Model: " << target_model <<"\n";

    std::cout << "Frequency (GHz):\n";
    std::cout << "  ∙ Host (BP) : " << host_frequency << "\n";

    std::cout << "Components:\n";
    std::cout << "  ∙ Enable DRAM : " << (enable_dram ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Enable PIM  : " << (enable_pim  ? "Yes" : "No") << "\n";

    std::cout << "\nMapping Scenario:\n";
    std::cout << "  ∙ Scenario ID : " << scenario_id << "\n";

    std::cout << "\nDRAM-PIM Structure: " << memory_structure << "\n";

    std::cout << "\nDRAM Configuration:\n";
    if (enable_dram) {
        std::cout << "  ∙ Type              : " << draminfo.type << "_" << draminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << draminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << draminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << draminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << draminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << draminfo.freq << "\n";
    } else {
        std::cout << "  ∙ [Disabled]\n";
    }

    std::cout << "\nPIM Configuration:\n";
    if (enable_pim) {
        std::cout << "  ∙ Type              : " << piminfo.type << "_" << piminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << piminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << piminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << piminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << piminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << piminfo.freq << "\n";
    } else {
        std::cout << "  [Disabled]\n";
    }

    std::cout << "\n Workload:\n";
    std::cout << "  ∙ Input Sequence Length  : " << input_sequence_length << "\n";
    std::cout << "  ∙ Output Sequence Length : " << output_sequence_length << "\n";
    std::cout << "  ∙ Decoder Block          : " << decoder_block << "\n";
    std::cout << "==============================================\n\n";
}

void Configurations::validate_system_config(const Json::Value& root) const
{
    require_key(root, "architecture", Json::objectValue, "system.json");
    require_key(root, "workload", Json::objectValue, "system.json");
    require_key(root, "clock", Json::objectValue, "system.json");
    require_key(root, "logging", Json::objectValue, "system.json");

    require_key(root["architecture"], "memory_structure", Json::stringValue, "system.json.architecture");

    require_key(root["workload"], "model_name", Json::stringValue, "system.json.workload");
    require_uint_key(root["workload"], "attention_heads", "system.json.workload");
    require_uint_key(root["workload"], "scenario_id", "system.json.workload");
    require_uint_key(root["workload"], "input_seq_len", "system.json.workload");
    require_uint_key(root["workload"], "output_seq_len", "system.json.workload");
    require_uint_key(root["workload"], "decoder_blocks", "system.json.workload");

    if (!root["clock"].isMember("host_freq_ghz") || !root["clock"]["host_freq_ghz"].isNumeric()) {
        throw std::runtime_error("Missing or invalid key 'host_freq_ghz' in system.json.clock");
    }

    require_key(root["logging"], "cycle_log_file", Json::stringValue, "system.json.logging");
}

void Configurations::init_system()
{
    Json::Value root = parse_json("system");
    validate_system_config(root);

    const Json::Value& arch = root["architecture"];
    const Json::Value& work = root["workload"];
    const Json::Value& clk = root["clock"];

    memory_structure = arch["memory_structure"].asString();

    target_model = work["model_name"].asString();
    num_attn_heads = work["attention_heads"].asUInt();
    scenario_id = work["scenario_id"].asUInt();
    input_sequence_length = work["input_seq_len"].asUInt();
    output_sequence_length = work["output_seq_len"].asUInt();
    decoder_block = work["decoder_blocks"].asUInt();

    host_frequency = clk["host_freq_ghz"].asDouble();
    if (host_frequency <= 0.0) {
        throw std::runtime_error("Invalid host clock: must be > 0");
    }

    cycle_log_file = root["logging"].get("cycle_log_file", "cycle_log.txt").asString();
}

void Configurations::validate_memory_config(const Json::Value& root) const
{
    require_uint_key(root, "request_size_bytes", "memory.json");
    require_key(root, "dram", Json::objectValue, "memory.json");
    require_key(root, "pim", Json::objectValue, "memory.json");

    require_key(root["dram"], "enabled", Json::booleanValue, "memory.json.dram");
    require_key(root["dram"], "backend", Json::stringValue, "memory.json.dram");
    require_key(root["dram"], "type", Json::stringValue, "memory.json.dram");
    require_key(root["dram"], "chip_config", Json::stringValue, "memory.json.dram");
    require_key(root["dram"], "timing_preset", Json::stringValue, "memory.json.dram");
    require_uint_key(root["dram"], "channels", "memory.json.dram");
    require_uint_key(root["dram"], "capacity_gb", "memory.json.dram");
    if (!root["dram"].isMember("clock_freq_ghz") || !root["dram"]["clock_freq_ghz"].isNumeric()) {
        throw std::runtime_error("Missing or invalid key 'clock_freq_ghz' in memory.json.dram");
    }

    require_key(root["pim"], "enabled", Json::booleanValue, "memory.json.pim");
    require_key(root["pim"], "backend", Json::stringValue, "memory.json.pim");
    require_key(root["pim"], "type", Json::stringValue, "memory.json.pim");
    require_key(root["pim"], "chip_config", Json::stringValue, "memory.json.pim");
    require_key(root["pim"], "timing_preset", Json::stringValue, "memory.json.pim");
    require_uint_key(root["pim"], "channels", "memory.json.pim");
    require_uint_key(root["pim"], "capacity_gb", "memory.json.pim");
    if (!root["pim"].isMember("clock_freq_ghz") || !root["pim"]["clock_freq_ghz"].isNumeric()) {
        throw std::runtime_error("Missing or invalid key 'clock_freq_ghz' in memory.json.pim");
    }
}

void Configurations::init_dram()
{
    Json::Value root = parse_json("memory");
    validate_memory_config(root);

    dram_req_size = root["request_size_bytes"].asUInt(); // Bytes
    if (dram_req_size == 0) {
        throw std::runtime_error("memory.request_size_bytes must be > 0");
    }

    const Json::Value& dram = root["dram"];
    const Json::Value& pim = root["pim"];

    enable_dram = dram["enabled"].asBool();
    draminfo.backend = dram["backend"].asString();
    draminfo.type = dram["type"].asString();
    draminfo.config = dram["chip_config"].asString();
    draminfo.preset = dram["timing_preset"].asString();
    draminfo.channels = dram["channels"].asUInt();
    draminfo.capacity = dram["capacity_gb"].asUInt();
    draminfo.freq = dram["clock_freq_ghz"].asDouble();

    enable_pim = pim["enabled"].asBool();
    piminfo.backend = pim["backend"].asString();
    piminfo.type = pim["type"].asString();
    piminfo.config = pim["chip_config"].asString();
    piminfo.preset = pim["timing_preset"].asString();
    piminfo.channels = pim["channels"].asUInt();
    piminfo.capacity = pim["capacity_gb"].asUInt();
    piminfo.freq = pim["clock_freq_ghz"].asDouble();

    // Strict memory-structure consistency checks:
    // baseline:   dram=true,  pim=false
    // unified:    dram=false, pim=true
    // partitioned:dram=true,  pim=true
    if (memory_structure == "baseline") {
        if (!(enable_dram && !enable_pim)) {
            throw std::runtime_error(
                "Invalid memory config for baseline: expected dram.enabled=true and pim.enabled=false");
        }
    } else if (memory_structure == "unified") {
        if (!(!enable_dram && enable_pim)) {
            throw std::runtime_error(
                "Invalid memory config for unified: expected dram.enabled=false and pim.enabled=true");
        }
    } else if (memory_structure == "partitioned") {
        if (!(enable_dram && enable_pim)) {
            throw std::runtime_error(
                "Invalid memory config for partitioned: expected dram.enabled=true and pim.enabled=true");
        }
    }

    if (enable_dram && draminfo.freq <= 0.0) {
        throw std::runtime_error("DRAM clock must be > 0");
    }
    if (enable_pim && piminfo.freq <= 0.0) {
        throw std::runtime_error("PIM clock must be > 0");
    }

    init_host_profile();

    if (pim_enabled())
        init_pim_profile(piminfo.type);
}

void Configurations::init_host_profile()
{
    Json::Value root = parse_json("profile/host");
    require_key(root, "host", Json::objectValue, "profile/host.json");

    host_profile.clear();
    const Json::Value& host = root["host"];
    for (const auto& op : host.getMemberNames()) {
        if (!host[op].isIntegral() || host[op].asLargestInt() < 0) {
            throw std::runtime_error("Invalid host profile latency for op: " + op);
        }
        host_profile[op] = host[op].asUInt();
    }
}

void Configurations::init_pim_profile(const std::string& pim_type)
{
    std::string file_name = "profile/";

    if (target_model == "moe") {
        file_name += "moe/";
    } else if (target_model == "llama3_2_3b") {
        file_name += "llama/";
    } else if (target_model == "qwen2_5_3b") {
        file_name += "qwen/";
    } else if (target_model == "qwen2_5_0_5b") {
        file_name += "qwen0_5b/";
    } else if (target_model == "qwen2_5_7b") {
        file_name += "qwen7b/";
    }

    if (input_sequence_length == 1024) {
        file_name += pim_type + "_1024";
    } else if (input_sequence_length == 512) {
        file_name += pim_type + "_512";
    } else {
        file_name += pim_type;
    }

    if (target_model == "gemma2_9b") {
        file_name += "_9b";
    }

    Json::Value root = parse_json(file_name);

    require_key(root, "mode_change_delay", Json::objectValue, (file_name + ".json").c_str());
    require_key(root, "operations", Json::objectValue, (file_name + ".json").c_str());
    require_key(root, "layers", Json::objectValue, (file_name + ".json").c_str());

    const Json::Value& mode_delay = root["mode_change_delay"];
    require_uint_key(mode_delay, "PIM_to_DRAM", "mode_change_delay");
    require_uint_key(mode_delay, "DRAM_to_PIM", "mode_change_delay");

    pim_to_dram_delay = mode_delay["PIM_to_DRAM"].asUInt();
    dram_to_pim_delay = mode_delay["DRAM_to_PIM"].asUInt();
    mode_change_delay = std::make_tuple(pim_to_dram_delay, dram_to_pim_delay);

    const Json::Value& operations = root["operations"];
    const Json::Value& layers = root["layers"];

    pim_profile.clear();
    layer_tile_map.clear();

    for (const auto& name : layers.getMemberNames()) {
        std::string operation = layers[name].asString();
        if (!operations.isMember(operation) || !operations[operation].isArray()) {
            throw std::runtime_error("Missing operation profile array for operation '" + operation + "'");
        }

        const Json::Value& cmd_array = operations[operation];
        PIMCmdProfile profile;
        bool has_common_latency = false;
        unsigned int common_latency_ns = 0;

        for (const auto& entry : cmd_array) {
            if (entry.isMember("output_tile")) {
                if (!entry["output_tile"].isIntegral() || entry["output_tile"].asLargestInt() < 0) {
                    throw std::runtime_error("Invalid 'output_tile' value in operation '" + operation + "'");
                }
                layer_tile_map[name] = entry["output_tile"].asUInt();
                continue;
            }

            // Common operation latency entry: { "latency_ns": <int> }
            if (entry.isMember("latency_ns") && !entry.isMember("name") && !entry.isMember("count")) {
                if (!entry["latency_ns"].isIntegral() || entry["latency_ns"].asLargestInt() < 0) {
                    throw std::runtime_error("Missing or invalid common 'latency_ns' in operation '" + operation + "'");
                }
                has_common_latency = true;
                common_latency_ns = entry["latency_ns"].asUInt();
                continue;
            }

            if (!entry.isMember("name") || !entry["name"].isString()) {
                throw std::runtime_error("Missing or invalid 'name' in operation '" + operation + "'");
            }
            if (!entry.isMember("count") || !entry["count"].isIntegral() || entry["count"].asLargestInt() < 0) {
                throw std::runtime_error("Missing or invalid 'count' in operation '" + operation + "'");
            }

            std::string cmd = entry["name"].asString();
            unsigned int delay_ns = 0;
            if (entry.isMember("latency_ns")) {
                if (!entry["latency_ns"].isIntegral() || entry["latency_ns"].asLargestInt() < 0) {
                    throw std::runtime_error("Invalid 'latency_ns' in command '" + cmd + "' of operation '" + operation + "'");
                }
                delay_ns = entry["latency_ns"].asUInt();
            } else if (has_common_latency) {
                delay_ns = common_latency_ns;
            } else {
                throw std::runtime_error(
                    "Missing 'latency_ns' for command '" + cmd +
                    "' in operation '" + operation + "' and no common latency is defined");
            }

            profile.cmd_map[cmd] = {delay_ns, entry["count"].asUInt()};
            profile.ordered_cmd_name.push_back(cmd);
        }

        pim_profile[name] = profile;
    }
}

void Configurations::validate_compiler_config(const Json::Value& compiler) const
{
    require_key(compiler, "quantized", Json::booleanValue, "compiler.json");
    require_key(compiler, "additional_flags", Json::stringValue, "compiler.json");
    require_key(compiler, "layer_compiler", Json::stringValue, "compiler.json");
    require_uint_key(compiler, "packet_size", "compiler.json");
    require_uint_key(compiler, "midap_level", "compiler.json");
    require_uint_key(compiler, "fmem_bank_num", "compiler.json");
    require_uint_key(compiler, "fmem_bank_size", "compiler.json");
    require_uint_key(compiler, "cim_num", "compiler.json");
    require_uint_key(compiler, "wmem_size", "compiler.json");
    require_uint_key(compiler, "ewmem_size", "compiler.json");
    require_key(compiler, "network", Json::arrayValue, "compiler.json");

    if (compiler["network"].empty() || !compiler["network"][0].isObject()) {
        throw std::runtime_error("compiler.json 'network' must contain at least one object");
    }
}

void Configurations::init_compiler()
{
    Json::Value compiler = parse_json("compiler");
    validate_compiler_config(compiler);

    compileinfo.quantized = compiler["quantized"].asBool();
    compileinfo.additional_flags = compiler["additional_flags"].asString();

    compileinfo.midap_compiler = compiler["layer_compiler"].asString();
    if (compileinfo.midap_compiler != "MIN_DRAM_ACCESS" &&
        compileinfo.midap_compiler != "HIDE_DRAM_LATENCY" &&
        compileinfo.midap_compiler != "DOUBLE_BUFFER") {
        throw std::runtime_error("Invalid compiler option: " + compileinfo.midap_compiler);
    }

    compileinfo.packet_size = compiler["packet_size"].asUInt();
    compileinfo.midap_level = compiler["midap_level"].asUInt();
    compileinfo.fmem_bank_num = compiler["fmem_bank_num"].asUInt();
    compileinfo.fmem_bank_size = compiler["fmem_bank_size"].asUInt();
    compileinfo.cim_num = compiler["cim_num"].asUInt();
    compileinfo.wmem_size = compiler["wmem_size"].asUInt();
    compileinfo.ewmem_size = compiler["ewmem_size"].asUInt();

    const Json::Value& network = compiler["network"];
    const Json::Value& net = network[0];

    require_key(net, "shared_offset", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "input_offset", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "output_offset", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "wb_offset", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "buf_offset", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "path", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "prefix", Json::stringValue, "compiler.json.network[0]");
    require_key(net, "precompiled", Json::booleanValue, "compiler.json.network[0]");

    if (target_model == "gemma2_2b") {
        compileinfo.net_info.name = "gemma2_layer_pim_offload";
    } else if (target_model == "gemma2_9b") {
        compileinfo.net_info.name = "gemma9b_layer";
    } else if (target_model == "moe") {
        compileinfo.net_info.name = "moe_layer";
    } else if (target_model == "llama3_2_3b") {
        compileinfo.net_info.name = "llama_layer";
    } else if (target_model == "qwen2_5_3b") {
        compileinfo.net_info.name = "qwen_layer";
    } else if (target_model == "qwen2_5_0_5b") {
        compileinfo.net_info.name = "qwen0_5b_layer";
    } else if (target_model == "qwen2_5_7b") {
        compileinfo.net_info.name = "qwen7b_layer";
    }

    std::stringstream shared_off_stream(net["shared_offset"].asString());
    std::stringstream input_off_stream(net["input_offset"].asString());
    std::stringstream output_off_stream(net["output_offset"].asString());
    std::stringstream wb_off_stream(net["wb_offset"].asString());
    std::stringstream buf_off_stream(net["buf_offset"].asString());

    shared_off_stream >> std::hex >> compileinfo.net_info.meminfo.shared_offset;
    input_off_stream >> std::hex >> compileinfo.net_info.meminfo.input_offset;
    output_off_stream >> std::hex >> compileinfo.net_info.meminfo.output_offset;
    wb_off_stream >> std::hex >> compileinfo.net_info.meminfo.wb_offset;
    buf_off_stream >> std::hex >> compileinfo.net_info.meminfo.buf_offset;

    std::stringstream temp;
    compileinfo.net_info.out_path = net["path"].asString();
    temp << ROOT_PATH << compileinfo.net_info.out_path << "/";
    compileinfo.net_info.out_path = temp.str();

    compileinfo.net_info.prefix = net["prefix"].asString();
    compileinfo.net_info.precompiled = net["precompiled"].asBool();
}

void Configurations::compile_network()
{
    if (compileinfo.net_info.precompiled) {
        return;
    }

    std::stringstream command;
    command << "cd " << std::filesystem::path(ROOT_PATH) / "MIDAPSim" << " ; "
            << "python3 tools/test_system.py"
            << " -n " << compileinfo.net_info.name
            << " -d DMA -fs -so "
            << (compileinfo.quantized ? "-q " : "")
            << compileinfo.additional_flags << " "
            << " -sd " << compileinfo.net_info.out_path
            << " -sp " << compileinfo.net_info.prefix
            << " -f " << compileinfo.fmem_bank_num << " " << compileinfo.fmem_bank_size
            << " -w " << compileinfo.cim_num << " " << compileinfo.wmem_size << " " << compileinfo.ewmem_size
            << " -ps " << compileinfo.packet_size
            << " -mo " << compileinfo.net_info.meminfo.shared_offset
            << " " << compileinfo.net_info.meminfo.input_offset
            << " " << compileinfo.net_info.meminfo.output_offset
            << " " << compileinfo.net_info.meminfo.wb_offset
            << " " << compileinfo.net_info.meminfo.buf_offset;

    std::cout << command.str() << "\n";

    int ret = std::system(command.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("MIDAP compilation failed with return code: " + std::to_string(ret));
    }

    compileinfo.net_info.precompiled = true;
}

void Configurations::init_sync_wait_map()
{
    NetworkInfo netinfo = get_netinfo();
    std::string base_path = netinfo.out_path + netinfo.prefix + "/core_1/";

    static const std::array<std::pair<std::string, SyncMap&>, 2> sync_maps = {{
        {"cpu", host_sync_map},
        {"pim", pim_sync_map}
    }};

    static const std::array<std::pair<std::string, WaitMap&>, 2> wait_maps = {{
        {"cpu", host_wait_map},
        {"pim", pim_wait_map}
    }};

    for (const auto& [device_name, sync_map] : sync_maps) {
        load_sync_wait_info(base_path + device_name + "_sync_info.txt", sync_map);
    }

    for (const auto& [device_name, wait_map] : wait_maps) {
        load_sync_wait_info(base_path + device_name + "_wait_info.txt", wait_map);
    }

    std::cout << "[HOST_SYNC_MAP]:\n";
    for (const auto& [key, value] : host_sync_map) {
        unsigned int id, size;
        uint64_t address;
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }

    std::cout << "[HOST_WAIT_MAP]:\n";
    for (const auto& [key, value] : host_wait_map) {
        unsigned int id, size;
        uint64_t address;
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }

    std::cout << "[PIM_SYNC_MAP]:\n";
    for (const auto& [key, value] : pim_sync_map) {
        unsigned int id, size;
        uint64_t address;
        std::tie(id, size, address) = value;
        pim_sync_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " << address << "\n";
    }

    std::cout << "[PIM_WAIT_MAP]:\n";
    for (const auto& [key, value] : pim_wait_map) {
        unsigned int id, size;
        uint64_t address;
        std::tie(id, size, address) = value;
        pim_wait_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " << address << "\n";
    }
}

void Configurations::load_sync_wait_info(const std::string& file_path, BaseMap& info_map)
{
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << "\n";
        return;
    }

    std::string line;

    while (getline(file, line)) {
        std::istringstream ss(line);
        std::string layer;
        unsigned int id, dim;
        uint64_t mem_id, offset;

        if (!(ss >> id) || ss.get() != ',' ||
            !(ss >> mem_id) || ss.get() != ',' ||
            !(ss >> offset) || ss.get() != ',' ||
            !(ss >> dim) || ss.get() != ',' ||
            !getline(ss, layer)) {
            std::cerr << "Warning: Invalid line format -> " << line << "\n";
            continue;
        }

        uint64_t base_address = (mem_id == 1)
                                    ? compileinfo.net_info.meminfo.input_offset
                                    : compileinfo.net_info.meminfo.output_offset;

        info_map.insert({layer, {id, dim * 2, base_address + offset}}); // FP16(2B)
    }
}

Json::Value Configurations::parse_json(const std::string& file_name) const
{
    std::string file_path = std::string(ROOT_PATH) + "hsim/configs/" + file_name + ".json";

    std::ifstream ifs(file_path);
    if (!ifs) {
        throw std::runtime_error(file_path + " file doesn't exist.");
    }

    std::string raw_json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    JSONCPP_STRING err;
    Json::Value root;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());

    const bool parsed = reader->parse(raw_json.c_str(),
                                      raw_json.c_str() + raw_json.length(),
                                      &root,
                                      &err);
    if (!parsed) {
        throw std::runtime_error("Failed to parse json (" + file_name + "): " + err);
    }

    return root;
}
