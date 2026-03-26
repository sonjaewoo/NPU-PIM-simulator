# NPU-PIM-simulator

**Trace-driven NPU-PIM co-simulation framework**
This repository provides the full source code and configurations for the simulator described in our CASES 2026 paper.

We provide **timing simulations** based on user parameters.

## Features
- Cycle-level modeling of NPU, PIM, DRAM, NoC, and memory controllers
- Unified vs. Partitioned DRAM–PIM configurations
- Configurable bandwidth, latency, channel count, and PIM micro-operation timing

## Index 
- [Prerequisite](#Prerequisite)
- [Simulation Configuration](#Simulation-Configuration)
- [How to Compile](#How-to-Compile)
- [How to Run](#How-to-Run)
  
## Prerequisite
- All dependencies required by **SystemC 3.0.0**

## Simulation Configuration
- All parameters are reconfigurable by users.
- **system.json**
```bash
   "architecture" : baseline / unified / partitioned,  

// Frequency
   "host": 0.5, // GHz

// Workload
  "model": "gemma2-2B",
  "attention_head": 8,
  "input_sequence_length": 128,
  "output_sequence_length": 64,
  "decoder_block": 26
```
  
- **memory.json**
```bash
  "dram" : {
    "backend"       : "dramsim3",
    "memory_type"   : "LPDDR5",
    "chip_config"   : "8Gb_x16",
    "timing_preset" : "6400",
    "num_channels"  : 2,
    "capacity"      : 16,
    "clock_freq"    : 3.2
  },

  "pim" : {
    "backend"       : "dramsim3",
    "memory_type"   : "LPDDR5",
    "chip_config"   : "8Gb_x16",
    "timing_preset" : "6400",
    "num_channels"  : 2,
    "capacity"      : 16,
    "clock_freq"    : 3.2
  }
```

## How to Compile
```bash
git clone NPU-PIM-simulator.git
cd NPU-PIM-simulator
mkdir shared
cd hsim
mkdir build && cd build
cmake ..
make
```

## How to Run
```bash
cd /hsim/build
./hsim
```
