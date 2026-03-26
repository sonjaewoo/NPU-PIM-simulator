cmake_minimum_required(VERSION 3.14)
project(hsim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(.
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/include
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/spdlog/include
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp/include
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/ext/headers
	)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/lib-linux64
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/spdlog
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3
	)

add_definitions(
	-DROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../"
	-DDRAM_DRAMSIM3
	-DDRAMSIM3_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/"
	)

link_libraries(rt systemc-3.0.0 spdlog jsoncpp dramsim3)

add_executable(hsim
	sc_main.cpp
	components/host.cpp
	components/arbiter.cpp
	components/midap_wrapper.cpp
	components/mem_wrapper.cpp
	components/pim_wrapper.cpp
	components/trace_generator.cpp
	utilities/mm.cpp
	utilities/logger.cpp
	utilities/packet_buffer.cpp
	utilities/shmem_communicator.cpp
	utilities/configurations.cpp
	scenarios/baseline.cpp
	scenarios/gemma2_2b/partitioned/s1.cpp
	scenarios/gemma2_2b/partitioned/s2.cpp
	scenarios/gemma2_2b/partitioned/s3.cpp
	scenarios/gemma2_2b/partitioned/s4.cpp
	scenarios/gemma2_2b/unified/s1.cpp
	scenarios/gemma2_2b/unified/s2.cpp
	scenarios/gemma2_2b/unified/s3.cpp
	scenarios/gemma2_2b/unified/s4.cpp
	scenarios/gemma2_9b/partitioned/s1.cpp
	scenarios/gemma2_9b/partitioned/s2.cpp
	scenarios/gemma2_9b/partitioned/s3.cpp
	scenarios/gemma2_9b/unified/s1.cpp
	scenarios/gemma2_9b/unified/s2.cpp
	scenarios/gemma2_9b/unified/s3.cpp
	scenarios/llama3_2_3b/baseline.cpp
	scenarios/llama3_2_3b/partitioned/s1.cpp
	scenarios/llama3_2_3b/partitioned/s2.cpp
	scenarios/llama3_2_3b/partitioned/s3.cpp
	scenarios/llama3_2_3b/unified/s1.cpp
	scenarios/llama3_2_3b/unified/s2.cpp
	scenarios/llama3_2_3b/unified/s3.cpp
	scenarios/qwen2_5_3b/baseline.cpp
	scenarios/qwen2_5_3b/partitioned/s1.cpp
	scenarios/qwen2_5_3b/partitioned/s2.cpp
	scenarios/qwen2_5_3b/partitioned/s3.cpp
	scenarios/qwen2_5_3b/unified/s1.cpp
	scenarios/qwen2_5_3b/unified/s2.cpp
	scenarios/qwen2_5_3b/unified/s3.cpp
	scenarios/qwen2_5_0_5b/test.cpp
	scenarios/qwen2_5_7b/partitioned/s1.cpp
	scenarios/qwen2_5_7b/unified/s2.cpp
	scenarios/qwen2_5_7b/baseline.cpp
	)
