#pragma once
#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_gemma_9b_s1_u(TraceGenerator& gen);