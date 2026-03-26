#pragma once
#include <deque>
#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_qwen_7b_s1_p(TraceGenerator& gen);