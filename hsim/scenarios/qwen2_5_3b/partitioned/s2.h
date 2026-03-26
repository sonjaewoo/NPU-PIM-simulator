#pragma once
#include <deque>
#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_qwen_s2_p(TraceGenerator& gen);