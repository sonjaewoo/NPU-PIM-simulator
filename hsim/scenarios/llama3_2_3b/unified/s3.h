#pragma once
#include <deque>
#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_llama_s3_u(TraceGenerator& gen);