#pragma once
#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_gemma_s3_u(TraceGenerator& gen);