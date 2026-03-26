#include "baseline.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_baseline(TraceGenerator& gen)
{
    gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {DRAM});
    gen.add_host_trace("RMSNorm_D_pre_attn", {}, {DRAM});

    gen.add_trace(TraceType::COMPUTE, "", NPU); /* Start MIDAP */

    gen.add_host_trace("Softmax_D_H0", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H1", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H2", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H3", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H4", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H5", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H6", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H7", {DRAM}, {DRAM});

    gen.add_host_trace("RMSNorm_D_post_attn", {DRAM}, {DRAM});
    gen.add_host_trace("D_residual_conn1",    {DRAM}, {DRAM});
    gen.add_host_trace("RMSNorm_D_pre_mlp",   {DRAM}, {DRAM});
    
    gen.add_host_trace("RMSNorm_D_post_mlp",  {DRAM}, {DRAM});
    gen.add_host_trace("D_residual_conn2",    {DRAM}, {});

    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST); /* Terminate */

    return gen.trace_queue;
}