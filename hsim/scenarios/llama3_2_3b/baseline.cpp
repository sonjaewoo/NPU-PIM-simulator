#include "baseline.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_llama_baseline(TraceGenerator& gen)
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
    gen.add_host_trace("Softmax_D_H8", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H9", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H10", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H11", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H12", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H13", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H14", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H15", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H16", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H17", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H18", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H19", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H20", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H21", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H22", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H23", {DRAM}, {DRAM});

    gen.add_host_trace("RMSNorm_D_post_attn", {DRAM}, {DRAM});
    gen.add_host_trace("D_residual_conn1",    {DRAM}, {DRAM});
    gen.add_host_trace("RMSNorm_D_pre_mlp",   {DRAM}, {DRAM});
    
    gen.add_host_trace("RMSNorm_D_post_mlp",  {DRAM}, {DRAM});
    gen.add_host_trace("D_residual_conn2",    {DRAM}, {});

    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST); /* Terminate */

    return gen.trace_queue;
}