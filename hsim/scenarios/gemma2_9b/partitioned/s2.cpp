#include "s2.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_gemma_9b_s2_p(TraceGenerator& gen)
{
    const std::vector<std::string> pim_linear_cmds = { "mode_change1", "bank_to_grf", "mac", "grf_to_bank", "mode_change2" };
    const std::vector<std::string> pim_rope_cmds =   { "bank_to_grf1", "bank_to_grf2", "mul", "grf_to_bank" };
    const std::vector<std::string> pim_mul_cmds =    { "bank_to_grf1", "mul", "grf_to_bank" };

    gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {DRAM, PIM});
    gen.add_host_trace("RMSNorm_D_pre_attn", {}, {DRAM, PIM});

    gen.add_trace(TraceType::COMPUTE, "", NPU); /* Start MIDAP */

    gen.add_gemv_trace("D_K0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q1", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V2", pim_linear_cmds, PIM);    

    gen.add_gemv_trace("D_K2", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q2", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q3", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V2", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K4", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q4", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q5", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V4", pim_linear_cmds, PIM);
    
    gen.add_gemv_trace("D_K6", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q6", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q7", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V6", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K8", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q8", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q9", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V8", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K10", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q10", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q11", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V10", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q13", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V12", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K14", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q14", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q15", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V14", pim_linear_cmds, PIM);

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

    gen.add_gemv_trace("D_MHA_out", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_attn", {PIM}, {DRAM});
    gen.add_host_trace("D_residual_conn1",    {DRAM}, {DRAM});
    gen.add_host_trace("RMSNorm_D_pre_mlp",   {DRAM}, {PIM});

    gen.add_gemv_trace("D_up_proj", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_gate_proj", pim_linear_cmds, PIM);
    gen.add_rope_trace("Mul1", pim_mul_cmds);
    gen.add_gemv_trace("D_down_proj", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_mlp",  {PIM}, {DRAM});
    gen.add_host_trace("D_residual_conn2",    {DRAM}, {});

    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST); /* Terminate */

    return gen.trace_queue;
}