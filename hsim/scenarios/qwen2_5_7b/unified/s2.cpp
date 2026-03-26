#include "s2.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_qwen_7b_s2_u(TraceGenerator& gen)
{
    const std::vector<std::string> pim_linear_cmds = { "mode_change1", "bank_to_grf", "mac", "grf_to_bank", "mode_change2" };
    const std::vector<std::string> pim_rope_cmds =   { "bank_to_grf1", "bank_to_grf2", "mul", "grf_to_bank" };
    const std::vector<std::string> pim_mul_cmds =    { "bank_to_grf1", "mul", "grf_to_bank" };

    gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {PIM});
    gen.add_host_trace("RMSNorm_D_pre_attn", {}, {PIM});

    gen.add_trace(TraceType::COMPUTE, "", NPU); /* Start MIDAP */

    gen.add_gemv_trace("D_K0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q1", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q2", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q3", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V0", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K4", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q4", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q5", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q6", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q7", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V4", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K8", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q8", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q9", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q10", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q11", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V8", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q13", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q14", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q15", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V12", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K16", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q16", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q17", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q18", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q19", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V16", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K20", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q20", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q21", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q22", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q23", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V20", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K24", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q24", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q25", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q26", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q27", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_V24", pim_linear_cmds, PIM);

    gen.add_host_trace("Softmax_D_H0", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H1", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H2", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H3", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H4", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H5", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H6", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H7", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H8", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H9", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H10", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H11", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H12", {PIM}, {PIM});    
    gen.add_host_trace("Softmax_D_H13", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H14", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H15", {PIM}, {PIM});    
    gen.add_host_trace("Softmax_D_H16", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H17", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H18", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H19", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H20", {PIM}, {PIM});    
    gen.add_host_trace("Softmax_D_H21", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H22", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H23", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H24", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H25", {PIM}, {PIM});
    gen.add_host_trace("Softmax_D_H26", {PIM}, {PIM}); 
    gen.add_host_trace("Softmax_D_H27", {PIM}, {PIM});
    
    gen.add_gemv_trace("D_MHA_out", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_attn", {PIM}, {PIM});
    gen.add_host_trace("D_residual_conn1",    {PIM}, {PIM});
    gen.add_host_trace("RMSNorm_D_pre_mlp",   {PIM}, {PIM});

    gen.add_gemv_trace("D_up_proj", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_gate_proj", pim_linear_cmds, PIM);
    gen.add_rope_trace("Mul1", pim_mul_cmds);
    gen.add_gemv_trace("D_down_proj", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_mlp",  {PIM}, {PIM});
    gen.add_host_trace("D_residual_conn2",    {PIM}, {});

    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST); /* Terminate */

    return gen.trace_queue;
}