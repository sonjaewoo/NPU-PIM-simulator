#include "s1.h"

/* Concurrency-driven mapping */

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_llama_s1_p(TraceGenerator& gen)
{
    const std::vector<std::string> pim_linear_cmds = { "mode_change1", "bank_to_grf", "mac", "grf_to_bank", "mode_change2" };
    const std::vector<std::string> pim_rope_cmds =   { "bank_to_grf1", "bank_to_grf2", "mul", "grf_to_bank" };
    const std::vector<std::string> pim_mul_cmds =    { "bank_to_grf1", "mul", "grf_to_bank" };

    gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {DRAM,PIM});
    gen.add_host_trace("RMSNorm_D_pre_attn", {}, {DRAM,PIM});

    gen.add_trace(TraceType::COMPUTE, "", NPU); /* Start MIDAP */

    gen.add_gemv_trace("D_K12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q13", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q14", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_K15", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q15", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q16", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q17", pim_linear_cmds, PIM);    

    gen.add_gemv_trace("D_K18", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q18", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q19", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q20", pim_linear_cmds, PIM);    

    gen.add_gemv_trace("D_K21", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q21", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q22", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_Q23", pim_linear_cmds, PIM);    

    gen.add_rope_trace("D_K12_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q12_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q13_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q14_RoPE", pim_rope_cmds);

    gen.add_rope_trace("D_K15_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q15_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q16_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q17_RoPE", pim_rope_cmds);

    gen.add_rope_trace("D_K18_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q18_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q19_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q20_RoPE", pim_rope_cmds);

    gen.add_rope_trace("D_K21_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q21_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q22_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q23_RoPE", pim_rope_cmds);
    
    gen.add_gemv_trace("D_QK_T12", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T13", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T14", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T15", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T16", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T17", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T18", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T19", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T20", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T21", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T22", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_QK_T23", pim_linear_cmds, PIM);

    gen.add_host_trace("Softmax_D_H0", {DRAM}, {PIM});
    gen.add_host_trace("Softmax_D_H1", {DRAM}, {PIM});
    gen.add_host_trace("Softmax_D_H2", {DRAM}, {PIM});

    gen.add_host_trace("Softmax_D_H12", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H13", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H14", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H15", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H16", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H17", {PIM}, {DRAM});

    gen.add_host_trace("Softmax_D_H3", {DRAM}, {PIM});
    gen.add_host_trace("Softmax_D_H4", {DRAM}, {PIM});
    gen.add_host_trace("Softmax_D_H5", {DRAM}, {PIM});
    gen.add_host_trace("Softmax_D_H6", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H7", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H8", {DRAM}, {DRAM});

    gen.add_host_trace("Softmax_D_H18", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H19", {PIM}, {DRAM});

    gen.add_host_trace("Softmax_D_H9", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H10", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H11", {DRAM}, {DRAM});

    gen.add_host_trace("Softmax_D_H20", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H21", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H22", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H23", {PIM}, {DRAM});

    gen.add_gemv_trace("D_S_V0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_S_V1", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_S_V2", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_S_V3", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_S_V4", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_S_V5", pim_linear_cmds, PIM);

    gen.add_gemv_trace("D_MHA_out", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_attn", {PIM}, {DRAM});
    gen.add_host_trace("D_residual_conn1",    {DRAM}, {DRAM});
    gen.add_host_trace("RMSNorm_D_pre_mlp",   {DRAM}, {DRAM,PIM});

    gen.add_gemv_trace("D_up_proj", pim_linear_cmds, PIM);
    gen.add_rope_trace("Mul1", pim_mul_cmds);
    gen.add_gemv_trace("D_down_proj", pim_linear_cmds, PIM);

    gen.add_host_trace("RMSNorm_D_post_mlp",  {PIM}, {DRAM});
    gen.add_host_trace("D_residual_conn2",    {DRAM}, {});

    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST); /* Terminate */

    return gen.trace_queue;
}