from ..model_builder import ModelBuilder
import torch
import torch.nn as nn
import json
import os
from software.generic_op import HostProcessOp

layernorm_support = False

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1024
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 128
MAX_SEQ_LEN = 8192
ACTIVE_EXPERTS = 8
TOTAL_EXPERTS = 64

def MoeLayer(
    input_len=1,
    input_pos=128,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=NUM_ATTN_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_active_experts=ACTIVE_EXPERTS,
):
    mb = ModelBuilder(f'Moe_layer')
    input_shape = (1, hidden_size, input_len, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    rope_weight = mb.RoPEWeight(dim=head_dim, input_pos=input_pos, input_len=input_len)
    x = DecoderBlock(
        mb,
        x,
        rope_weight,
        input_pos,
        input_len,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_active_experts=num_active_experts,
        max_seq_len=max_seq_len,
    )

    return mb

def MoeLayer_pim_offload(
    input_len=1,
    input_pos=128,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=NUM_ATTN_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_active_experts=ACTIVE_EXPERTS,
):
    scenario_id=0
    target_model = ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../../../hsim/configs/system.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        workload = config.get("workload", {})
        target_model = workload.get("model_name", "")
        scenario_id = workload.get("scenario_id", 0)
        input_pos = workload.get("input_seq_len", 0)

    mb = MoeLayer(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, max_seq_len=max_seq_len, num_active_experts=num_active_experts)
    cpu_offload_name_list = ['D_residual_conn1', 'D_residual_conn2']
    
    if scenario_id == 0:
        pim_offload_name_list = []
    elif scenario_id == 1:
        pim_offload_name_list = [
            'D_K0',
            'D_up_proj_e0',
            'D_up_proj_e1',
            'D_up_proj_e2',
            'D_up_proj_e3',
            'D_up_proj_e4',
        ]
    mb = offload_layers(mb, cpu_offload_name_list, mapping='cpu')
    return offload_layers(mb, pim_offload_name_list, mapping='pim')

def offload_layers(mb: ModelBuilder, offload_name_list, mapping='cpu'):
    for name in offload_name_list:
        previous_generic_op = mb.model_dict[name].generic
        new_generic_op = HostProcessOp(
            name=name,
            input_layers=previous_generic_op.input_layers,
            output_tensor=previous_generic_op.output_tensor,
            mapping=mapping
        )
        mb.model_dict[name].generic = new_generic_op
    return mb

def LayerNorm(mb, x, shape, name_postfix='', npu_support=False):
    if npu_support:
        return mb.LayerNorm(x, shape)
    if mb.model_dict[x].generic.is_first_tensor:
        return x
    out = mb.Concat([x], name='LayerNorm'+name_postfix)     # Baseline: Offload to the host
    out_tensor = mb.model_dict[out].output
    out_tensor = nn.LayerNorm(out_tensor.shape)(out_tensor) # XXX: Temporal solution just to compile
    mb.model_dict[out].output = out_tensor
    mb.model_dict[out].generic.output_tensor = out_tensor.detach().numpy()
    return out

def RMSNorm(mb, x, shape, name_postfix='', npu_support=False):
    name = 'RMSNorm' + name_postfix
    def RMSNormOp(tensor):
        input = tensor.permute(0, 3, 2, 1)  # batch, 1, seq_len, dim
        out = nn.RMSNorm(shape)(input)
        return out.permute(0, 3, 2, 1)
    return mb.HostProcessNode(x, RMSNormOp, name=name, mapping='cpu')

def Softmax(mb, x, name_postfix=''):
    name = 'Softmax' + name_postfix
    return mb.HostProcessNode(x, nn.Softmax(1), name=name, mapping='cpu')

def RouterLinear(mb, x, input_len, hidden_size, num_experts=TOTAL_EXPERTS, block_idx=''):
    return mb.MatMul(
        [x],
        input_len,
        hidden_size,
        num_experts,
        name=f'D{block_idx}_router_linear',
    )

def Router(mb, router_logits, name_postfix='', top_k=ACTIVE_EXPERTS, num_experts=TOTAL_EXPERTS):
    name = 'Router' + name_postfix
    def RouterOp(logits):
        expert_dim = 1 if logits.shape[1] == num_experts else (3 if logits.shape[3] == num_experts else 1)
        prob = torch.softmax(logits, dim=expert_dim)
        topk_idx = torch.topk(prob, k=top_k, dim=expert_dim).indices
        mask = torch.zeros_like(prob).scatter_(expert_dim, topk_idx, 1.0)
        sparse = prob * mask
        # Timing-only simplification: return fixed shape [N,1,L,1] for direct MUL in MLP_MoE.
        if expert_dim == 1:
            return sparse.sum(dim=1, keepdim=True)
        return sparse.sum(dim=3, keepdim=True).permute(0, 3, 1, 2)
    return mb.HostProcessNode(router_logits, RouterOp, name=name, mapping='cpu')

def RouterSyncInput(mb, x, routing_weights, name_postfix=''):
    name = 'RouterSync' + name_postfix
    # Barrier node: enforce router completion before starting MLP path.
    return mb.HostProcessNode([x, routing_weights], lambda a, b: a, name=name, mapping='cpu')

def DecoderBlock(
    mb,
    input,
    rope_weight,
    input_pos,
    input_len,
    num_attention_heads=NUM_ATTN_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_active_experts=ACTIVE_EXPERTS,
    max_seq_len=MAX_SEQ_LEN,
    cache_start_pos=0,
    block_idx=''
):
    # print(f"hidden_size:{hidden_size}, int_size:{intermediate_size}")
    x = RMSNorm(mb, input, hidden_size, name_postfix=f'_D{block_idx}_pre_attn', npu_support=layernorm_support)
    x = MultiheadAttention(mb, x, rope_weight, input_pos, input_len, num_attention_heads, num_key_value_heads, head_dim, hidden_size, max_seq_len, cache_start_pos=cache_start_pos, block_idx=block_idx)
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_post_attn', npu_support=layernorm_support)
    x = mb.Sum(x, input, name=f'D{block_idx}_residual_conn1')
    y = x
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_pre_mlp', npu_support=layernorm_support)
    router_logits = RouterLinear(
        mb,
        x,
        input_len,
        hidden_size,
        num_experts=TOTAL_EXPERTS,
        block_idx=block_idx,
    )
    router_weights = Router(
        mb,
        router_logits,
        name_postfix=f'_D{block_idx}_pre_mlp',
        top_k=min(8, TOTAL_EXPERTS),
        num_experts=TOTAL_EXPERTS,
    )
    x = RouterSyncInput(mb, x, router_weights, name_postfix=f'_D{block_idx}_pre_mlp')
    expert_outputs = MLP_MoE(
        mb,
        x,
        router_weights,
        input_len,
        hidden_size,
        intermediate_size,
        block_idx=block_idx,
        num_active_experts=num_active_experts,
    )
    x = ListSum(mb, expert_outputs, block_idx=block_idx)
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_post_mlp', npu_support=layernorm_support)
    return mb.Sum(y, x, name=f'D{block_idx}_residual_conn2')

def MultiheadAttention(mb, input, rope_weight, input_pos, input_len, num_attention_heads=16, num_key_value_heads=16, head_dim=128, hidden_size=2048, max_seq_len=8192, cache_start_pos=0, block_idx=''):
    assert num_attention_heads % num_key_value_heads == 0   # GQA
    query_per_kv = num_attention_heads // num_key_value_heads
    out_w = {i: mb.Constant(tensor_shape=(1, head_dim, hidden_size, 1), name=f'D{block_idx}_H{i}_out_w') for i in range(0, num_attention_heads)}
    total_len = min(max_seq_len-cache_start_pos, input_pos+input_len-cache_start_pos)
    V_w = {i: mb.set_input_tensor(tensor_shape=(1, hidden_size, head_dim, 1), name=f'D{block_idx}_V_w{i}') for i in range(0, num_attention_heads)}
    V_T = {i: mb.Constant(tensor_shape=(1, total_len, head_dim, 1), name=f'D{block_idx}_V{i}_Cached') for i in range(0, num_key_value_heads)}
    mha_out = []
    prev_mha = None
    for i in range(0, num_attention_heads, query_per_kv):
        input_ = {k: mb.DummyNode(input, name=f'D{block_idx}_H{k}_in') for k in range(i, i+8)}
        Q = []
        for j in range(i, i + query_per_kv):
            Q.append(mb.RoPE(mb.MatMul([input_[j]], input_len, hidden_size, head_dim, name=f'D{block_idx}_Q{j}'), rope_weight, name=f'D{block_idx}_Q{j}_RoPE'))
        K = mb.MatMul([input_[i]], input_len, hidden_size, head_dim, name=f'D{block_idx}_K{i}')
        K = mb.RoPE(K, rope_weight, name=f'D{block_idx}_K{i}_RoPE')
        KCache = mb.CacheWrite(K, max_shape=(1, head_dim, max_seq_len, 1), write_offset=(0, input_pos, 0), write_shape=(head_dim, input_len, 1), name=f'D{block_idx}_K{i}_Cache')
        K = mb.CacheRead(KCache, read_offset=(0, cache_start_pos, 0), read_shape=(head_dim, total_len, 1), name=f'D{block_idx}_K{i}_Cached')
        V = mb.MatMul([input_[i]], input_len, hidden_size, head_dim, name=f'D{block_idx}_V{i}')        
        for j in range(i, i + query_per_kv):
            QK_T = mb.MatMul_binary_transpose(Q[j-i], K, input_len, head_dim, total_len, name=f'D{block_idx}_QK_T{j}')
            A = Softmax(mb, QK_T, name_postfix=f'_D{block_idx}_H{j}')
            mha_out.append(mb.MatMul_binary_transpose(A, V_T[i//2], input_len, total_len, head_dim, name=(f'D{block_idx}_S_V{j}')))
    concat = mb.Concat(mha_out, axis='c')
    out = mb.MatMul([concat], input_len, head_dim * num_attention_heads, hidden_size, name=f'D{block_idx}_MHA_out')
    return out

def MLP_MoE(
    mb,
    input,
    routing_weights,
    input_len,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    block_idx='',
    num_active_experts=ACTIVE_EXPERTS
):
    expert_outputs = []

    for expert_idx in range(num_active_experts):
        suffix = f'_e{expert_idx}'
        gate = mb.MatMul(
            [input],
            input_len,
            hidden_size,
            intermediate_size,
            activation='GELU',
            name=f'D{block_idx}_gate_proj{suffix}',
        )
        up = mb.MatMul(
            [input],
            input_len,
            hidden_size,
            intermediate_size,
            name=f'D{block_idx}_up_proj{suffix}',
        )
        mul_name = f'Mul1{suffix}' if block_idx == '' else f'D{block_idx}_Mul1{suffix}'
        expert_hidden = mb.Mul(gate, up, name=mul_name)
        down = mb.MatMul(
            [expert_hidden],
            input_len,
            intermediate_size,
            hidden_size,
            name=f'D{block_idx}_down_proj{suffix}',
        )
        weighted_down = mb.Mul(
            down,
            routing_weights,
            name=f'D{block_idx}_expert_weighted{suffix}',
        )
        expert_outputs.append(weighted_down)

    return expert_outputs

def ListSum(mb, expert_outputs, block_idx=''):
    merged = expert_outputs[0]
    for expert_idx in range(1, len(expert_outputs)):
        merge_name = f'D{block_idx}_expert_sum{expert_idx}'
        merged = mb.Sum(merged, expert_outputs[expert_idx], name=merge_name)

    return merged
