from ..model_builder import ModelBuilder
import torch.nn as nn
import json
import os
from software.generic_op import HostProcessOp

layernorm_support = False

HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 8192
NUM_ATTN_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 131072

def LlamaLayer(
    input_len=1,
    input_pos=128,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=NUM_ATTN_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    max_seq_len=MAX_SEQ_LEN,
):
    mb = ModelBuilder(f'Llama_layer')

    input_shape = (1, hidden_size, input_len, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    rope_weight = mb.RoPEWeight(dim=head_dim, input_pos=input_pos, input_len=input_len)
    x = DecoderBlock(mb, x, rope_weight, input_pos, input_len, num_attention_heads, num_key_value_heads, head_dim, hidden_size, intermediate_size, max_seq_len)

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
    x = MLP(mb, x, input_len, hidden_size, intermediate_size, block_idx=block_idx)
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_post_mlp', npu_support=layernorm_support)
    return mb.Sum(y, x, name=f'D{block_idx}_residual_conn2')

def MultiheadAttention(mb, input, rope_weight, input_pos, input_len, num_attention_heads=16, num_key_value_heads=16, head_dim=256, hidden_size=3072, max_seq_len=8192, cache_start_pos=0, block_idx=''):
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
            mha_out.append(mb.MatMul_binary_transpose(A, V_T[i//3], input_len, total_len, head_dim, name=(f'D{block_idx}_S_V{j}')))

    concat = mb.Concat(mha_out, axis='c')
    out = mb.MatMul([concat], input_len, head_dim * num_attention_heads, hidden_size, name=f'D{block_idx}_MHA_out')
    return out

def MLP(mb, input, input_len, hidden_size, intermediate_size, block_idx=''):
    x = mb.MatMul([input], input_len, hidden_size, intermediate_size, activation='GELU', name=f'D{block_idx}_gate_proj')
    y = mb.MatMul([input], input_len, hidden_size, intermediate_size, name=f'D{block_idx}_up_proj')
    x = mb.Mul(x, y)
    x = mb.MatMul([x], input_len, intermediate_size, hidden_size, name=f'D{block_idx}_down_proj')
    return x

def offload_layers(mb: ModelBuilder, offload_name_list, mapping='cpu'):
    for name in offload_name_list:
        # Replace the previous generic_op instance with a HostProcessOp instance
        previous_generic_op = mb.model_dict[name].generic
        new_generic_op = HostProcessOp(name=name, input_layers=previous_generic_op.input_layers, output_tensor=previous_generic_op.output_tensor, mapping=mapping)
        mb.model_dict[name].generic = new_generic_op
    return mb

def _normalize_model_name(model_name):
    return str(model_name).strip().lower().replace('-', '_')

LLAMA_PIM_OFFLOAD_MAP = {
    ('partitioned', 1): [ # Concurrency-driven
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23',

        'D_K12_RoPE', 'D_Q12_RoPE', 'D_Q13_RoPE', 'D_Q14_RoPE',
        'D_K15_RoPE', 'D_Q15_RoPE', 'D_Q16_RoPE', 'D_Q17_RoPE',
        'D_K18_RoPE', 'D_Q18_RoPE', 'D_Q19_RoPE', 'D_Q20_RoPE',
        'D_K21_RoPE', 'D_Q21_RoPE', 'D_Q22_RoPE', 'D_Q23_RoPE',

        'D_QK_T12',
        'D_QK_T13', 'D_QK_T14', 'D_QK_T15', 'D_QK_T16',
        'D_QK_T17', 'D_QK_T18', 'D_QK_T19', 'D_QK_T20',
        'D_QK_T21', 'D_QK_T22', 'D_QK_T23',
        'D_S_V0', 'D_S_V1', 'D_S_V2', 'D_S_V3', 'D_S_V4', 'D_S_V5',
        'D_MHA_out',
        'D_up_proj', 'Mul1', 'D_down_proj'
    ], 
    ('partitioned', 2): [ # Latency-driven
        'D_K0', 'D_Q0', 'D_Q1', 'D_Q2', 'D_V0',
        'D_K3', 'D_Q3', 'D_Q4', 'D_Q5', 'D_V3',
        'D_K6', 'D_Q6', 'D_Q7', 'D_Q8', 'D_V6',
        'D_K9', 'D_Q9', 'D_Q10', 'D_Q11', 'D_V9',
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14', 'D_V12',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17', 'D_V15',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20', 'D_V18',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23', 'D_V21',
        'D_MHA_out',
        'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
    ], 
    ('partitioned', 3): [ # Naive
        'D_K0', 'D_Q0', 'D_Q1', 'D_Q2', 'D_V0',
        'D_K3', 'D_Q3', 'D_Q4', 'D_Q5', 'D_V3',
        'D_K6', 'D_Q6', 'D_Q7', 'D_Q8', 'D_V6',
        'D_K9', 'D_Q9', 'D_Q10', 'D_Q11', 'D_V9',
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14', 'D_V12',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17', 'D_V15',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20', 'D_V18',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23', 'D_V21',
        
        'D_QK_T0', 'D_QK_T1', 'D_QK_T2', 'D_QK_T3',
        'D_QK_T4', 'D_QK_T5', 'D_QK_T6', 'D_QK_T7',
        'D_QK_T8', 'D_QK_T9', 'D_QK_T10', 'D_QK_T11',
        'D_QK_T12', 'D_QK_T13', 'D_QK_T14', 'D_QK_T15',
        'D_QK_T16', 'D_QK_T17', 'D_QK_T18', 'D_QK_T19',
        'D_QK_T20', 'D_QK_T21', 'D_QK_T22', 'D_QK_T23',

        'D_S_V0', 'D_S_V1', 'D_S_V2', 'D_S_V3',
        'D_S_V4', 'D_S_V5', 'D_S_V6', 'D_S_V7',
        'D_S_V8', 'D_S_V9', 'D_S_V10', 'D_S_V11',
        'D_S_V12', 'D_S_V13', 'D_S_V14', 'D_S_V15',
        'D_S_V16', 'D_S_V17', 'D_S_V18', 'D_S_V19',
        'D_S_V20', 'D_S_V21', 'D_S_V22', 'D_S_V23',

        'D_MHA_out',
        'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
    ],
    ('unified', 1): [ # Concurrency-driven
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23',

        'D_K12_RoPE', 'D_Q12_RoPE', 'D_Q13_RoPE', 'D_Q14_RoPE',
        'D_K15_RoPE', 'D_Q15_RoPE', 'D_Q16_RoPE', 'D_Q17_RoPE',
        'D_K18_RoPE', 'D_Q18_RoPE', 'D_Q19_RoPE', 'D_Q20_RoPE',
        'D_K21_RoPE', 'D_Q21_RoPE', 'D_Q22_RoPE', 'D_Q23_RoPE',

        'D_QK_T12',
        'D_QK_T13', 'D_QK_T14', 'D_QK_T15', 'D_QK_T16',
        'D_QK_T17', 'D_QK_T18', 'D_QK_T19', 'D_QK_T20',
        'D_QK_T21', 'D_QK_T22', 'D_QK_T23',
        'D_S_V0', 'D_S_V1', 'D_S_V2', 'D_S_V3', 'D_S_V4', 'D_S_V5',
        'D_MHA_out',
        'D_up_proj', 'Mul1', 'D_down_proj'
    ], 
    ('unified', 2): [ # Latency-driven
        'D_K0', 'D_Q0', 'D_Q1', 'D_Q2', 'D_V0',
        'D_K3', 'D_Q3', 'D_Q4', 'D_Q5', 'D_V3',
        'D_K6', 'D_Q6', 'D_Q7', 'D_Q8', 'D_V6',
        'D_K9', 'D_Q9', 'D_Q10', 'D_Q11', 'D_V9',
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14', 'D_V12',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17', 'D_V15',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20', 'D_V18',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23', 'D_V21',
        'D_MHA_out',
        'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
    ], 
    ('unified', 3): [ # Naive
        'D_K0', 'D_Q0', 'D_Q1', 'D_Q2', 'D_V0',
        'D_K3', 'D_Q3', 'D_Q4', 'D_Q5', 'D_V3',
        'D_K6', 'D_Q6', 'D_Q7', 'D_Q8', 'D_V6',
        'D_K9', 'D_Q9', 'D_Q10', 'D_Q11', 'D_V9',
        'D_K12', 'D_Q12', 'D_Q13', 'D_Q14', 'D_V12',
        'D_K15', 'D_Q15', 'D_Q16', 'D_Q17', 'D_V15',
        'D_K18', 'D_Q18', 'D_Q19', 'D_Q20', 'D_V18',
        'D_K21', 'D_Q21', 'D_Q22', 'D_Q23', 'D_V21',
        
        'D_QK_T0', 'D_QK_T1', 'D_QK_T2', 'D_QK_T3',
        'D_QK_T4', 'D_QK_T5', 'D_QK_T6', 'D_QK_T7',
        'D_QK_T8', 'D_QK_T9', 'D_QK_T10', 'D_QK_T11',
        'D_QK_T12', 'D_QK_T13', 'D_QK_T14', 'D_QK_T15',
        'D_QK_T16', 'D_QK_T17', 'D_QK_T18', 'D_QK_T19',
        'D_QK_T20', 'D_QK_T21', 'D_QK_T22', 'D_QK_T23',

        'D_S_V0', 'D_S_V1', 'D_S_V2', 'D_S_V3',
        'D_S_V4', 'D_S_V5', 'D_S_V6', 'D_S_V7',
        'D_S_V8', 'D_S_V9', 'D_S_V10', 'D_S_V11',
        'D_S_V12', 'D_S_V13', 'D_S_V14', 'D_S_V15',
        'D_S_V16', 'D_S_V17', 'D_S_V18', 'D_S_V19',
        'D_S_V20', 'D_S_V21', 'D_S_V22', 'D_S_V23',

        'D_MHA_out',
        'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
    ],    
}

def _resolve_pim_offload_list(memory_structure, scenario_id):
    memory = str(memory_structure)
    scenario = int(scenario_id)

    if memory == 'baseline':
        return []

    key = (memory, scenario)
    if key in LLAMA_PIM_OFFLOAD_MAP:
        return LLAMA_PIM_OFFLOAD_MAP[key]

    raise ValueError(
        f"Unsupported offload config: memory_structure={memory}, scenario_id={scenario}"
    )

def LlamaLayer_pim_offload(
    input_len=1,
    input_pos=128,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=NUM_ATTN_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    max_seq_len=MAX_SEQ_LEN,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../../../hsim/configs/system.json")

    with open(config_path, "r") as f:
        config = json.load(f)
        architecture = config.get("architecture", {})
        memory_structure = architecture.get("memory_structure", "")
        workload = config.get("workload", {})
        scenario_id = workload.get("scenario_id", 0)
        input_pos = workload.get("input_seq_len", 0)

    mb = LlamaLayer(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, max_seq_len=max_seq_len)
    cpu_offload_name_list = ['D_residual_conn1', 'D_residual_conn2']
    
    pim_offload_name_list = _resolve_pim_offload_list(
        memory_structure=memory_structure,
        scenario_id=scenario_id,
    )

    mb = offload_layers(mb, cpu_offload_name_list, mapping='cpu')
    
    return offload_layers(mb, pim_offload_name_list, mapping='pim')
