import torch
import pickle
import os
import vllm
import sys
# 添加 so 文件所在目录
sys.path.append("/home/whx/vllm-workspace/whx/lib/python3.12/site-packages/vllm/vllm_flash_attn")

# 然后尝试导入
import torch
_vllm_fa3_C = torch.ops.load_library("/home/whx/vllm-workspace/whx/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so")  # 假设 so 名字是这个

save_dir = "/home/whx/vllm-workspace/dumped_tensors/fa/"

# 加载保存的输入
with open(os.path.join(save_dir, "fa3_inputs.pkl"), "rb") as f:
    data = pickle.load(f)

# 调用 CUDA 算子
out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
    data["q"],
    data["k"],
    data["v"],
    data["k_new"],
    data["v_new"],
    data["q_v"],
    data["out"],
    data["cu_seqlens_q"],
    data["cu_seqlens_k"],
    data["cu_seqlens_k_new"],
    data["seqused_q"],
    data["seqused_k"],
    data["max_seqlen_q"],
    data["max_seqlen_k"],
    data["block_table"],
    data["kv_batch_idx"],
    data["leftpad_k"],
    data["rotary_cos"],
    data["rotary_sin"],
    data["seqlens_rotary"],
    data["q_descale"],
    data["k_descale"],
    data["v_descale"],
    data["softmax_scale"],
    data["causal"],
    data["real_window_size"][0],
    data["real_window_size"][1],
    data["softcap"],
    data["rotary_interleaved"],
    data["scheduler_metadata"],
    data["num_splits"],
    data["pack_gqa"],
    data["sm_margin"],
)

import torch
import torch.nn.functional as F

def fa3_fwd_torch(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    causal=False,
    softmax_scale=None
):
    """
    PyTorch FA3 forward (supports GQA-style Q/K head mismatch)

    Args:
        q, k, v: [seq_len, num_heads, head_dim] (Q.num_heads >= K.num_heads)
        cu_seqlens_q, cu_seqlens_k: cumulative sequence lengths for batch segmentation
        causal: apply causal mask
        softmax_scale: optional scaling factor

    Returns:
        out: [seq_len_q, num_heads_q, head_dim]
        softmax_lse: [seq_len_q, num_heads_q] logsumexp of attention
    """
    device = q.device
    out = torch.zeros_like(q)
    softmax_lse = torch.zeros(q.shape[0], q.shape[1], device=device)

    batch_size = len(cu_seqlens_q) - 1

    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()

        q_b = q[q_start:q_end]   # [Lq, Hq, D]
        k_b = k[k_start:k_end]   # [Lk, Hk, D]
        v_b = v[k_start:k_end]   # [Lk, Hk, D]

        Hq, Hk = q_b.shape[1], k_b.shape[1]
        group_size = Hq // Hk
        assert Hq % Hk == 0, "Q heads must be divisible by K heads"

        # reshape Q for GQA
        q_grouped = q_b.reshape(q_b.shape[0], Hk, group_size, q_b.shape[2])  # [Lq, Hk, group, D]
        q_grouped = q_grouped.permute(1,0,2,3)  # [Hk, Lq, group, D]

        k_b = k_b.permute(1,0,2)  # [Hk, Lk, D]
        v_b = v_b.permute(1,0,2)  # [Hk, Lk, D]

        out_heads = []

        for h in range(Hk):
            qh = q_grouped[h]  # [Lq, group, D]
            kh = k_b[h]        # [Lk, D]
            vh = v_b[h]        # [Lk, D]

            # flatten group dimension
            qh_flat = qh.reshape(-1, qh.shape[-1])  # [Lq*group, D]

            # QK^T
            attn_scores = torch.matmul(qh_flat, kh.T)  # [Lq*group, Lk]

            if softmax_scale is not None:
                attn_scores = attn_scores * softmax_scale

            # causal mask
            if causal:
                Lq, group = qh.shape[0], qh.shape[1]
                Lk = kh.shape[0]
                mask = torch.zeros((Lq, Lk), dtype=torch.bool, device=device)
                mask[q_start:q_end, Lk-Lq:] = torch.triu(torch.ones(Lq, Lq, device=device), diagonal=1)
                mask = mask.bool().repeat(group, 1)
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            # softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            softmax_lse[q_start:q_end, h*group_size:(h+1)*group_size] = torch.logsumexp(attn_scores, dim=-1).reshape(Lq, group)

            # output
            out_flat = torch.matmul(attn_probs, vh)  # [Lq*group, D]
            out_heads.append(out_flat.reshape(qh.shape[0], qh.shape[1], -1))

        # concat head dimension
        out[q_start:q_end] = torch.cat(out_heads, dim=1)

    return out, softmax_lse

torch_out, _ = fa3_fwd_torch(
    q=data["q"],
    k=data["k"],
    v=data["v"],
    cu_seqlens_q=data["cu_seqlens_q"],
    cu_seqlens_k=data["cu_seqlens_k"],
    causal=data["causal"],
    softmax_scale=data["softmax_scale"]
)

# print("out.shape =", out.shape)

gloden_out = torch.load(os.path.join(save_dir, "golden_output.pkl"))
print("✅ Replay success, output:", gloden_out.shape)

