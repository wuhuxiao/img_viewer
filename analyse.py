import matplotlib.pyplot as plt
import os


import torch

def get_kv_tensor_from_dirs(layer_id, dirs):
    for dir in dirs:
        path = dir + f"layer_{layer_id}_kv.pt"
        yield torch.load(path).cpu()

def draw_tensors(tensor, title, prop):
    assert len(tensor.shape) == 2
    plt.figure(figsize=(8, 6))
    plt.imshow(tensor.float().cpu().numpy(), cmap='coolwarm', aspect='auto')
    plt.colorbar(label=f' Absolute Error')
    plt.title(f'{title} Difference Heatmap: {prop}')
    plt.xlabel('num_tokens')  # 横轴标签
    plt.ylabel('num_blks')  # 纵轴标签
    save_dir = f"imgs/layer_{layer_id}/"
    os.makedirs(save_dir, exist_ok=True)  # 若不存在则自动创建
    plt.savefig(os.path.join(save_dir, f'{title}_{prop}.png'), dpi=300, bbox_inches='tight')  # 高分辨率保存
    # plt.show()

def describe_tensors_per_blk(tensor, title):
    draw_tensors(tensor.reshape(num_blks,num_tokens, -1).mean(-1), title, "Mean")
    draw_tensors(tensor.reshape(num_blks,num_tokens, -1).std(-1), title, "Std")
    # draw_tensors(tensor.reshape(num_blks,num_tokens, -1).sum(-1), title, "Sum")
    # draw_tensors(tensor.reshape(num_blks,num_tokens, -1).max(-1)[0], title, "Max")

for layer_id in range(48):
    tensors = get_kv_tensor_from_dirs(layer_id, ["base_line/","no_rope/","re_rope/"])
    base_line_kv, no_rope_kv, re_rope_kv = list(tensors)
    diff_no_rope = (no_rope_kv - base_line_kv).abs()
    diff_re_rope = (re_rope_kv - base_line_kv).abs()
    diff_rope = (re_rope_kv - no_rope_kv).abs()


    num_kv , num_blks, num_tokens, num_head, head_size = base_line_kv.shape

    # diff_no_rope
    describe_tensors_per_blk(diff_no_rope[0], "no rope and baseline")
    describe_tensors_per_blk(diff_re_rope[0], "re rope and baseline")
    describe_tensors_per_blk(diff_rope[0], "re rope and no rope")
