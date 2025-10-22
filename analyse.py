import os
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp

TENSOR_DIR = "/home/whx/vllm-workspace/dumped_tensors"
IMG_DIR = "/home/whx/vllm-workspace/dumped_tensors/img_viewer/imgs"


def get_kv_tensor_from_dirs(layer_id, dirs):
    """读取多个目录下相同layer的kv张量"""
    for dir in dirs:
        path = os.path.join(dir, f"layer_{layer_id}_kv.pt")
        yield torch.load(path).cpu()


def draw_tensors(tensor, layer_id, title, prop):
    """绘制tensor热图并保存"""
    assert len(tensor.shape) == 2
    plt.figure(figsize=(8, 6))
    plt.imshow(tensor.float().cpu().numpy(), cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Absolute Error')
    plt.title(f'{title} Difference Heatmap: {prop}')
    plt.xlabel('num_tokens')
    plt.ylabel('num_blks')

    save_dir = f"{IMG_DIR}/layer_{layer_id}/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{title}_{prop}.png'), dpi=300, bbox_inches='tight')
    plt.close()  # ✅ 防止内存泄漏


def describe_tensors_per_blk(tensor, layer_id, title, num_blks, num_tokens):
    """绘制每个块的统计指标"""
    tensor_reshaped = tensor.reshape(num_blks, num_tokens, -1)
    draw_tensors(tensor_reshaped.mean(-1), layer_id, title, "Mean")
    draw_tensors(tensor_reshaped.std(-1), layer_id, title, "Std")
    # 可以按需添加更多统计图：
    # draw_tensors(tensor_reshaped.sum(-1), layer_id, title, "Sum")
    # draw_tensors(tensor_reshaped.max(-1)[0], layer_id, title, "Max")


def process_layer(layer_id):
    """单层的处理逻辑（可并行）"""
    try:
        dirs = [f"{TENSOR_DIR}/base_line/", f"{TENSOR_DIR}/re_rope/", f"{TENSOR_DIR}/blend/"]
        tensors = list(get_kv_tensor_from_dirs(layer_id, dirs))
        base_line_kv, re_rope_kv, blend_kv = tensors

        diff_rope = (re_rope_kv - base_line_kv).abs()
        diff_blend = (blend_kv - base_line_kv).abs()
        diff_blend_rope = (blend_kv - re_rope_kv).abs()

        num_kv, num_blks, num_tokens, num_head, head_size = base_line_kv.shape

        describe_tensors_per_blk(diff_rope[0], layer_id, "rope_vs_baseline", num_blks, num_tokens)
        describe_tensors_per_blk(diff_blend[0], layer_id, "blend_vs_baseline", num_blks, num_tokens)
        describe_tensors_per_blk(diff_blend_rope[0], layer_id, "blend_vs_rope", num_blks, num_tokens)

        print(f"✅ Layer {layer_id} finished")
    except Exception as e:
        print(f"❌ Layer {layer_id} failed: {e}")


if __name__ == "__main__":
    # 自动检测可用CPU核数
    num_processes = min(16, mp.cpu_count())  # ✅ 限制最多8个进程（防止内存打爆）
    layer_ids = list(range(48))

    print(f"🚀 Using {num_processes} processes to process {len(layer_ids)} layers...")

    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_layer, layer_ids)

    print("🎉 All layers processed successfully!")
