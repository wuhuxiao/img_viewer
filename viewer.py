import os
import streamlit as st
from PIL import Image

# ========== 基础配置 ==========
st.set_page_config(page_title="Layer 图像对比", layout="wide")

base_dir = './imgs'
layers = sorted(
    [d for d in os.listdir(base_dir) if d.startswith("layer_")],
    key=lambda x: int(x.split("_")[1])  # 提取数字部分进行排序
)

dir_path = base_dir + "layer_0"
image_names = [f for f in os.listdir(dir_path) if f.endswith(".png")]

print(image_names)

# ========== 状态保持（避免每次刷新重置） ==========
if "layer_idx" not in st.session_state:
    st.session_state.layer_idx = 0

# ========== 顶部控制栏 ==========
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("⬅️ 上一层", use_container_width=True):
        st.session_state.layer_idx = max(0, st.session_state.layer_idx - 1)

with col2:
    st.session_state.layer_idx = st.slider(
        "选择 Layer",
        0,
        len(layers) - 1,
        st.session_state.layer_idx,
        step=1,
        label_visibility="collapsed",
        help="拖动或点击按钮切换 Layer"
    )

with col3:
    if st.button("下一层 ➡️", use_container_width=True):
        st.session_state.layer_idx = min(len(layers) - 1, st.session_state.layer_idx + 1)

# 当前层
layer_idx = st.session_state.layer_idx
layer_dir = os.path.join(base_dir, layers[layer_idx])
st.markdown(f"### 🔹 当前 Layer：{layer_idx}  —  `{layers[layer_idx]}`")

# ========== 图片展示区域 ==========
cols = st.columns(3)
for i, img_name in enumerate(image_names):
    path = os.path.join(layer_dir, img_name)
    if os.path.exists(path):
        with cols[i % 3]:
            st.image(Image.open(path), caption=img_name, use_container_width=True)

# ========== 底部说明 ==========
st.markdown("---")
st.markdown("💡 **提示**：使用左右按钮或滑块快速浏览不同 Layer。")

