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

# 初始化状态
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

# 获取当前 layer 的所有图片
image_names = sorted([f for f in os.listdir(layer_dir) if f.endswith(".png")])

# ========== 显示控制 ==========
st.markdown("#### ✅ 选择要显示的图片")
selected_images = []
cols = st.columns(4)

for i, img_name in enumerate(image_names):
    with cols[i % 4]:
        if st.checkbox(img_name, value=True, key=f"{layer_idx}_{img_name}"):
            selected_images.append(img_name)

st.markdown("---")

# ========== 图片展示 ==========
if selected_images:
    st.markdown("#### 🖼️ 显示的图片：")
    cols = st.columns(4)
    for i, img_name in enumerate(selected_images):
        path = os.path.join(layer_dir, img_name)
        if os.path.exists(path):
            with cols[i % 4]:
                st.image(Image.open(path), caption=img_name, use_container_width=True)
else:
    st.info("未选择任何图片，请勾选上方复选框显示对应图片。")

# ========== 底部说明 ==========
st.markdown("---")
st.markdown("💡 **提示**：使用左右按钮或滑块切换 Layer，勾选控制要显示的图片。")
