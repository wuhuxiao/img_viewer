import os
import streamlit as st
from PIL import Image

# ========== 基础配置 ==========
st.set_page_config(page_title="Layer 图像对比（同图跨层）", layout="wide")

base_dir = './imgs'
layers = sorted(
    [d for d in os.listdir(base_dir) if d.startswith("layer_")],
    key=lambda x: int(x.split("_")[1])
)

image_names = [
    "no rope and baseline_Mean.png",
    "re rope and baseline_Mean.png",
    "re rope and no rope_Mean.png",
    "no rope and baseline_Std.png",
    "re rope and baseline_Std.png",
    "re rope and no rope_Std.png"
]

# ========== 选择图像类型 ==========
st.title("📊 不同 Layer 的同图像对比查看")
selected_image = st.selectbox(
    "选择要查看的图像类型：",
    image_names,
    index=0,
    help="选择一张图片类型以查看其在不同 Layer 的变化情况"
)

# ========== 滑块控制 ==========
col1, col2 = st.columns([3, 1])
with col1:
    max_layers_to_show = st.slider(
        "选择显示的 Layer 数量：",
        min_value=1,
        max_value=len(layers),
        value=min(8, len(layers)),
        step=1
    )
with col2:
    step = st.number_input("步长（每隔几层显示）", min_value=1, max_value=5, value=1, step=1)

# ========== 图片展示区域 ==========
st.markdown(f"### 🔹 对比图：`{selected_image}`")

# 每行显示4张
cols_per_row = 3
shown = 0
selected_layers = layers[::step][:max_layers_to_show]

# 按4个一组分组
for i in range(0, len(selected_layers), cols_per_row):
    row_layers = selected_layers[i:i + cols_per_row]
    cols = st.columns(len(row_layers))
    for j, layer in enumerate(row_layers):
        img_path = os.path.join(base_dir, layer, selected_image)
        if os.path.exists(img_path):
            with cols[j]:
                st.image(Image.open(img_path), caption=f"{layer}", use_container_width=True)
        else:
            with cols[j]:
                st.warning(f"{layer} 无此图片")

# ========== 底部说明 ==========
st.markdown("---")
st.markdown("💡 **提示**：通过滑块控制显示层数，用步长调节间隔，比如只看每隔2层的变化。")
