from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from llm.explainer import explain_prediction
from predict import MODEL_PATH, load_model, prepare_mnist_image, preprocess_image


ROOT_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="手写数字识别系统",
    layout="centered",
)


@st.cache_resource
def get_cached_model():
    """Streamlit 缓存模型，避免每次交互都重新加载权重。"""
    return load_model(MODEL_PATH)


def run_prediction(image: Image.Image):
    """执行图片预处理、模型推理和概率计算。"""
    model = get_cached_model()
    device = next(model.parameters()).device
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_digit = int(probabilities.argmax())
    confidence = float(probabilities[predicted_digit])
    return predicted_digit, confidence, probabilities.tolist()


with st.sidebar:
    st.header("团队信息")
    st.write("课程名称：人工智能基础B")
    st.write("项目名称：基于卷积神经网络与大模型解释的手写数字识别系统")
    st.write("小组成员：邓凯中202536020314 张智成202536020240 肖熠202536020237")
    st.write("指导目标：深度学习建模、可视化交互、大模型解释、测试分析")

st.title("基于卷积神经网络与大模型解释的手写数字识别系统")
st.caption("课程项目：人工智能基础B | CNN + MNIST + Streamlit + 大模型解释")

with st.expander("项目简介", expanded=True):
    st.markdown(
        """
        本系统面向手写数字识别场景，使用 PyTorch 构建卷积神经网络模型，
        基于 MNIST 数据集训练 0 到 9 的数字分类器。系统通过 Streamlit
        提供可视化界面，并结合大模型或本地模板生成识别结果解释，便于课程展示和实验报告撰写。
        """
    )

with st.expander("功能流程说明", expanded=True):
    st.markdown(
        """
        1. 用户上传一张包含单个手写数字的图片。
        2. 系统将图片转为灰度图，并缩放为 MNIST 模型需要的 28×28 输入。
        3. CNN 模型输出 10 个数字类别的概率分布。
        4. 页面展示预测数字、最高置信度、概率柱状图和详细概率表。
        5. 解释模块根据预测结果生成识别说明、置信度分析、误差原因和应用价值说明。
        """
    )

if not MODEL_PATH.exists():
    st.warning(
        "未发现模型权重文件 models/mnist_cnn.pth。请先在项目目录运行："
        "`python train.py --epochs 3`"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "上传手写数字图片",
    type=["png", "jpg", "jpeg", "bmp"],
    help="建议上传白底黑字或黑底白字、只包含单个数字的图片。",
)

if uploaded_file is None:
    st.info("请先上传一张包含单个手写数字的图片，然后系统会自动完成识别和解释。")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("L")
except (UnidentifiedImageError, OSError):
    st.error("图片格式无法识别。请上传有效的 PNG、JPG、JPEG 或 BMP 图片文件。")
    st.stop()

try:
    preprocessed_image = prepare_mnist_image(image)
except Exception as exc:
    st.error(f"图片预处理失败，请更换更清晰的手写数字图片。错误信息：{exc}")
    st.stop()

image_col, processed_col = st.columns([1, 1])
with image_col:
    st.subheader("原始图片")
    st.image(image, caption="用户上传的原图", use_container_width=True)
with processed_col:
    st.subheader("预处理结果")
    st.image(
        preprocessed_image.resize((180, 180), Image.Resampling.NEAREST),
        caption="28×28 灰度图，页面中已放大显示",
        use_container_width=False,
    )

predicted_digit, confidence, probabilities = run_prediction(image)

left_col, right_col = st.columns([1, 1])
with left_col:
    st.subheader("预测数字")
    st.metric("模型判断结果", str(predicted_digit))

with right_col:
    st.subheader("最高置信度")
    st.metric("Top-1 概率", f"{confidence * 100:.2f}%")

st.subheader("10 个数字类别的概率分布")
probability_df = pd.DataFrame(
    {
        "数字类别": [str(i) for i in range(10)],
        "概率": probabilities,
    }
)
st.bar_chart(probability_df.set_index("数字类别"))

st.subheader("大模型/本地解释分析")
explanation = explain_prediction(
    predicted_digit=predicted_digit,
    confidence=confidence,
    probabilities=probabilities,
)
st.markdown(explanation)

with st.expander("查看详细概率"):
    display_df = probability_df.copy()
    display_df["概率"] = display_df["概率"].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(display_df, use_container_width=True)
