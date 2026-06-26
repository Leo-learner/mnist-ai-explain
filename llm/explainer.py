import os
from pathlib import Path
from typing import Sequence

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_local_env():
    """读取本地环境变量文件，避免为了 API Key 额外引入 python-dotenv 依赖。"""
    for env_path in (ROOT_DIR / ".env.local", ROOT_DIR / ".env"):
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()


def _format_probabilities(probabilities: Sequence[float]) -> str:
    """把 10 类概率格式化成便于大模型阅读的文本。"""
    return "\n".join(
        f"数字 {index}: {probability * 100:.2f}%"
        for index, probability in enumerate(probabilities)
    )


def _top_classes(probabilities: Sequence[float], top_k: int = 3):
    """返回概率最高的若干类别，用于本地解释模板。"""
    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


def _local_explanation(predicted_digit: int, confidence: float, probabilities: Sequence[float]) -> str:
    """未配置 API_KEY 或在线调用失败时使用的离线解释。"""
    top_classes = _top_classes(probabilities)
    top_text = "、".join(
        f"数字 {digit}（{probability * 100:.2f}%）"
        for digit, probability in top_classes
    )

    if confidence >= 0.90:
        certainty_text = "模型对该结果非常有把握。"
    elif confidence >= 0.70:
        certainty_text = "模型对该结果较有把握，但仍建议结合图片质量判断。"
    else:
        certainty_text = "模型置信度偏低，图片可能存在模糊、倾斜或数字形态不典型的问题。"

    return (
        f"**识别结果说明：** 系统预测该图片中的数字是 {predicted_digit}。"
        "卷积神经网络会先提取笔画边缘、曲线、闭环等局部特征，"
        "再通过全连接层综合判断最可能的数字类别。\n\n"
        f"**置信度分析：** 最高置信度为 {confidence * 100:.2f}%。"
        f"{certainty_text}从概率分布看，排名靠前的类别为：{top_text}。\n\n"
        "**可能误差原因：** 如果多个类别概率接近，通常说明这些数字在书写形态上有相似之处，"
        "例如 3 与 8、1 与 7、4 与 9 等。图片模糊、背景噪声、数字倾斜、笔画过细或过粗，"
        "也可能导致模型判断不稳定。\n\n"
        "**应用价值说明：** 该系统可以用于课堂演示深度学习图像分类流程，"
        "也可以作为票据识别、表单数字录入、教育练习批改等应用的基础原型。"
    )


def _build_prompt(predicted_digit: int, confidence: float, probabilities: Sequence[float]) -> str:
    """构造发送给大模型的提示词。"""
    return (
        "你是人工智能课程项目中的模型解释助手。请用中文简洁解释一个 MNIST 手写数字识别结果，"
        "必须按以下四个小标题输出：识别结果说明、置信度分析、可能误差原因、应用价值说明。"
        "说明预测数字、置信度、概率分布含义，以及可能影响识别的图片因素。"
        "不要编造图片中不存在的细节。\n\n"
        f"预测数字: {predicted_digit}\n"
        f"置信度: {confidence * 100:.2f}%\n"
        f"各类别概率:\n{_format_probabilities(probabilities)}"
    )


def _resolve_endpoint() -> str:
    """
    默认使用 OpenAI 兼容的 Chat Completions 接口。
    如需使用其他兼容平台，可设置 LLM_API_BASE，例如：https://api.example.com/v1
    """
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    if api_base.endswith("/chat/completions"):
        return api_base
    return f"{api_base}/chat/completions"


def _call_llm_api(predicted_digit: int, confidence: float, probabilities: Sequence[float]) -> str:
    """调用在线大模型 API 生成解释。"""
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 API_KEY 或 OPENAI_API_KEY")

    endpoint = _resolve_endpoint()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(predicted_digit, confidence, probabilities)

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("LLM_HTTP_REFERER", ""),
            "X-Title": os.getenv("LLM_APP_TITLE", "mnist-ai-explain"),
        },
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你擅长用通俗中文解释深度学习分类结果。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def explain_prediction(predicted_digit: int, confidence: float, probabilities: Sequence[float]) -> str:
    """
    对外暴露的解释函数。
    - 配置 API_KEY 或 OPENAI_API_KEY：优先调用在线大模型。
    - 未配置或调用失败：自动回退到本地模板，保证离线也能运行。
    """
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _local_explanation(predicted_digit, confidence, probabilities)

    try:
        return _call_llm_api(predicted_digit, confidence, probabilities)
    except Exception as exc:
        fallback = _local_explanation(predicted_digit, confidence, probabilities)
        return (
            f"{fallback}\n\n"
            f"提示：在线大模型解释调用失败，系统已自动切换到本地解释模板。错误信息：{exc}"
        )
