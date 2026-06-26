import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


# 所有路径都相对于当前项目，避免换电脑后路径失效。
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "mnist_cnn.pth"


class MNISTCNN(nn.Module):
    """与 train.py 保持一致的 CNN 网络结构。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_device():
    """推理阶段也自动选择可用设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _center_by_mass(array: np.ndarray) -> np.ndarray:
    """根据笔画重心把数字移动到 28x28 画布中心。"""
    total = float(array.sum())
    if total <= 0:
        return array

    y_indices, x_indices = np.indices(array.shape)
    center_y = float((y_indices * array).sum() / total)
    center_x = float((x_indices * array).sum() / total)
    shift_y = int(round(array.shape[0] / 2 - center_y))
    shift_x = int(round(array.shape[1] / 2 - center_x))

    shifted = np.zeros_like(array)
    src_y_start = max(0, -shift_y)
    src_y_end = min(array.shape[0], array.shape[0] - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(array.shape[1], array.shape[1] - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = array[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
    ]
    return shifted


def prepare_mnist_image(image_or_path):
    """
    将用户上传或指定的图片转换成 MNIST 风格的 28x28 灰度图。
    处理流程：
    1. 自动判断白底黑字或黑底白字；
    2. 裁剪有效笔画区域，去掉大面积空白；
    3. 按比例缩放到约 20x20，再居中放入 28x28 画布。
    """
    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    array = np.asarray(image).astype(np.float32) / 255.0
    if array.size == 0:
        raise ValueError("图片内容为空")

    # 用图片边缘估计背景颜色，比直接看整图平均值更适合有大面积留白的照片。
    border_pixels = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )
    background_value = float(np.median(border_pixels))

    # MNIST 的输入习惯是黑底白字，所以这里统一把“数字笔画”变成高亮前景。
    if background_value > 0.5:
        foreground = 1.0 - array
    else:
        foreground = array

    foreground = np.clip(foreground, 0.0, 1.0)
    threshold = max(0.15, float(foreground.max()) * 0.25)
    mask = foreground > threshold

    # 如果用户上传的是几乎空白的图片，直接给出明确错误，而不是输出随机结果。
    if not mask.any():
        raise ValueError("未检测到明显的数字笔画")

    ys, xs = np.where(mask)
    top = max(int(ys.min()) - 2, 0)
    bottom = min(int(ys.max()) + 3, foreground.shape[0])
    left = max(int(xs.min()) - 2, 0)
    right = min(int(xs.max()) + 3, foreground.shape[1])
    digit = foreground[top:bottom, left:right]

    digit_height, digit_width = digit.shape
    if digit_height <= 0 or digit_width <= 0:
        raise ValueError("数字区域裁剪失败")

    # MNIST 中数字通常不会占满 28x28，缩放到 20 像素左右并留出边距更稳定。
    scale = 20.0 / max(digit_width, digit_height)
    new_width = max(1, int(round(digit_width * scale)))
    new_height = max(1, int(round(digit_height * scale)))

    digit_image = Image.fromarray((digit * 255).astype(np.uint8))
    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    digit_array = np.asarray(digit_image).astype(np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    paste_top = (28 - new_height) // 2
    paste_left = (28 - new_width) // 2
    canvas[paste_top:paste_top + new_height, paste_left:paste_left + new_width] = digit_array
    canvas = _center_by_mass(canvas)

    return Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8))


def preprocess_image(image_or_path):
    """将 28x28 灰度图继续转换成模型需要的 1x1x28x28 标准化张量。"""
    image = prepare_mnist_image(image_or_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(image).unsqueeze(0)


def load_model(model_path=MODEL_PATH, device=None):
    """加载训练好的模型权重。"""
    device = device or get_device()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    model = MNISTCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_image(image_or_path, model_path=MODEL_PATH):
    """对单张图片进行推理，返回预测类别、置信度和 10 类概率。"""
    device = get_device()
    model = load_model(model_path=model_path, device=device)
    image_tensor = preprocess_image(image_or_path).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_digit = int(probabilities.argmax())
    confidence = float(probabilities[predicted_digit])
    return predicted_digit, confidence, probabilities.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="对单张手写数字图片进行推理")
    parser.add_argument("--image", required=True, help="待识别图片路径")
    parser.add_argument("--model", default=str(MODEL_PATH), help="模型权重路径")
    return parser.parse_args()


def main():
    args = parse_args()
    predicted_digit, confidence, probabilities = predict_image(args.image, args.model)
    result = {
        "预测数字": predicted_digit,
        "置信度": round(confidence, 6),
        "各类别概率": {str(i): round(probabilities[i], 6) for i in range(10)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
