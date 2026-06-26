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


def prepare_mnist_image(image_or_path):
    """
    将用户上传或指定的图片转换成 MNIST 风格的 28x28 灰度图。
    常见手写图片是白底黑字，而 MNIST 是黑底白字，因此这里会自动判断并反色。
    """
    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    array = np.asarray(image).astype(np.float32) / 255.0
    # 如果背景整体偏亮，说明大概率是白底黑字，需要反色成黑底白字。
    if array.mean() > 0.5:
        array = 1.0 - array

    image = Image.fromarray((array * 255).astype(np.uint8))
    return image.resize((28, 28), Image.Resampling.LANCZOS)


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
