import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 项目根目录。所有数据和模型都使用相对路径，方便复制到其他电脑运行。
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "mnist_cnn.pth"


class MNISTCNN(nn.Module):
    """用于 MNIST 手写数字识别的简单卷积神经网络。"""

    def __init__(self):
        super().__init__()
        # 第一组卷积：提取低级笔画特征，例如边缘、横竖线和弯钩。
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第二组卷积：组合更复杂的局部结构，例如闭环、斜线和交叉。
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接分类器：把卷积特征映射到 0-9 共 10 个类别。
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


def set_seed(seed: int = 42):
    """固定随机种子，让训练结果更容易复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """自动选择可用设备：优先 CUDA，其次 Apple MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(batch_size: int, limit_train: int | None, limit_test: int | None):
    """下载并构建 MNIST 训练集和测试集加载器。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # MNIST 官方常用均值和标准差，有助于模型更稳定训练。
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    # limit 参数用于快速演示或课堂验收时做小样本训练。
    if limit_train is not None:
        train_dataset = Subset(train_dataset, range(min(limit_train, len(train_dataset))))
    if limit_test is not None:
        test_dataset = Subset(test_dataset, range(min(limit_test, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """完成一轮训练，并输出平均损失。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / total_samples


def evaluate(model, loader, criterion, device):
    """在测试集上评估模型准确率和损失。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MNIST CNN 手写数字识别模型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数，默认 3")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小，默认 64")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率，默认 0.001")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--limit-train", type=int, default=None, help="仅使用部分训练样本，方便快速测试")
    parser.add_argument("--limit-test", type=int, default=None, help="仅使用部分测试样本，方便快速测试")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"当前训练设备: {device}")

    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        best_accuracy = max(best_accuracy, test_accuracy)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.4%}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_accuracy": best_accuracy,
            "class_names": [str(i) for i in range(10)],
        },
        MODEL_PATH,
    )
    print(f"训练完成，模型权重已保存到: {MODEL_PATH.relative_to(ROOT_DIR)}")
    print(f"最佳测试准确率: {best_accuracy:.4%}")


if __name__ == "__main__":
    main()
