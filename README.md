# 基于卷积神经网络与大模型解释的手写数字识别系统

本项目用于《人工智能基础B》期末大作业，使用 Python、PyTorch、MNIST、Streamlit 和大模型解释模块实现一个完整的手写数字识别系统。

## 1. 项目功能

- 使用 PyTorch 构建 CNN 手写数字识别模型。
- 使用 MNIST 数据集训练并保存模型权重到 `models/mnist_cnn.pth`。
- 支持命令行单张图片推理。
- 使用 Streamlit 提供可视化界面，支持图片上传、预测结果展示、置信度展示、概率柱状图展示。
- 支持大模型解释：配置 `API_KEY` 时调用在线大模型；未配置时自动使用本地中文解释模板，保证离线可运行。

## 2. 项目结构

```text
mnist_ai_explain/
├── data/
├── models/
├── llm/
│   └── explainer.py
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── tests.md
├── run.bat
├── run.ps1
├── run.sh
└── README.md
```

## 3. 环境安装

建议使用 Python 3.10、3.11 或 3.12。

### macOS / Linux

```bash
cd ~/Desktop/mnist_ai_explain
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

PowerShell 写法：

```powershell
cd "$env:USERPROFILE\Desktop\mnist_ai_explain"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 提示不允许运行脚本，可先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

CMD 写法：

```bat
cd %USERPROFILE%\Desktop\mnist_ai_explain
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. 训练模型

完整训练：

```bash
python train.py --epochs 3
```

快速测试训练流程：

```bash
python train.py --epochs 1 --limit-train 5000 --limit-test 1000
```

训练完成后会生成：

```text
models/mnist_cnn.pth
```

## 5. 单张图片推理

```bash
python predict.py --image path/to/your_digit.png
```

输出示例：

```json
{
  "预测数字": 7,
  "置信度": 0.987654,
  "各类别概率": {
    "0": 0.000001,
    "1": 0.000123,
    "2": 0.000456
  }
}
```

## 6. 启动可视化系统

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

### Windows

双击 `run.bat`，或在 CMD 中运行：

```bat
run.bat
```

如果使用 PowerShell，请运行：

```powershell
.\run.ps1
```

或：

```powershell
.\run.bat
```

也可以直接运行：

```bash
streamlit run app.py
```

启动后浏览器会打开 Streamlit 页面，上传手写数字图片即可查看预测结果。

页面包含：

- 项目简介、功能流程说明和团队信息区域。
- 原始上传图片和预处理后的 28×28 灰度图。
- 预测数字、最高置信度、10 个数字类别概率柱状图。
- 大模型或本地模板生成的四段式解释分析。

## 7. 大模型解释配置

如果不配置 API Key，系统会自动使用本地模板解释，适合离线演示。

如需调用在线大模型，可设置环境变量：

### macOS / Linux

```bash
export API_KEY="你的API_KEY"
export LLM_MODEL="gpt-4o-mini"
export LLM_API_BASE="https://api.openai.com/v1"
```

### Windows

```bat
set API_KEY=你的API_KEY
set LLM_MODEL=gpt-4o-mini
set LLM_API_BASE=https://api.openai.com/v1
```

说明：`LLM_API_BASE` 使用 OpenAI 兼容接口格式，默认会请求 `/chat/completions`。

## 8. 测试与分析建议

详细测试表格见 `tests.md`，可直接用于课程报告的测试分析部分。

课程报告中可以记录以下内容：

- CNN 网络结构：两层卷积、两层池化、两层全连接。
- 训练参数：训练轮数、批大小、学习率、优化器。
- 测试集准确率：训练结束时控制台会输出 `Test Accuracy`。
- 错误分析：上传书写模糊、倾斜、笔画断裂的图片，观察置信度和概率分布变化。
- 大模型解释：对比配置 API_KEY 和未配置 API_KEY 两种模式下的解释效果。

## 9. 注意事项

- 第一次训练会自动下载 MNIST 数据集到 `data/` 目录。
- 运行 Streamlit 前请先训练模型，确保 `models/mnist_cnn.pth` 存在。
- 项目代码使用相对路径，复制整个 `mnist_ai_explain` 文件夹到其他电脑后仍可运行。
