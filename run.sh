#!/usr/bin/env bash
set -e

# 进入脚本所在目录，确保所有相对路径都从项目根目录开始。
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  source ".venv/bin/activate"
fi

python3 -m streamlit run app.py
