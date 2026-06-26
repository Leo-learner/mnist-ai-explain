# 进入脚本所在目录，确保所有相对路径都从项目根目录开始。
Set-Location -Path $PSScriptRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . ".venv\Scripts\Activate.ps1"
}

python -m streamlit run app.py
