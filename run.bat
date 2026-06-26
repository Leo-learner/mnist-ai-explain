@echo off
REM 进入脚本所在目录，确保所有相对路径都从项目根目录开始。
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python -m streamlit run app.py
pause
