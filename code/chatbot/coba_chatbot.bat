@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
..\Eksperimen\eksperimen\.venv\Scripts\python.exe coba_chatbot.py
pause
