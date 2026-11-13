@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if not exist hasil mkdir hasil
..\Eksperimen\eksperimen\.venv\Scripts\python.exe uji_krisis.py
pause
