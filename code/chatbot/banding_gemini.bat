@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
..\Eksperimen\eksperimen\.venv\Scripts\python.exe banding_gemini.py %*
pause
