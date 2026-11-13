@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist hasil mkdir hasil
..\Eksperimen\eksperimen\.venv\Scripts\python.exe uji_privasi.py
pause
