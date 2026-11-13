@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist hasil mkdir hasil
..\Eksperimen\eksperimen\.venv\Scripts\python.exe -u banding_gemini.py > hasil\log_banding_gemini.txt 2>&1
