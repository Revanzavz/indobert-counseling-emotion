@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe analisis_gemini.py hasil\banding_gemini_konseling_lengkap.csv hasil
pause
