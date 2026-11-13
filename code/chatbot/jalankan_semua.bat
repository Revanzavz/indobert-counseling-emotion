@echo off
cd /d "%~dp0"
rem Membuka ngrok di jendela terpisah, lalu menjalankan server chatbot di jendela ini.
where ngrok >nul 2>nul
if errorlevel 1 (
  echo ngrok tidak ditemukan di PATH. Jalankan ngrok secara manual:
  echo   ngrok http --url=unslicked-connectively-vanda.ngrok-free.dev 5000
) else (
  start "ngrok" cmd /k ngrok http --url=unslicked-connectively-vanda.ngrok-free.dev 5000
)
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
..\Eksperimen\eksperimen\.venv\Scripts\python.exe server_whatsapp.py
pause
