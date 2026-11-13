@echo off
setlocal
cd /d "%~dp0"
set LOG=hasil\log_fase2_6.txt
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
echo ==== [%date% %time%] menunggu fase 1 selesai ==== >> %LOG%
:tunggu
findstr /C:"FASE 1 SELESAI" hasil\log_fase1.txt >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto tunggu
)
echo ==== [%date% %time%] mulai fase 2-6 ==== >> %LOG%
for %%F in (2 3 4 5 6) do (
  echo ---- [%date% %time%] FASE %%F ---- >> %LOG%
  .venv\Scripts\python.exe -u jalankan_eksperimen.py --fase %%F >> %LOG% 2>&1
  .venv\Scripts\python.exe rekap_hasil.py >> %LOG% 2>&1
)
.venv\Scripts\python.exe ensemble.py >> %LOG% 2>&1
echo ==== [%date% %time%] SEMUA FASE SELESAI ==== >> %LOG%
endlocal
