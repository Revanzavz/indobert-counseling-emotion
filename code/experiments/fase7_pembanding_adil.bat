@echo off
setlocal
cd /d "%~dp0"
set LOG=hasil\log_fase7.txt
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
echo ==== [%date% %time%] mulai fase 7 ==== >> %LOG%
.venv\Scripts\python.exe -u jalankan_eksperimen.py --fase 7 >> %LOG% 2>&1
.venv\Scripts\python.exe rekap_fase7.py >> %LOG% 2>&1
echo ==== [%date% %time%] FASE 7 SELESAI ==== >> %LOG%
endlocal
