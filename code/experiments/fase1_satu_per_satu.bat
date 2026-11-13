@echo off
setlocal
cd /d "%~dp0"
if not exist hasil mkdir hasil
set LOG=hasil\log_fase1.txt
set PYBASE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
echo ==== [%date% %time%] mulai fase 1 ==== >> %LOG%

rem --- lingkungan khusus TA (memakai ulang PyTorch CUDA yang sudah terpasang) ---
if not exist .venv\Scripts\python.exe (
  echo membuat .venv >> %LOG%
  "%PYBASE%" -m venv --system-site-packages .venv >> %LOG% 2>&1
  .venv\Scripts\python.exe -m pip install -q "transformers==4.46.3" tabulate bitsandbytes >> %LOG% 2>&1
)
.venv\Scripts\python.exe cek_env.py >> %LOG% 2>&1

rem --- satu backbone per proses: memori GPU dibersihkan setiap ganti model ---
for %%B in (indobert-lite-base-p1 indobert-lite-base-p2 indobert-base-p1 indobert-base-p2 indolem-indobert-base indobertweet-base indobert-lite-large-p1 indobert-lite-large-p2 indobert-large-p1 indobert-large-p2) do (
  echo ---- [%date% %time%] backbone %%B ---- >> %LOG%
  .venv\Scripts\python.exe -u jalankan_eksperimen.py --fase 1 --backbone %%B >> %LOG% 2>&1
)
.venv\Scripts\python.exe rekap_hasil.py >> %LOG% 2>&1
echo ==== [%date% %time%] FASE 1 SELESAI ==== >> %LOG%
endlocal
