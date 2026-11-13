@echo off
cd /d "%~dp0"
echo Menjalankan turnamen eksperimen IndoBERT (aman dihentikan dan dilanjutkan)...
python jalankan_eksperimen.py %*
python rekap_hasil.py
pause
