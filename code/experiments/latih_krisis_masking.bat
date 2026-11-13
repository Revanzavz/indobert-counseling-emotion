@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set MASKING=1
set P_MASK=0.3
echo Melatih model deteksi krisis dengan penyamaran kata pemicu (sekitar 6 menit)
echo Model lama di hasil\model_krisis tidak diubah.
.venv\Scripts\python.exe latih_krisis.py || goto gagal
echo.
echo SELESAI. Hasil: Eksperimen\eksperimen\hasil\evaluasi_krisis_masking.md
pause
exit /b 0
:gagal
echo.
echo TERJADI KESALAHAN, lihat pesan di atas.
pause
