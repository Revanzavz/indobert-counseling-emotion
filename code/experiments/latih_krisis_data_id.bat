@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set DATA_KRISIS=Krisis_ID
set SUFIKS=_id
echo Melatih model deteksi krisis dengan data tweet Indonesia asli (apricitea), sekitar 10 menit
echo Model lama di hasil\model_krisis tidak diubah.
.venv\Scripts\python.exe latih_krisis.py || goto gagal
echo.
echo SELESAI. Hasil: Eksperimen\eksperimen\hasil\evaluasi_krisis_id.md
pause
exit /b 0
:gagal
echo.
echo TERJADI KESALAHAN, lihat pesan di atas.
pause
