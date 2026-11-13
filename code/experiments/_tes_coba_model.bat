@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe coba_model.py "aku takut banget besok sidang skripsi" "hari ini aku senang sekali bisa lulus" "aku kesal banget sama temanku yang bohong" "aku merasa sendirian dan sedih terus" "besok aku mau ke perpustakaan" > hasil\tes_coba_model.txt 2>&1
