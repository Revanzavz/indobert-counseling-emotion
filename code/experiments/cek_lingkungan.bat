@echo off
cd /d "%~dp0"
if not exist hasil mkdir hasil
set OUT=hasil\cek_lingkungan.txt
echo [%date% %time%] > %OUT%
nvidia-smi >> %OUT% 2>&1
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe") do if exist %%P %%P cek_env.py >> %OUT% 2>&1
for /d %%E in ("%USERPROFILE%\anaconda3\envs\*" "%USERPROFILE%\.conda\envs\*" "%USERPROFILE%\miniconda3\envs\*") do if exist "%%E\python.exe" "%%E\python.exe" cek_env.py >> %OUT% 2>&1
for /d %%E in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%E\python.exe" "%%E\python.exe" cek_env.py >> %OUT% 2>&1
where python >> %OUT% 2>&1
echo SELESAI >> %OUT%
