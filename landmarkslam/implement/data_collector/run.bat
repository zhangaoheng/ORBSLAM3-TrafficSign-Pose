@echo off
cd /d "%~dp0"

REM Generate timestamp-based experiment name: YYYYMMDD_HHMMSS
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "dt=%%I"
set "NAME=run_%dt:~0,8%_%dt:~8,6%"

echo ====================================
echo   GPS-CAM Data Collector
echo ====================================
echo  Name : %NAME%
echo  Out  : data\%NAME%\
echo ====================================
echo.

"C:\Users\zah\anaconda3\envs\gps_cam_env\python.exe" d456_gps2.py --name "%NAME%"

echo.
pause
