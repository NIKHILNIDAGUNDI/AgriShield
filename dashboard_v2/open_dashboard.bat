@echo off
cd /d "%~dp0frontend"
if errorlevel 1 exit /b 1
npm run dev -- --open
