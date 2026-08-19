@echo off
setlocal

set "REPO=C:\Users\user\Desktop\Ising_model"
set "LOG=%REPO%\logs\git_auto_push.log"

cd /d "%REPO%" || exit /b 1
echo [%date% %time%] Starting automatic Git push >> "%LOG%"

git add . >> "%LOG%" 2>&1 || exit /b 1
git diff --cached --quiet
if not errorlevel 1 (
    echo [%date% %time%] No changes to commit >> "%LOG%"
    exit /b 0
)

git commit -m "Automated update %date% %time%" >> "%LOG%" 2>&1 || exit /b 1
git push origin main >> "%LOG%" 2>&1
echo [%date% %time%] Finished with exit code %errorlevel% >> "%LOG%"
exit /b %errorlevel%