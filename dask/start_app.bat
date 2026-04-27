@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if "%CLUSTER_APP_HOST%"=="" set "CLUSTER_APP_HOST=0.0.0.0"
if "%CLUSTER_APP_PORT%"=="" set "CLUSTER_APP_PORT=18080"
if "%CLUSTER_APP_OPEN_HOST%"=="" set "CLUSTER_APP_OPEN_HOST=127.0.0.1"
if "%CLUSTER_APP_CONFIG%"=="" set "CLUSTER_APP_CONFIG=config.yaml"
if "%CLUSTER_APP_VENV%"=="" set "CLUSTER_APP_VENV=.venv"

set "APP_HOST=%CLUSTER_APP_HOST%"
set "APP_PORT=%CLUSTER_APP_PORT%"
set "APP_OPEN_HOST=%CLUSTER_APP_OPEN_HOST%"
set "APP_URL=http://%APP_OPEN_HOST%:%APP_PORT%"
set "CONFIG_FILE=%CLUSTER_APP_CONFIG%"
set "VENV_DIR=%CLUSTER_APP_VENV%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "INSTALL_STAMP=%VENV_DIR%\.cluster_app_install_stamp"

echo.
echo [Dask Cluster App launcher]
echo Project: %CD%
echo Bind: %APP_HOST%:%APP_PORT%
echo URL: %APP_URL%

call :app_is_running
if "%ERRORLEVEL%"=="0" (
  echo Application is already running.
  start "" "%APP_URL%"
  exit /b 0
)

if not exist "%VENV_PYTHON%" (
  call :find_python
  if errorlevel 1 goto no_python
  echo.
  echo Creating virtual environment with !PYTHON_CMD!
  !PYTHON_CMD! -m venv "%VENV_DIR%"
  if errorlevel 1 goto venv_failed
)

call :check_python "%VENV_PYTHON%"
if errorlevel 1 goto bad_venv

set "NEEDS_INSTALL=0"
if not exist "%INSTALL_STAMP%" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="0" (
  "%VENV_PYTHON%" -c "import cluster_app, fastapi, distributed, uvicorn, zeroconf, cryptography" >nul 2>nul
  if errorlevel 1 set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="1" (
  echo.
  echo Installing or updating dependencies
  "%VENV_PYTHON%" -m pip install --upgrade pip
  if errorlevel 1 goto install_failed
  "%VENV_PYTHON%" -m pip install -e .
  if errorlevel 1 goto install_failed
  echo installed > "%INSTALL_STAMP%"
) else (
  echo.
  echo Virtual environment is ready
)

if not exist "%CONFIG_FILE%" (
  echo.
  echo Creating default %CONFIG_FILE%
  "%VENV_PYTHON%" -m cluster_app.main --config "%CONFIG_FILE%" init
  if errorlevel 1 goto config_failed
)

echo.
echo Starting application
start "Dask Cluster App Server" "%VENV_PYTHON%" -m cluster_app.main --config "%CONFIG_FILE%" start --host "%APP_HOST%" --port "%APP_PORT%"

for /L %%I in (1,1,60) do (
  call :app_is_running
  if "!ERRORLEVEL!"=="0" goto started
  timeout /t 1 /nobreak >nul
)

echo.
echo ERROR: Application did not start. Check the server window output.
pause
exit /b 1

:started
echo.
echo Application started at %APP_URL%
start "" "%APP_URL%"
exit /b 0

:find_python
set "PYTHON_CMD="
if not "%PYTHON%"=="" (
  "%PYTHON%" -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD="%PYTHON%""
)
if "%PYTHON_CMD%"=="" (
  py -3.12 -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)
if "%PYTHON_CMD%"=="" (
  python -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if "%PYTHON_CMD%"=="" exit /b 1
exit /b 0

:check_python
%~1 -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
exit /b %ERRORLEVEL%

:app_is_running
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%APP_URL%/api/metrics/status' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
exit /b %ERRORLEVEL%

:no_python
echo.
echo ERROR: Python 3.12 or 3.13 is required. Install Python 3.12 and run this file again.
pause
exit /b 1

:venv_failed
echo.
echo ERROR: Could not create virtual environment.
pause
exit /b 1

:bad_venv
echo.
echo ERROR: Existing %VENV_DIR% is not Python 3.12/3.13. Remove it or set CLUSTER_APP_VENV to another folder.
pause
exit /b 1

:install_failed
echo.
echo ERROR: Could not install dependencies.
pause
exit /b 1

:config_failed
echo.
echo ERROR: Could not create config file.
pause
exit /b 1
