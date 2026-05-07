@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "VENV_ACTIVATE="
if exist ".venv\Scripts\activate.bat" set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
if not defined VENV_ACTIVATE if exist "venv\Scripts\activate.bat" set "VENV_ACTIVATE=venv\Scripts\activate.bat"

if defined VENV_ACTIVATE (
  call "%VENV_ACTIVATE%"
) else (
  call :create_env
  if errorlevel 1 exit /b 1
)

call :load_env ".env"
call :load_env ".env.local"

streamlit run streamlit_main.py
exit /b %errorlevel%

:create_env
  set "BOOTSTRAP_PY="
  where py >nul 2>&1 && py -3.12 -c "import sys" >nul 2>&1 && set "BOOTSTRAP_PY=py -3.12"
  if not defined BOOTSTRAP_PY (
    where python >nul 2>&1 && set "BOOTSTRAP_PY=python"
  )
  if not defined BOOTSTRAP_PY (
    echo Python not found. Install Python 3 to continue.
    exit /b 1
  )
  set "TORCH_BACKEND_ARG=auto"
  if defined TORCH_BACKEND set "TORCH_BACKEND_ARG=%TORCH_BACKEND%"
  echo Creating virtual environment in .venv via venv_start.py...
  %BOOTSTRAP_PY% "venv_start.py" --torch-backend "%TORCH_BACKEND_ARG%"
  if errorlevel 1 exit /b 1
  if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
  ) else (
    echo .venv was not created correctly.
    exit /b 1
  )
  exit /b 0

:load_env
  set "ENV_FILE=%~1"
  if not exist "%ENV_FILE%" exit /b 0
  for /f "usebackq delims=" %%A in ("%ENV_FILE%") do (
    set "LINE=%%A"
    if not "!LINE!"=="" (
      if not "!LINE:~0,1!"=="#" (
        for /f "tokens=1* delims==" %%K in ("!LINE!") do (
          set "KEY=%%K"
          set "VALUE=%%L"
          if /i "!KEY:~0,7!"=="export " set "KEY=!KEY:~7!"
          if not "!KEY!"=="" set "!KEY!=!VALUE!"
        )
      )
    )
  )
  exit /b 0
