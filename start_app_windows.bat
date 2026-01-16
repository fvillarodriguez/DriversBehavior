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
  set "PY_CMD="
  where python >nul 2>&1 && set "PY_CMD=python"
  if not defined PY_CMD (
    where py >nul 2>&1 && set "PY_CMD=py -3"
  )
  if not defined PY_CMD (
    echo Python not found. Install Python 3 to continue.
    exit /b 1
  )
  echo Creating virtual environment in .venv...
  %PY_CMD% -m venv ".venv"
  if errorlevel 1 exit /b 1
  call ".venv\Scripts\activate.bat"
  if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r "requirements.txt"
    if errorlevel 1 exit /b 1
  ) else (
    echo requirements.txt not found. Cannot install dependencies.
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
