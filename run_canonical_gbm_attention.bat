@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

if "%PYTHON%"=="" set "PYTHON=python"
if "%PYTORCH_ENABLE_MPS_FALLBACK%"=="" set "PYTORCH_ENABLE_MPS_FALLBACK=1"
if "%AT_OUTPUT_ROOT%"=="" set "AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

%PYTHON% -u scripts\gbm\run_canonical_gbm_attention.py %*

exit /b %ERRORLEVEL%
