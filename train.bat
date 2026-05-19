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

if "%EXP_NAME%"=="" set "EXP_NAME=experiment3_joint"
if "%DATA_PATH%"=="" set "DATA_PATH=datasets/SP500_event_taxonomy_w100"
if "%WINDOW_SIZE%"=="" set "WINDOW_SIZE=100"
if "%FEATURES%"=="" set "FEATURES=all"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=32"
if "%EPOCHS%"=="" set "EPOCHS=20"
if "%DEVICE%"=="" set "DEVICE=auto"

%PYTHON% -u scripts\gbm\train_joint.py ^
  --exp-name "%EXP_NAME%" ^
  --data-path "%DATA_PATH%" ^
  --window-size "%WINDOW_SIZE%" ^
  --features "%FEATURES%" ^
  --batch-size "%BATCH_SIZE%" ^
  --epochs "%EPOCHS%" ^
  --device "%DEVICE%" ^
  %*

exit /b %ERRORLEVEL%