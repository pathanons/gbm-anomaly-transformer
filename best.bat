@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul

set "PYTHON_BIN=python"
set "CONFIG_PATH=configs\general\best_model_suite.yaml"
if not defined AT_OUTPUT_ROOT set "AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns"
if not defined DEVICE set "DEVICE=auto"

set "RUN_TESTS=1"
set "RUN_SUITE=1"
set "DRY_RUN_ONLY=0"

:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="--dry-run" (
    set "DRY_RUN_ONLY=1"
    shift
    goto parse_args
)
if /I "%~1"=="--test-only" (
    set "RUN_SUITE=0"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-tests" (
    set "RUN_TESTS=0"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto help
echo Unknown option: %~1
goto help

:after_args
echo.
echo [best] GBM Anomaly Transformer best-model suite
echo [best] repo: %CD%
echo [best] config: %CONFIG_PATH%
echo [best] AT_OUTPUT_ROOT: %AT_OUTPUT_ROOT%
echo [best] DEVICE: %DEVICE%
echo.

if "%RUN_TESTS%"=="1" (
    echo [best] running pytest gate...
    %PYTHON_BIN% -m pytest unittest
    if errorlevel 1 goto fail
    echo.
)

if "%RUN_SUITE%"=="0" goto done

echo [best] checking suite expansion with dry-run...
%PYTHON_BIN% run.py --config "%CONFIG_PATH%" --dry-run
if errorlevel 1 goto fail
echo.

if "%DRY_RUN_ONLY%"=="1" goto done

echo [best] starting long suite...
echo [best] this runs train -^> validate -^> test for all configured trials, then MAD k sweep.
%PYTHON_BIN% run.py --config "%CONFIG_PATH%"
if errorlevel 1 goto fail

:done
echo.
echo [best] completed successfully.
popd >nul
endlocal
exit /b 0

:fail
echo.
echo [best] failed. See the error above.
popd >nul
endlocal
exit /b 1

:help
echo.
echo Usage:
echo   best.bat              Run tests, dry-run, then the full best-model suite.
echo   best.bat --dry-run    Run tests and dry-run only.
echo   best.bat --test-only  Run pytest only.
echo   best.bat --skip-tests Skip pytest, dry-run, then run the full suite.
echo.
echo Environment overrides:
echo   set AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns
echo   set DEVICE=auto
echo.
popd >nul
endlocal
exit /b 2
