@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =================================================================
:: Step 1: Check Python installation and version
:: =================================================================
echo [Step 1] Checking Python environment...

:: Check if python command exists
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python not detected.
    echo Please install Python 3.10 or higher and add it to PATH.
    echo.
    pause
    exit /b 1
)

:: Check Python version >= 3.10
for /f "tokens=2" %%v in ('python --version 2^>^&1') do (
    set "PYTHON_VERSION=%%v"
)

for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set "PYTHON_MAJOR=%%a"
    set "PYTHON_MINOR=%%b"
)

if !PYTHON_MAJOR! LSS 3 (
    set "VERSION_OK=0"
) else if !PYTHON_MAJOR! EQU 3 (
    if !PYTHON_MINOR! LSS 10 (
        set "VERSION_OK=0"
    ) else (
        set "VERSION_OK=1"
    )
) else (
    set "VERSION_OK=1"
)

if "!VERSION_OK!"=="0" (
    echo.
    echo Error: Python version is !PYTHON_VERSION!. Version 3.10 or higher is required.
    echo Please upgrade your Python installation.
    echo.
    pause
    exit /b 1
)

echo Python version !PYTHON_VERSION! is OK.
echo.

:: =================================================================
:: Step 2: Check MATLAB Engine for Python
:: =================================================================
echo [Step 2] Checking MATLAB Engine for Python...
python -c "import matlab.engine" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo MATLAB Engine for Python already installed, skipping installation.
    goto INSTALL_DEPS
)

echo MATLAB Engine for Python not installed, preparing installation...
echo.

:: =================================================================
:: Step 3: Enter MATLAB path manually
:: =================================================================
echo [Step 3] Please input MATLAB root path
echo Example: C:\Program Files\MATLAB\R2023a
echo.
set /p "MATLAB_ROOT=Enter path and press Enter: "

:: Trim spaces
for /f "tokens=* delims= " %%a in ("!MATLAB_ROOT!") do set "MATLAB_ROOT=%%a"

:: Validate input
if not defined MATLAB_ROOT (
    echo.
    echo Error: No path entered, installation aborted.
    echo.
    pause
    exit /b 1
)

if not exist "!MATLAB_ROOT!\extern\engines\python\setup.py" (
    echo.
    echo Error: MATLAB Engine setup.py not found under "!MATLAB_ROOT!".
    echo Please check if the path is correct.
    echo.
    pause
    exit /b 1
)

echo Your MATLAB path: !MATLAB_ROOT!
echo.

:: =================================================================
:: Step 4: Install MATLAB Engine
:: =================================================================
echo [Step 4] Installing MATLAB Engine...
set "ENGINE_PATH=!MATLAB_ROOT!\extern\engines\python"

pushd "!ENGINE_PATH!"
python -m pip install .
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: MATLAB Engine installation failed.
    echo Please check Python version and MATLAB version compatibility.
    echo Or try running this script with administrator privileges.
    echo.
    popd
    pause
    exit /b 1
)
popd
echo MATLAB Engine installed successfully!
echo.

:: =================================================================
:: Step 5: Install project dependencies
:: =================================================================
:INSTALL_DEPS
echo [Step 5] Installing project dependencies (requirements.txt)...
cd /d %~dp0
if not exist "requirements.txt" (
    echo Warning: requirements.txt not found, skipping step.
) else (
    pip install -r requirements.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo Error: Dependency installation failed.
        echo Please check requirements.txt and your network.
        echo.
        pause
        exit /b 1
    )
)
echo Dependencies installed/checked successfully.
echo.

:: =================================================================
:: Step 6: Launch application
:: =================================================================
echo [Step 6] Starting GUI application (run_gui.py)...
echo.
python run_gui.py

echo.
echo Application exited.
pause
endlocal
