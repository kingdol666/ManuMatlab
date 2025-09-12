@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =================================================================
:: Step 1: Check for Python
:: =================================================================
echo [Step 1] Checking for Python...

:: Check if python command exists
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Python is not installed or not in the system PATH.
    echo Please install Python and ensure it's added to the PATH.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do (
    set "PYTHON_VERSION=%%v"
)
echo Found Python version !PYTHON_VERSION!. Proceeding...
echo.

:: =================================================================
:: Step 2: Validate or Create Python Virtual Environment
:: =================================================================
echo [Step 2] Validating or creating Python virtual environment...

set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=!VENV_DIR!\Scripts\python.exe"
set "VENV_VALID=0"

:: Check if venv seems to exist and is valid
if exist "!VENV_PYTHON!" (
    echo Found existing venv. Validating...
    "!VENV_PYTHON!" --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo venv is valid.
        set "VENV_VALID=1"
    ) else (
        echo venv is invalid or corrupted. It will be recreated.
    )
)

:: If venv is not valid, recreate it
if "!VENV_VALID!"=="0" (
    if exist "!VENV_DIR!" (
        echo Deleting invalid venv...
        rmdir /s /q "!VENV_DIR!"
    )
    
    echo Creating new virtual environment 'venv' with Python !PYTHON_VERSION!...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo Error: Failed to create the virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

:: =================================================================
:: Step 3: Activate Python Virtual Environment
:: =================================================================
echo [Step 3] Activating Python virtual environment...
set "VENV_ACTIVATOR=%~dp0venv\Scripts\activate.bat"
call "!VENV_ACTIVATOR!"
echo Virtual environment activated.
echo.

:: =================================================================
:: Step 4: Check MATLAB Engine and Install Dependencies
:: =================================================================
echo [Step 4] Checking MATLAB Engine for Python...
python -c "import matlab.engine" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo MATLAB Engine for Python already installed.
    goto INSTALL_DEPS
)

echo MATLAB Engine not found. Preparing for installation...
echo.

:: --- Sub-step: Find or Enter MATLAB path (with caching) ---
echo Finding MATLAB path...
set "MATLAB_PATH_CACHE_FILE=%~dp0matlab_path.cfg"
if exist "!MATLAB_PATH_CACHE_FILE!" (
    set /p MATLAB_ROOT=<"!MATLAB_PATH_CACHE_FILE!"
    echo Found cached MATLAB path: !MATLAB_ROOT!
)

set "PATH_VALID=0"
if defined MATLAB_ROOT (
    if exist "!MATLAB_ROOT!\extern\engines\python\setup.py" ( set "PATH_VALID=1" ) else ( echo Cached path is invalid. )
)

if "!PATH_VALID!"=="0" (
    echo.
    echo Please input MATLAB root path (e.g., C:\Program Files\MATLAB\R2023a)
    set /p "MATLAB_ROOT=Enter path: "
)

if not defined MATLAB_ROOT ( echo. & echo Error: No path entered. & pause & exit /b 1 )
if not exist "!MATLAB_ROOT!\extern\engines\python\setup.py" ( echo. & echo Error: MATLAB Engine setup.py not found at the specified path. & pause & exit /b 1 )

echo !MATLAB_ROOT!>"!MATLAB_PATH_CACHE_FILE!"
echo Using MATLAB path: !MATLAB_ROOT!
echo.

:: --- Sub-step: Install setuptools if needed (for Python >= 3.10) ---
echo Checking Python version for setuptools requirement...
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if "!PY_MAJOR!" EQU "3" (
    if "!PY_MINOR!" GEQ "10" (
        echo Python version is 3.10 or newer. Ensuring setuptools is installed...
        python -m pip install setuptools
        IF !ERRORLEVEL! NEQ 0 ( echo. & echo Error: setuptools installation failed. & pause & exit /b 1 )
    )
)

:: --- Sub-step: Install MATLAB Engine ---
echo Installing MATLAB Engine...
pushd "!MATLAB_ROOT!\extern\engines\python"
python setup.py install
IF %ERRORLEVEL% NEQ 0 ( echo. & echo Error: MATLAB Engine installation failed. & popd & pause & exit /b 1 )
popd
echo MATLAB Engine installed successfully!
echo.

:: =================================================================
:: Step 5: Install Project Dependencies
:: =================================================================
:INSTALL_DEPS
echo [Step 5] Installing project dependencies from requirements.txt...
cd /d %~dp0
if not exist "requirements.txt" (
    echo Warning: requirements.txt not found.
) else (
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    IF %ERRORLEVEL% NEQ 0 ( echo. & echo Error: Dependency installation failed. & pause & exit /b 1 )
)
echo Dependencies installed/checked successfully.
echo.

:: =================================================================
:: Step 6: Launch Application
:: =================================================================
echo [Step 6] Starting GUI application (run_gui.py)...
echo.
python run_gui.py

echo.
echo Application exited.
pause
endlocal
