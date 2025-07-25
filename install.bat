@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo   LSTM Trading System - Complete Installer
echo   Version 1.0 - Jason Rusk 2025
echo ===============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

:: Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo ✅ pip found: 
pip --version
echo.

:: Install Python dependencies
echo 📦 Installing Python dependencies...
pip install torch>=1.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 joblib>=1.1.0 matplotlib>=3.5.0

echo    ✅ All dependencies installed successfully
echo.

echo ===============================================
echo   Installation Complete! 🎉
echo ===============================================
echo.
echo 📋 Next Steps:
echo 1. Copy your EURUSD60.csv data file to this directory
echo 2. Double-click train_model.bat to train your model
echo 3. Double-click start_daemon.bat to start the prediction server
echo 4. Run your MT5 EA (GGTH4_hardened.mq5)
echo.
pause
