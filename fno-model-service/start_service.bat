@echo off
echo ========================================
echo FNO Model Service Startup Script
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python first.
    pause
    exit /b 1
)

echo.
echo Setting up environment variables...
set MODEL_PATH=%SCRIPT_DIR%..\fno2_model.pth
set DEVICE=cpu
set PORT=5000
set DEBUG=True

echo MODEL_PATH=%MODEL_PATH%
echo DEVICE=%DEVICE%
echo PORT=%PORT%
echo.

echo Checking model file...
if exist "%MODEL_PATH%" (
    echo Model file found: %MODEL_PATH%
) else (
    echo Warning: Model file not found at %MODEL_PATH%
    echo You can load a model later using POST /api/model/load
)
echo.

echo Starting FNO Model Service...
echo.
echo Service will be available at:
echo   http://localhost:%PORT%
echo.
echo API Endpoints:
echo   GET  /api/health          - Health check
echo   GET  /api/info            - Service information
echo   POST /api/model/load      - Load model
echo   POST /api/model/unload    - Unload model
echo   GET  /api/model/info      - Get model info
echo   POST /api/predict         - Run prediction
echo   POST /api/predict/batch   - Run batch prediction
echo   GET  /api/test/sample     - Get sample input
echo   POST /api/test/predict    - Test prediction
echo.

python app.py

pause
