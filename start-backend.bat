@echo off
setlocal EnableDelayedExpansion

echo Starting YOLO Backend Service...

REM Use absolute path to conda Python executable
set "PYTHON_EXE=C:\Users\pc\.conda\envs\yolo_cpu\python.exe"
set "WORKING_DIR=C:\Users\pc\object-detection\backend"

REM Set environment variables
set "DJANGO_SETTINGS_MODULE=yolo_api.settings"
set "YOLO_MODEL_PATH=C:/Users/pc/object-detection/backend/weights/best.pt"
set "DEFAULT_YOLO_DEVICE=cpu"
set "CUDA_VISIBLE_DEVICES="

REM Change to backend directory
cd /d "%WORKING_DIR%"

REM Start uvicorn directly with the Python executable
echo Starting Uvicorn server...
"%PYTHON_EXE%" -m uvicorn yolo_api.asgi:application --host 0.0.0.0 --port 8000 --workers 4 --timeout-keep-alive 120

echo Uvicorn process ended