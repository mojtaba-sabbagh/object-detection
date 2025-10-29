@echo off
setlocal EnableDelayedExpansion

echo Starting YOLO Backend Service... >> C:\Users\pc\object-detection\backend-service.log 2>&1

REM Set Python paths for service environment
set "PATH="C:\Users\pc\.conda\envs\yolo_cpu;C:\Users\pc\.conda\envs\yolo_cpu\Scripts;C:\Users\pc\.conda\envs\yolo_cpu\Library\bin;%PATH%"
set "CONDA_DEFAULT_ENV=yolo_cpu"
set "CONDA_PREFIX=C:\ProgramData\anaconda3\envs\yolo_cpu"

REM === Env vars for Django & your app ===
set "DJANGO_SETTINGS_MODULE=yolo_api.settings"
set "YOLO_MODEL_PATH=C:/Users/pc/object-detection/backend/weights/best.pt"
set "DEFAULT_YOLO_DEVICE=cpu"
set "CUDA_VISIBLE_DEVICES="

REM === Change to backend dir ===
cd /d C:\Users\pc\object-detection\backend

REM === Activate conda and start uvicorn ===
call "C:\ProgramData\anaconda3\condabin\conda.bat" activate yolo_cpu

echo Starting Uvicorn server...
uvicorn yolo_api.asgi:application --host 0.0.0.0 --port 8000 --workers 4 --timeout-keep-alive 120

echo Uvicorn process ended