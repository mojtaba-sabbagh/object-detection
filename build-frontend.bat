@echo off
setlocal EnableDelayedExpansion

REM Builds the frontend into yolo_ui\dist, which is no longer committed to the
REM repository. Run this after every `git pull` and before starting
REM frontend-service, which serves dist with `vite preview` on port 4173
REM (nginx proxies / to that port).

echo Building YOLO frontend...

set "UI_DIR=%~dp0yolo_ui"
cd /d "%UI_DIR%"
if !ERRORLEVEL! neq 0 (
    echo Could not enter "%UI_DIR%".
    exit /b 1
)

REM `npm ci`, not `npm install`: ci installs exactly what package-lock.json
REM pins, including the platform-specific @rollup and @esbuild binaries that
REM `npm install` is known to skip (npm/cli#4828). Skipping them makes
REM `vite build` fail with "Cannot find module @rollup/rollup-win32-x64-msvc".
echo Installing dependencies from package-lock.json...
REM Guards use `neq 0` rather than `if errorlevel 1`: npm reports Windows
REM errno values directly, so a file held open exits with -4048, and
REM `if errorlevel 1` means "errorlevel >= 1" -- it misses negative
REM codes and would carry straight on to the build.
call npm ci
if !ERRORLEVEL! neq 0 (
    echo.
    echo npm ci FAILED. It deletes node_modules first, so if it reported that
    echo files were in use, stop frontend-service and try again.
    exit /b 1
)

echo Building...
call npm run build
if !ERRORLEVEL! neq 0 (
    echo.
    echo Build FAILED.
    exit /b 1
)

echo.
echo Build complete. yolo_ui\dist is ready for frontend-service.
