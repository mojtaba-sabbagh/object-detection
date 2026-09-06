Deploy
======

yolo_ui\dist and yolo_ui\node_modules are NOT in the repository, so the
frontend has to be built after each pull. frontend-service runs
`vite preview` on port 4173 and serves yolo_ui\dist; nginx proxies / to it.

    git pull
    .\build-frontend.bat
    .\nssm.exe start backend-service
    .\nssm.exe start frontend-service
    start nginx

If build-frontend.bat reports that files are in use, stop frontend-service
first, then re-run it.

Backend
-------
Requires Python 3.12 or older -- torch 2.4.1 has no 3.13 wheels.

    pip install -r backend\requirements.txt

requirements.txt pins the CUDA 12.4 torch builds, so detection runs on the
GPU automatically (DEFAULT_YOLO_DEVICE defaults to "auto"). Those wheels also
work on a machine with no NVIDIA GPU, falling back to the CPU.
