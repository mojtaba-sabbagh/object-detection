# YOLO API Starter (Django REST)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip -r requirements.txt

export YOLO_MODEL_PATH=$(pwd)/weights/best.pt  # point to your trained weights
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Test:
```bash
curl -X POST -F "image=@/path/to/image.jpg" "http://localhost:8000/api/detect/?conf=0.25&imgsz=640"
```
