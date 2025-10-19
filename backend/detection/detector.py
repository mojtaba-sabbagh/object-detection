from ultralytics import YOLO
from django.conf import settings
from .models import YoloModel
import time, threading, logging, os, io

try:
    import torch
except Exception:
    torch = None

from PIL import Image

CLASS_NAMES = {0: "0", 1: "1", 2: "2"}
log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_active_model_id = None  # track which DB model is loaded

def _current_active():
    # DB first; fallback to settings.YOLO_MODEL_PATH if no rows exist
    m = YoloModel.objects.filter(is_active=True).first()
    if m:
        return m
    # Optional fallback if admin hasn't added any rows yet
    path = getattr(settings, "YOLO_MODEL_PATH", None)
    if path:
        pseudo = type("M", (), {})()
        pseudo.id = -1
        pseudo.name = os.path.basename(path)
        pseudo.weights_path = path
        return pseudo
    return None

def _load_model():
    global _model, _active_model_id
    with _model_lock:
        active = _current_active()
        if not active:
            raise RuntimeError("No active YOLO model found. Add one in admin and mark it active.")
        # Reload if first time or active changed
        if (_model is None) or (_active_model_id != active.id):
            log.info("Loading YOLO model from %s (id=%s)", active.weights_path, getattr(active, "id", None))
            _model = YOLO(active.weights_path)
            _active_model_id = active.id
    return _model

def _resolve_device(requested):
    req = (str(requested).lower() if requested not in (None, "", "auto")
           else str(getattr(settings, "DEFAULT_YOLO_DEVICE", "auto")).lower())

    # ONNX -> stick to CPU (portable)
    active = _current_active()
    if active and str(active.weights_path).lower().endswith(".onnx"):
        return "cpu"

    if torch is None:
        return "cpu"
    if req in ("cpu", "-1"):
        return "cpu"
    if req in ("mps", "apple", "mac"):
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if req == "auto":
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if all(c.isdigit() or c == "," for c in req):
        if not torch.cuda.is_available():
            return "cpu"
        first = req.split(",")[0]
        try:
            idx = int(first)
            return req if idx < torch.cuda.device_count() else "cpu"
        except ValueError:
            return "cpu"
    return "cpu"

def _read_image_pil(file_obj):
    try:
        data = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return im
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid image: {e}")

def run_inference(file_obj, conf=0.25, imgsz=640, device=None):
    model = _load_model()
    device_str = _resolve_device(device)
    pil_img = _read_image_pil(file_obj)

    t0 = time.time()
    results = model.predict(source=pil_img, conf=conf, imgsz=imgsz, device=device_str, verbose=False)
    t1 = time.time()

    r = results[0]
    boxes = r.boxes
    detections = []
    counts = {CLASS_NAMES[i]: 0 for i in CLASS_NAMES}

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
            conf_score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            detections.append({
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1},
                "class_name": cls_name,
                "confidence": conf_score,
            })
            counts[cls_name] = counts.get(cls_name, 0) + 1

    # echo active model name in response
    active = _current_active()
    active_name = getattr(active, "name", None)

    return {
        "image": {"width": int(r.orig_shape[1]), "height": int(r.orig_shape[0])},
        "inference_ms": int((t1 - t0) * 1000),
        "detections": detections,
        "counts": counts,
        "total": sum(counts.values()),
        "device": device_str,
        "active_model": active_name,
    }
