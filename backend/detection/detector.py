# detection/detector.py
from ultralytics import YOLO
from django.conf import settings
import time, threading, logging, os, io

try:
    import torch
except Exception:
    torch = None

from PIL import Image  # robust open + format validation

CLASS_NAMES = {0: '0', 1: '1', 2: '2'}

log = logging.getLogger(__name__)
_model = None
_model_lock = threading.Lock()

def _load_model():
    """Load once, raise a clear error if weights missing/incompatible."""
    global _model
    with _model_lock:
        if _model is None:
            path = str(getattr(settings, "YOLO_MODEL_PATH", "weights/best.pt"))
            if not os.path.exists(path):
                raise FileNotFoundError(f"YOLO weights not found at '{path}'. "
                                        f"Set YOLO_MODEL_PATH or place weights/best.pt")
            try:
                _model = YOLO(path)
            except Exception as e:
                # Common: checkpoint built with newer ultralytics (e.g., C3k2)
                raise RuntimeError(
                    f"Failed to load model at '{path}'. "
                    f"Possible Ultralytics/torch mismatch or unsupported layer. Cause: {e}"
                )
    return _model

def _resolve_device(requested):
    req = (str(requested).lower() if requested not in (None, "", "auto") 
           else str(getattr(settings, "DEFAULT_YOLO_DEVICE", "auto")).lower())

    # ONNX -> run on CPU by default (simple and portable)
    if str(getattr(settings, "YOLO_MODEL_PATH", "")).lower().endswith(".onnx"):
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
    # numeric or multi-GPU like "0,1"
    if all(c.isdigit() or c=="," for c in req):
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
    """Read uploaded file into PIL to fail fast on invalid/corrupt images."""
    try:
        data = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)  # so Ultralytics can read again if needed
        im = Image.open(io.BytesIO(data))
        im.verify()          # check corruption
        im = Image.open(io.BytesIO(data)).convert("RGB")  # reopen usable handle
        return im
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid image: {e}")

def run_inference(file_obj, conf=0.25, imgsz=640, device=None):
    model = _load_model()
    device_str = _resolve_device(device)
    pil_img = _read_image_pil(file_obj)

    t0 = time.time()
    try:
        results = model.predict(
            source=pil_img,       # pass validated image
            conf=conf,
            imgsz=imgsz,
            device=device_str,    # safe on cpu/gpu/mps
            verbose=False,
        )
    except Exception as e:
        # Surface low-level device/ops problems clearly
        raise RuntimeError(f"Ultralytics predict() failed on device='{device_str}': {e}")
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
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "width": x2 - x1, "height": y2 - y1},
                "class_id": cls_idx, "class_name": cls_name,
                "confidence": conf_score,
            })
            counts[cls_name] = counts.get(cls_name, 0) + 1

    return {
        "image": {"width": int(r.orig_shape[1]), "height": int(r.orig_shape[0])},
        "inference_ms": int((t1 - t0) * 1000),
        "detections": detections,
        "counts": counts,
        "total": sum(counts.values()),
        "device": device_str,
        "model_path": str(getattr(settings, "YOLO_MODEL_PATH", "")),
    }
