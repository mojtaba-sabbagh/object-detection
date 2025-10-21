# detection/detector.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time, base64, io, logging
import numpy as np
import cv2
from PIL import Image, ImageOps
from django.conf import settings

log = logging.getLogger(__name__)

# COLORS: BGR (OpenCV)
COLORS = {
    "0": (0, 255, 255),   # Yellow
    "1": (0, 165, 255),   # Orange
    "2": (0, 0, 0),       # Black
}

# ------------------ model/device helpers ------------------

_MODEL = None
_MODEL_WEIGHTS = None

def _load_model():
    """
    Lazy-load and cache a single YOLO model instance.
    Uses settings.YOLO_MODEL_PATH (must point to your trained .pt).
    """
    global _MODEL, _MODEL_WEIGHTS
    if _MODEL is not None:
        return _MODEL

    weights = getattr(settings, "YOLO_MODEL_PATH", None)
    if not weights:
        # Give a precise error so the view can surface it when DEBUG=True
        raise RuntimeError(
            "YOLO_MODEL_PATH is not configured in settings. "
            "Set it to the path of your trained weights (e.g., '/path/to/best.pt')."
        )

    try:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
        _MODEL_WEIGHTS = weights
        log.info("Loaded YOLO weights: %s", weights)
        return _MODEL
    except Exception as e:
        log.exception("Failed to load YOLO weights from %s", weights)
        raise

def _resolve_device(device: str | None) -> str:
    """
    device in {'auto','cpu','mps','0','0,1',...}
    - 'auto' => '0' if CUDA available, else 'mps' (if available), else 'cpu'
    """
    dev = (device or "auto").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev in ("mps", "metal"):
        return "mps"  # Apple GPU if your torch was built with MPS

    try:
        import torch
        if dev == "auto":
            if torch.cuda.is_available():
                return "0"
            # Prefer MPS on Apple if available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        # For explicit strings like "0" or "0,1"
        if dev == "0" or dev.replace(",", "").isdigit():
            if torch.cuda.is_available():
                return dev
            return "cpu"
    except Exception:
        pass
    return "cpu"

# ------------------ geometry & NMS ------------------

def _iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _nms_classwise(dets: List[Dict[str, Any]], iou_th: float = 0.5) -> List[Dict[str, Any]]:
    """NMS per class on global boxes (expects det['bbox'] as x1,y1,x2,y2)."""
    out: List[Dict[str, Any]] = []
    by_cls: Dict[str, List[int]] = {}
    for i, d in enumerate(dets):
        by_cls.setdefault(d["class_name"], []).append(i)
    for _, idxs in by_cls.items():
        idxs = sorted(idxs, key=lambda i: dets[i]["confidence"], reverse=True)
        keep: List[int] = []
        while idxs:
            i = idxs.pop(0)
            keep.append(i)
            xi = dets[i]["bbox"]
            idxs = [j for j in idxs if _iou_xyxy(xi, dets[j]["bbox"]) < iou_th]
        out.extend(dets[k] for k in keep)
    return out

# ------------------ IO & model calls ------------------

def _pil_from_file(file_obj) -> Image.Image:
    if hasattr(file_obj, "read"):
        pos = file_obj.tell() if hasattr(file_obj, "tell") else None
        try:
            file_obj.seek(0)
        except Exception:
            pass
        data = file_obj.read()
        if pos is not None:
            try:
                file_obj.seek(pos)
            except Exception:
                pass
    else:
        data = file_obj

    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def _predict_on_pil(model, pil_img: Image.Image, conf: float, imgsz: int, device: str):
    # Ultralytics accepts PIL directly
    results = model.predict(source=pil_img, conf=conf, imgsz=imgsz, device=device, verbose=False)
    return results[0]

def _boxes_from_result(r) -> List[Dict[str, Any]]:
    boxes = getattr(r, "boxes", None)
    dets: List[Dict[str, Any]] = []
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls
    conf = boxes.conf.cpu().numpy().astype(float) if hasattr(boxes.conf, "cpu") else boxes.conf
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
        dets.append({
            "class_name": str(int(cls[i])),
            "confidence": float(conf[i]),
            "bbox": (x1, y1, x2, y2),
        })
    return dets

def _draw_rects_bgr(img_bgr: np.ndarray, dets: List[Dict[str, Any]], thickness: int = 2) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
        xi2, yi2 = min(w-1, int(round(x2))), min(h-1, int(round(y2)))
        color = COLORS.get(d["class_name"], (34, 197, 94))  # emerald default
        cv2.rectangle(img_bgr, (xi1, yi1), (xi2, yi2), color, thickness, lineType=cv2.LINE_AA)
    return img_bgr

# ------------------ main API ------------------

def run_inference(
    file_obj,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    annotate: bool = False,
    tile: str | int | bool = "auto",   # "auto"|1|0
    tile_size: int = 640,
    overlap: float = 0.20,             # 20% overlap
    nms_iou: float = 0.50,
) -> Dict[str, Any]:
    """
    Tiled inference for large images. Returns:
      image: {width,height}, detections, counts, total, inference_ms, image_b64 (if annotate)
    """
    # Load model & device
    model = _load_model()
    dev = _resolve_device(device)

    # Read image
    pil = _pil_from_file(file_obj)
    W, H = pil.size

    def _package(dets: List[Dict[str, Any]], t_ms: int, annotated_bgr: np.ndarray | None):
        # build counts + width/height + expand bbox dict format
        counts: Dict[str, int] = {}
        out_dets: List[Dict[str, Any]] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            out_dets.append({
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1
                }
            })
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

        payload: Dict[str, Any] = {
            "image": {"width": W, "height": H},
            "inference_ms": t_ms,
            "detections": out_dets,
            "counts": counts,
            "total": sum(counts.values()),
        }

        if annotate and annotated_bgr is not None:
            ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                payload["image_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return payload

    # Decide tiling
    if isinstance(tile, str):
        t = tile.lower()
        tile_flag = (t == "auto" and max(W, H) > tile_size) or (t in ("1", "true", "yes"))
    else:
        tile_flag = bool(tile)

    t0 = time.time()

    if not tile_flag:
        # ----- Simple single-pass inference -----
        r = _predict_on_pil(model, pil, conf=conf, imgsz=imgsz, device=dev)
        dets = _boxes_from_result(r)
        dets = _nms_classwise(dets, iou_th=nms_iou)
        annotated = None
        if annotate:
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            annotated = _draw_rects_bgr(bgr, dets)
        t1 = time.time()
        return _package(dets, int((t1 - t0) * 1000), annotated)

    # ----- Tiled inference -----
    step = max(1, int(tile_size * (1.0 - overlap)))
    all_dets: List[Dict[str, Any]] = []

    for top in range(0, H, step):
        for left in range(0, W, step):
            right = min(left + tile_size, W)
            bottom = min(top + tile_size, H)
            if right <= left or bottom <= top:
                continue
            crop = pil.crop((left, top, right, bottom))  # RGB crop
            r = _predict_on_pil(model, crop, conf=conf, imgsz=imgsz, device=dev)
            dets = _boxes_from_result(r)
            # translate to global coords
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = (x1 + left, y1 + top, x2 + left, y2 + top)
                all_dets.append(d)

    # NMS across all tiles (class-wise)
    merged = _nms_classwise(all_dets, iou_th=nms_iou)

    annotated = None
    if annotate:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        annotated = _draw_rects_bgr(bgr, merged)

    t1 = time.time()
    return _package(merged, int((t1 - t0) * 1000), annotated)
