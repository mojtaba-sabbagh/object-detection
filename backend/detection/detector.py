# detection/detector.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time, base64, io, logging, os
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

def _resolve_weights_path() -> str:
    """
    Decide which weights file to load, in priority order:
      1) The 'weights_path' of the active YoloModel row in the database
         (YoloModel.objects.filter(is_active=True).first()), if that file
         actually exists on this machine.
      2) settings.YOLO_MODEL_PATH, as a fallback if no active model is set
         in the database, if the DB can't be queried yet (e.g. during
         migrations), or if the active row's weights_path is missing here.
    """
    try:
        from .models import YoloModel
        active = YoloModel.objects.filter(is_active=True).first()
        if active and active.weights_path:
            if os.path.exists(active.weights_path):
                return active.weights_path
            # The row travels with the database between checkouts, so it can
            # hold an absolute path from another OS (a macOS '/Users/...' path
            # seen from Windows, say). That is recoverable, not fatal: fall
            # through to the settings default, which is derived from BASE_DIR.
            log.warning(
                "Active YoloModel %r has weights_path %r, which does not exist "
                "on this machine; falling back to settings.YOLO_MODEL_PATH.",
                getattr(active, "name", None) or active.pk,
                active.weights_path,
            )
    except Exception:
        log.exception(
            "Could not query the active YoloModel from the database; "
            "falling back to settings.YOLO_MODEL_PATH."
        )

    weights = getattr(settings, "YOLO_MODEL_PATH", None)
    if not weights:
        raise RuntimeError(
            "No active model is set in the database (YoloModel.is_active) and "
            "YOLO_MODEL_PATH is not configured in settings. Mark a model ACTIVE "
            "in the admin, or set YOLO_MODEL_PATH as a fallback."
        )
    return weights


def _load_model():
    """
    Lazy-load and cache a YOLO model instance, keyed to whichever weights path
    _resolve_weights_path() currently returns (the database's active model,
    or settings.YOLO_MODEL_PATH as a fallback).

    If the active model changes in the database (e.g. via the admin's
    "Mark as ACTIVE" action), the next inference request will detect the
    path change and reload the new weights automatically -- no server
    restart required.
    """
    global _MODEL, _MODEL_WEIGHTS

    weights = _resolve_weights_path()

    if _MODEL is not None and _MODEL_WEIGHTS == weights:
        return _MODEL

    try:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
        _MODEL_WEIGHTS = weights
        log.info("Loaded YOLO weights: %s", weights)
        return _MODEL
    except Exception:
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
            log.info(
                "No GPU detected (cuda.is_available=%s, mps available=%s); falling back to CPU.",
                torch.cuda.is_available(),
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            )
            return "cpu"
        # For explicit strings like "0" or "0,1"
        if dev == "0" or dev.replace(",", "").isdigit():
            if torch.cuda.is_available():
                return dev
            log.warning("Requested CUDA device %r but CUDA is not available; using CPU.", dev)
            return "cpu"
    except Exception:
        log.warning("torch import/device check failed; defaulting to CPU.", exc_info=True)
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

def _counts_from_dets(dets: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return counts

def _draw_legend_bgr(
    img_bgr: np.ndarray,
    counts: Dict[str, int],
    total: int,
    colors: Dict[str, Tuple[int, int, int]] = COLORS,
    margin: int = 12,
    scale: float = 2.0,
) -> np.ndarray:
    """
    Draws a small semi-transparent legend box in the bottom-left corner of
    the image, showing per-class counts (color-matched to the bounding box
    colors) and the grand total -- so the annotated image is self-contained.

    `scale` multiplies the overall legend size (font, swatches, padding,
    margin) -- 1.0 is the original size, 2.0 (the default) is twice as big.
    """
    h, w = img_bgr.shape[:2]
    margin = int(round(margin * scale))

    # Scale text/line sizing to the image's resolution (and the requested
    # `scale`) so the legend stays readable on both small previews and
    # large (tiled) images.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(1.1, h / 1000.0)) * scale
    thickness = max(1, int(round(font_scale * 2)))
    line_height = int(28 * font_scale) + int(round(10 * scale))
    swatch = int(16 * font_scale) + int(round(4 * scale))
    pad = int(10 * font_scale) + int(round(6 * scale))

    # Stable ordering: numeric class IDs first (sorted numerically), then
    # any non-numeric class names alphabetically.
    def _sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    class_keys = sorted(counts.keys(), key=_sort_key)
    lines = [f"Class {k}: {counts[k]}" for k in class_keys]
    lines.append(f"Total: {total}")

    text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    text_w = max((sz[0] for sz in text_sizes), default=0)
    box_w = text_w + swatch + pad * 3
    box_h = line_height * len(lines) + pad * 2

    x0 = margin
    y0 = h - margin - box_h
    x1 = x0 + box_w
    y1 = h - margin

    # Clip to image bounds (defensive, in case the legend would be taller
    # than the image itself on a tiny thumbnail).
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)

    # Semi-transparent dark background so the legend stays readable over
    # any part of the photo.
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, dst=img_bgr)
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    ty = y0 + pad + line_height - int(6 * font_scale)
    for i, k in enumerate(class_keys):
        color = colors.get(k, (34, 197, 94))
        sx0 = x0 + pad
        sy0 = ty - swatch + int(4 * font_scale)
        cv2.rectangle(img_bgr, (sx0, sy0), (sx0 + swatch, sy0 + swatch), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            img_bgr, lines[i], (sx0 + swatch + pad, ty),
            font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
        )
        ty += line_height

    # Total line, same style, drawn last (bottom of the box).
    cv2.putText(
        img_bgr, lines[-1], (x0 + pad, ty),
        font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
    )

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
    log.info("run_inference: requested device=%r resolved device=%r", device, dev)

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
            annotated = _draw_legend_bgr(annotated, _counts_from_dets(dets), total=len(dets))
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
        annotated = _draw_legend_bgr(annotated, _counts_from_dets(merged), total=len(merged))

    t1 = time.time()
    return _package(merged, int((t1 - t0) * 1000), annotated)