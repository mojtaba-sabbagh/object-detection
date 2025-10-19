# detection/views.py
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .detector import run_inference
from .utils import merge_counts
import io, zipfile, time

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class BatchDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        conf = float(request.query_params.get('conf', 0.25))
        imgsz = int(request.query_params.get('imgsz', 640))
        device = request.query_params.get('device', getattr(settings, 'DEFAULT_YOLO_DEVICE', 'auto'))

        items = []
        collection_counts = {}
        total_objects = 0
        t0 = time.time()

        # Case A) multiple files via <input multiple name="images">
        files = list(request.FILES.getlist('images'))

        # Case B) one ZIP via field name "zip"
        zip_file = request.FILES.get('zip')
        if zip_file and not files:
            try:
                with zipfile.ZipFile(zip_file) as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        name_lower = info.filename.lower()
                        if not any(name_lower.endswith(ext) for ext in IMAGE_EXTS):
                            continue
                        with z.open(info) as f:
                            # wrap as bytes for detector
                            data = io.BytesIO(f.read())
                            data.name = info.filename
                            files.append(data)
            except Exception as e:
                return Response(
                    {"error": "Invalid ZIP archive", "detail": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )

        if not files:
            return Response(
                {"error": 'Upload images under field "images" (multiple allowed) or a ZIP under field "zip".'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Hard limit (avoid accidental huge batches)
        max_imgs = int(request.query_params.get('max', 200))
        files = files[:max_imgs]

        for f in files:
            name = getattr(f, 'name', 'image')
            try:
                res = run_inference(f, conf=conf, imgsz=imgsz, device=device)
                merge_counts(collection_counts, res.get('counts', {}))
                total_objects += int(res.get('total', 0))
                items.append({
                    "name": name,
                    "image": res.get("image"),
                    "inference_ms": res.get("inference_ms"),
                    "counts": res.get("counts"),
                    "total": res.get("total"),
                    "detections": res.get("detections"),
                })
            except Exception as e:
                items.append({
                    "name": name,
                    "error": str(e),
                })

        t1 = time.time()
        return Response({
            "params": {"conf": conf, "imgsz": imgsz, "device": device, "images": len(files)},
            "items": items,  # per-image
            "collection": {
                "counts": collection_counts,
                "total": total_objects,
                "inference_ms_total": int((t1 - t0) * 1000),
            }
        }, status=status.HTTP_200_OK)

class DetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": 'No image uploaded. Use form field name "image".'},
                            status=status.HTTP_400_BAD_REQUEST)

        conf = float(request.query_params.get('conf', 0.25))
        imgsz = int(request.query_params.get('imgsz', 640))
        device = request.query_params.get('device', getattr(settings, 'DEFAULT_YOLO_DEVICE', 'auto'))

        try:
            data = run_inference(request.FILES['image'], conf=conf, imgsz=imgsz, device=device)
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            payload = {"error": "Model inference failed"}
            if getattr(settings, "DEBUG", False):
                payload["detail"] = str(e)
            return Response(payload, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HealthView(APIView):
    def get(self, request):
        return Response({'status': 'ok'})

