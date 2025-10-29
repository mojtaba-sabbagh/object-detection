# detection/views.py
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .detector import run_inference
from .utils import merge_counts
import io, zipfile, time
from .models import YoloModel
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

@method_decorator(csrf_exempt, name='dispatch')
class CurrentModelView(APIView):
    def get(self, request):
        m = YoloModel.objects.filter(is_active=True).first()
        if not m:
            from django.conf import settings
            p = getattr(settings, "YOLO_MODEL_PATH", None)
            if not p:
                return Response({"active": None}, status=status.HTTP_200_OK)
            return Response({
                "active": {
                    "id": -1,
                    "name": p.split("/")[-1],
                    "date_built": None,
                    "base_model": None,
                    "num_params": None,
                    "map": None,
                    "map_5095": None,
                    "size": None,
                    "weights_path": p,
                }
            }, status=status.HTTP_200_OK)

        return Response({
            "active": {
                "id": m.id,
                "name": m.name,
                "date_built": m.date_built,
                "base_model": m.base_model,
                "num_params": m.num_params,
                "map": m.map,
                "map_5095": m.map_5095,
                "size": m.size,
                "weights_path": m.weights_path,
            }
        }, status=status.HTTP_200_OK)

@method_decorator(csrf_exempt, name='dispatch')
class BatchDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        conf = float(request.query_params.get('conf', 0.25))
        imgsz = int(request.query_params.get('imgsz', 640))
        device = request.query_params.get('device', getattr(settings, 'DEFAULT_YOLO_DEVICE', 'auto'))
        annotate = str(request.query_params.get('annotate', '0')).lower() in ('1','true','yes')

        # NEW: tiling knobs
        tile = request.query_params.get('tile', 'auto')
        tile_size = int(request.query_params.get('tile_size', 640))
        overlap = float(request.query_params.get('overlap', 0.20))
        nms_iou = float(request.query_params.get('nms_iou', 0.50))

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
                            data = io.BytesIO(f.read())  # wrap bytes for detector
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
                res = run_inference(
                    f, conf=conf, imgsz=imgsz, device=device, annotate=annotate,
                    tile=tile, tile_size=tile_size, overlap=overlap, nms_iou=nms_iou
                )
                merge_counts(collection_counts, res.get('counts', {}))
                total_objects += int(res.get('total', 0))
                items.append({
                    "name": name,
                    "image": res.get("image"),
                    "inference_ms": res.get("inference_ms"),
                    "counts": res.get("counts"),
                    "total": res.get("total"),
                    "detections": res.get("detections"),
                    "image_b64": res.get("image_b64"),
                })
            except Exception as e:
                items.append({"name": name, "error": str(e)})


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

@method_decorator(csrf_exempt, name='dispatch')
class DetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": 'No image uploaded. Use form field name "image".'},
                            status=status.HTTP_400_BAD_REQUEST)

        conf = float(request.query_params.get('conf', 0.25))
        imgsz = int(request.query_params.get('imgsz', 640))
        device = request.query_params.get('device', getattr(settings, 'DEFAULT_YOLO_DEVICE', 'auto'))
        annotate = str(request.query_params.get('annotate', '0')).lower() in ('1','true','yes')

        # NEW: tiling knobs
        tile = request.query_params.get('tile', 'auto')
        tile_size = int(request.query_params.get('tile_size', 640))
        overlap = float(request.query_params.get('overlap', 0.20))
        nms_iou = float(request.query_params.get('nms_iou', 0.50))

        try:
            data = run_inference(
                request.FILES['image'],
                conf=conf, imgsz=imgsz, device=device, annotate=annotate,
                tile=tile, tile_size=tile_size, overlap=overlap, nms_iou=nms_iou
            )
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            payload = {"error": "Model inference failed"}
            if getattr(settings, "DEBUG", False):
                payload["detail"] = str(e)
            return Response(payload, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class HealthView(APIView):
    def get(self, request):
        return Response({'status': 'ok'})
