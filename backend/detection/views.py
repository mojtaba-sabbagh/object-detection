# detection/views.py
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .detector import run_inference

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

