export interface BBox {
  x1: number; y1: number; x2: number; y2: number;
  width: number; height: number;
}
export interface Detection {
  bbox: BBox;
  class_id: number;
  class_name: string;
  confidence: number;
}
export interface ApiResult {
  image?: { width: number; height: number };
  inference_ms: number;
  detections: Detection[];
  counts: Record<string, number>;
  total: number;
  image_b64?: string; // optional if backend returns annotated image
}
