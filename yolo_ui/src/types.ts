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
  /** EXIF orientation the uploaded file was stored with; 1 means already upright. */
  exif_orientation?: number;
  /** Upright, un-annotated copy of the original; only sent when it was rotated. */
  upright_b64?: string;
}
