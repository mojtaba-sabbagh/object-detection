import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import type { ApiResult, Detection } from '../types';
import OverlayBoxes from './OverlayBoxes';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const ALL_CLASSES = ['0', '1', '2'] as const;

type Counts = Record<string, number>;
type BatchItem = {
  name: string;
  image?: { width: number; height: number };
  inference_ms?: number;
  counts?: Counts;
  total?: number;
  detections?: Detection[];
  error?: string;
};
type BatchResponse = {
  params: { conf: number; imgsz: number; device: string; images: number };
  items: BatchItem[];
  collection: { counts: Counts; total: number; inference_ms_total: number };
};

const classColor = (k: string) =>
  k === '0' ? 'text-yellow-600'
: k === '1' ? 'text-orange-600'
: k === '2' ? 'text-black'
: 'text-emerald-700';

const withAllClasses = (counts?: Record<string, number>) =>
  ALL_CLASSES.map(k => [k, Number(counts?.[k] ?? 0)] as const);

const sumCounts = (counts?: Record<string, number>) =>
  Object.values(counts ?? {}).reduce((a, b) => a + Number(b || 0), 0);

function isZipFile(f: File) {
  return f.type === 'application/zip' || /\.zip$/i.test(f.name);
}
function isImageFile(f: File) {
  return /^image\//.test(f.type) || /\.(png|jpe?g|bmp|webp)$/i.test(f.name);
}

type CurrentModel = {
  id: number;
  name: string | null;
  date_built: string | null;
  base_model: string | null;
  num_params: number | null;
  map: number | null;
  map_5095: number | null;
  size: string | null;
  weights_path: string | null;
} | null;

export default function UnifiedDetector(): JSX.Element {
  // ---------------- state ----------------
  const [files, setFiles] = useState<FileList | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [singleResult, setSingleResult] = useState<ApiResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);

  const [preview, setPreview] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [currentModel, setCurrentModel] = useState<CurrentModel>(null);
  const [imgReady, setImgReady] = useState(0);

  // ---------------- effects ----------------
  useEffect(() => {
    (async () => {
      try {
        const { data } = await axios.get<{ active: CurrentModel }>(`${API_BASE_URL}/api/model/current/`);
        setCurrentModel(data.active);
      } catch {
        setCurrentModel(null);
      }
    })();
  }, []);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  // ---------------- handlers ----------------
  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files;
    setFiles(f);
    setError('');
    setBatchResult(null);
    setSingleResult(null);
    setPreview(null);

    if (f && f.length === 1 && isImageFile(f[0])) {
      setPreview(URL.createObjectURL(f[0]));
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');

    if (!files || files.length === 0) {
      setError('Please choose at least one image or a ZIP.');
      return;
    }

    const hasZip = Array.from(files).some(isZipFile);
    const allImages = Array.from(files).every(isImageFile);

    setLoading(true);
    try {
      if (files.length === 1 && isImageFile(files[0]) && !hasZip) {
        const form = new FormData();
        form.append('image', files[0]);
        const { data } = await axios.post<ApiResult>(
          `${API_BASE_URL}/api/detect/?conf=0.25&imgsz=640`,
          form, { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        setSingleResult(data);
        setBatchResult(null);
      } else if (hasZip && files.length === 1) {
        const form = new FormData();
        form.append('zip', files[0]);
        const { data } = await axios.post<BatchResponse>(
          `${API_BASE_URL}/api/detect/batch/?conf=0.25&imgsz=640`,
          form, { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        setBatchResult(data);
        setSingleResult(null);
      } else if (allImages) {
        const form = new FormData();
        Array.from(files).forEach(f => form.append('images', f));
        const { data } = await axios.post<BatchResponse>(
          `${API_BASE_URL}/api/detect/batch/?conf=0.25&imgsz=640`,
          form, { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        setBatchResult(data);
        setSingleResult(null);
      } else {
        setError('Please select only images, or a single ZIP of images.');
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Detection failed');
    } finally {
      setLoading(false);
    }
  }

  // ---------------- render ----------------
  return (
    <div className="space-y-8">
      <section className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Upload</h2>
          <div className="text-xs md:text-sm text-slate-600">
            Current model: <span className="font-semibold">{currentModel?.name ?? '—'}</span>
          </div>
        </div>

        <p className="text-sm text-slate-600">
          Choose a single image, multiple images, or one ZIP of images. Then click <em>Run Detection</em>.
        </p>

        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-[1fr_auto] items-center">
          <input
            type="file"
            onChange={onFileChange}
            multiple
            accept=".zip,image/*"
            className="block w-full rounded-xl border border-slate-300 bg-white
                       file:mr-3 file:rounded-lg file:border-0 file:bg-slate-900
                       file:px-4 file:py-2 file:text-white file:hover:bg-slate-800"
          />
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center rounded-xl bg-emerald-600
                       px-5 py-2.5 font-semibold text-white shadow-sm transition
                       disabled:opacity-50 hover:bg-emerald-700"
          >
            {loading ? 'Detecting…' : 'Run Detection'}
          </button>
        </form>

        {error && <div className="text-red-600">{error}</div>}
      </section>

      {/* SINGLE RESULT */}
      {singleResult && (
        <section className="space-y-6">
          <h3 className="text-lg font-semibold">Single image result</h3>

          {preview && (
            <div className="w-full flex justify-center">
              <div className="relative inline-block">
                <img
                  ref={imgRef}
                  src={preview}
                  alt="preview"
                  className="block max-h-[70vh] max-w-full object-contain"
                  onLoad={() => setImgReady((n) => n + 1)}   // ← ensure overlay recalculates after image layout
                />

                <OverlayBoxes
                    imgEl={imgRef.current}
                    original={singleResult.image}
                    detections={singleResult.detections ?? []}
                    readyTick={imgReady}
                />
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card p-3"><div className="text-sm text-slate-500">Width</div><div className="font-mono">{singleResult.image?.width ?? '—'}</div></div>
            <div className="card p-3"><div className="text-sm text-slate-500">Height</div><div className="font-mono">{singleResult.image?.height ?? '—'}</div></div>
            <div className="card p-3"><div className="text-sm text-slate-500">Time</div><div className="font-mono">{singleResult.inference_ms} ms</div></div>
            <div className="card p-3"><div className="text-sm text-slate-500">Total</div><div className="font-mono">{singleResult.total}</div></div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Per-class</h4>
            <table className="w-full rounded-xl border text-sm border-separate border-spacing-y-2">
              <thead><tr className="bg-slate-50 text-left"><th className="p-2">Class</th><th className="p-2">Count</th></tr></thead>
              <tbody>
                {withAllClasses(singleResult.counts).map(([k, v]) => (
                  <tr key={k} className="border-t">
                    <td className={`p-2 font-mono ${classColor(k)}`}>{k}</td>
                    <td className="p-2">{v}</td>
                  </tr>
                ))}
                <tr className="border-t font-semibold bg-slate-50/50">
                  <td className="p-2">Total</td>
                  <td className="p-2">{sumCounts(singleResult.counts)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* BATCH RESULT */}
      {batchResult && (
        <section className="space-y-6">
          <h3 className="text-lg font-semibold">Batch results</h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card p-3"><div className="text-sm text-slate-500">Images</div><div className="font-mono">{batchResult.params?.images}</div></div>
            <div className="card p-3"><div className="text-sm text-slate-500">Total objects</div><div className="font-mono">{batchResult.collection.total}</div></div>
            <div className="card p-3"><div className="text-sm text-slate-500">Total time</div><div className="font-mono">{batchResult.collection.inference_ms_total} ms</div></div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Collection per-class</h4>
            <table className="w-full rounded-xl border text-sm border-separate border-spacing-y-2">
              <thead><tr className="bg-slate-50 text-left"><th className="p-2">Class</th><th className="p-2">Count</th></tr></thead>
              <tbody>
                {withAllClasses(batchResult.collection.counts).map(([k, v]) => (
                  <tr key={k} className="border-t">
                    <td className={`p-2 font-mono ${classColor(k)}`}>{k}</td>
                    <td className="p-2">{v}</td>
                  </tr>
                ))}
                <tr className="border-t font-semibold bg-slate-50/50">
                  <td className="p-2">Total</td>
                  <td className="p-2">{sumCounts(batchResult.collection.counts)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <h4 className="font-medium mb-2">Per-image</h4>
            <div className="space-y-3">
              {batchResult.items.map((it, idx) => (
                <div key={idx} className="card p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-700">{it.name}</div>
                    {it.image && (<div className="text-xs text-slate-500">{it.image.width}×{it.image.height}</div>)}
                  </div>

                  {it.error ? (
                    <div className="mt-2 text-red-600 text-sm">Error: {it.error}</div>
                  ) : (
                    <>
                      <div className="mt-1 text-sm text-slate-700">
                        time <span className="font-mono">{it.inference_ms}</span> ms · total <span className="font-mono">{it.total}</span>
                      </div>
                      <table className="mt-3 w-full border rounded-xl text-sm border-separate border-spacing-y-2">
                        <thead><tr className="bg-slate-50 text-left"><th className="p-2">Class</th><th className="p-2">Count</th></tr></thead>
                        <tbody>
                          {withAllClasses(it.counts).map(([k, v]) => (
                            <tr key={k} className="border-t">
                              <td className={`p-2 font-mono ${classColor(k)}`}>{k}</td>
                              <td className="p-2">{v}</td>
                            </tr>
                          ))}
                          <tr className="border-t font-semibold bg-slate-50/50">
                            <td className="p-2">Total</td>
                            <td className="p-2">{sumCounts(it.counts)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
