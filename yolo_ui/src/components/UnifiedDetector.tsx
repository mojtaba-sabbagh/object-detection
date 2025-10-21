import React, { useEffect, useRef, useState, useMemo } from 'react';
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
  image_b64?: string;
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

function isZipFile(f: File) { return f.type === 'application/zip' || /\.zip$/i.test(f.name); }
function isImageFile(f: File) { return /^image\//.test(f.type) || /\.(png|jpe?g|bmp|webp)$/i.test(f.name); }

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

/** Helper: right-side AFTER image with overlay tooltips (class+confidence); borders optional */
function AfterWithOverlay({
  src,
  detections,
  original,
  showBorders,
  maxHeight = '40vh',
}: {
  src: string;
  detections: Detection[] | undefined;
  original: { width: number; height: number } | undefined;
  showBorders: boolean;
  maxHeight?: string;
}) {
  const ref = useRef<HTMLImageElement | null>(null);
  const [tick, setTick] = useState(0);
  return (
    <div className="relative inline-block">
      <img
        ref={ref}
        src={src}
        alt="after"
        className="block max-w-full object-contain rounded-lg border"
        style={{ maxHeight }}
        onLoad={() => setTick(t => t + 1)}
      />
      <OverlayBoxes
        imgEl={ref.current}
        original={original}
        detections={detections ?? []}
        readyTick={tick}
        showBorders={showBorders}
        showTooltip={true}   // class + confidence
      />
    </div>
  );
}

export default function UnifiedDetector(): JSX.Element {
  // state
  const [files, setFiles] = useState<FileList | null>(null);
  const [previews, setPreviews] = useState<Map<string, string>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [singleResult, setSingleResult] = useState<ApiResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);

  const [preview, setPreview] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgReady, setImgReady] = useState(0);

  const [currentModel, setCurrentModel] = useState<CurrentModel>(null);

  // pagination for batch
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const totalPages = useMemo(
    () => (batchResult ? Math.max(1, Math.ceil(batchResult.items.length / pageSize)) : 1),
    [batchResult]
  );
  const pagedItems = useMemo(() => {
    if (!batchResult) return [];
    const start = (page - 1) * pageSize;
    return batchResult.items.slice(start, start + pageSize);
  }, [batchResult, page]);

  // derived batch stats
  const successItems = useMemo(() => batchResult?.items.filter(i => !i.error) ?? [], [batchResult]);
  const failedItems  = useMemo(() => batchResult?.items.filter(i => !!i.error) ?? [], [batchResult]);
  const zeroDetItems = useMemo(() => successItems.filter(i => (i.total ?? 0) === 0), [successItems]);

  const timingStats = useMemo(() => {
    const times = successItems.map(i => i.inference_ms || 0).filter(n => Number.isFinite(n));
    if (times.length === 0) return { avg: 0, min: 0, max: 0 };
    const sum = times.reduce((a, b) => a + b, 0);
    return { avg: Math.round(sum / times.length), min: Math.min(...times), max: Math.max(...times) };
  }, [successItems]);

  // effects
  useEffect(() => {
    (async () => {
      try {
        const { data } = await axios.get<{ active: CurrentModel }>(`${API_BASE_URL}/api/model/current/`);
        setCurrentModel(data.active);
      } catch { setCurrentModel(null); }
    })();
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
      previews.forEach(url => URL.revokeObjectURL(url));
    };
  }, [preview, previews]);

  // file change
  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files;
    setFiles(f);
    setError('');
    setBatchResult(null);
    setSingleResult(null);
    setPreview(null);
    setPage(1);

    const map = new Map<string, string>();
    if (f) {
      Array.from(f).forEach(file => {
        if (isImageFile(file)) {
          map.set(file.name, URL.createObjectURL(file));
        }
      });
    }
    setPreviews(map);

    if (f && f.length === 1) {
      const first = f.item(0);
      if (first && isImageFile(first)) {
        setPreview(URL.createObjectURL(first));
      }
    }
  }

  // submit
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
      if (files.length === 1 && !hasZip) {
        // SINGLE
        const first = files.item(0);
        if (!first || !isImageFile(first)) {
          setError('Please select a valid image.');
          setLoading(false);
          return;
        }
        const form = new FormData();
        form.append('image', first, first.name);        
        const { data } = await axios.post<ApiResult>(
          `${API_BASE_URL}/api/detect/?conf=0.25&imgsz=640&annotate=1`,
          form, { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        setSingleResult(data);
        setBatchResult(null);
      } else if (hasZip && files.length === 1) {
        // BATCH via ZIP
        const first = files.item(0);
        if (!first) {
          setError('ZIP file missing.');
          setLoading(false);
          return;
        }
        const form = new FormData();
        form.append('zip', first, first.name);
        const { data } = await axios.post<BatchResponse>(
          `${API_BASE_URL}/api/detect/batch/?conf=0.25&imgsz=640&annotate=1`,
          form, { headers: { 'Content-Type': 'multipart/form-data' } }
        );
        setBatchResult(data);
        setSingleResult(null);
      } else if (allImages) {
        // BATCH via multiple images
        const form = new FormData();
        Array.from(files).forEach(f => form.append('images', f));
        const { data } = await axios.post<BatchResponse>(
          `${API_BASE_URL}/api/detect/batch/?conf=0.25&imgsz=640&annotate=1`,
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

  // render
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

        {/* Preview (Before) while browsing */}
        {files && files.length > 0 && !singleResult && !batchResult && (
          <section className="space-y-4">
            <h3 className="text-lg font-semibold">Preview (Before)</h3>

            {files.length === 1 && preview && (
              <div className="w-full flex justify-center">
                <img
                  src={preview}
                  alt="before"
                  className="block max-h-[60vh] max-w-full object-contain rounded-lg border"
                />
              </div>
            )}

            {files.length > 1 && (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                {Array.from(files).map((f) => {
                  const url = previews.get(f.name);
                  return (
                    <div key={f.name} className="border rounded-lg p-1 bg-white">
                      {url ? (
                        <img src={url} alt={f.name} className="h-32 w-full object-cover rounded" />
                      ) : (
                        <div className="h-32 w-full grid place-items-center text-xs text-slate-500">No preview</div>
                      )}
                      <div className="mt-1 text-[11px] truncate text-slate-600">{f.name}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}
      </section>

      {/* SINGLE RESULT (side-by-side: Original | Annotated) */}
      {singleResult && (
        <section className="space-y-4">
          <h3 className="text-lg font-semibold">Result</h3>

          <div className="grid gap-4 md:grid-cols-2">
            {/* Left: Original (Before) */}
            <div>
              <div className="text-xs text-slate-500 mb-1">Original</div>
              <div className="relative inline-block">
                <img
                  src={preview || ''}
                  alt="original"
                  className="block max-w-full object-contain rounded-lg border"
                  style={{ maxHeight: '70vh' }}
                />
              </div>
            </div>

            {/* Right: Annotated (After) with tooltips; hide borders if server drew them */}
            <div>
              <div className="text-xs text-slate-500 mb-1">Annotated</div>
              <div className="relative inline-block">
                <img
                  ref={imgRef}
                  src={singleResult.image_b64 ? `data:image/jpeg;base64,${singleResult.image_b64}` : (preview ?? '')}
                  alt="annotated"
                  className="block max-w-full object-contain rounded-lg border"
                  style={{ maxHeight: '70vh' }}
                  onLoad={() => setImgReady((n) => n + 1)}
                />
                <OverlayBoxes
                  imgEl={imgRef.current}
                  original={singleResult.image}
                  detections={singleResult.detections ?? []}
                  readyTick={imgReady}
                  showBorders={!singleResult.image_b64}
                  showTooltip={true}
                />
              </div>
            </div>
          </div>

          {/* Compact single-image stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-2.5"><div className="text-[12px] text-slate-500">Width</div><div className="font-mono text-sm">{singleResult.image?.width ?? '—'}</div></div>
            <div className="card p-2.5"><div className="text-[12px] text-slate-500">Height</div><div className="font-mono text-sm">{singleResult.image?.height ?? '—'}</div></div>
            <div className="card p-2.5"><div className="text-[12px] text-slate-500">Time</div><div className="font-mono text-sm whitespace-nowrap">{singleResult.inference_ms} ms</div></div>
            <div className="card p-2.5"><div className="text-[12px] text-slate-500">Total</div><div className="font-mono text-sm">{singleResult.total}</div></div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Per-class</h4>
            <table className="w-full border rounded-xl text-xs leading-tight">
              <thead><tr className="bg-slate-50 text-left"><th className="px-2 py-1">Class</th><th className="px-2 py-1">Count</th></tr></thead>
              <tbody>
                {withAllClasses(singleResult.counts).map(([k, v]) => (
                  <tr key={k} className="border-t">
                    <td className={`px-2 py-1 font-mono ${classColor(k)}`}>{k}</td>
                    <td className="px-2 py-1 font-mono">{v}</td>
                  </tr>
                ))}
                <tr className="border-t font-semibold bg-slate-50/50">
                  <td className="px-2 py-1">Total</td>
                  <td className="px-2 py-1 font-mono">{sumCounts(singleResult.counts)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

        {/* BATCH RESULT with PAGINATION + STATS + side-by-side image rows */}
        {batchResult && (
        <section className="space-y-6">
            <h3 className="text-lg font-semibold">Batch results</h3>

            {/* Batch health */}
            <div>
            <h4 className="font-medium mb-2">Batch health</h4>
            <table className="w-full border rounded-xl text-xs leading-tight">
                <thead>
                <tr className="bg-slate-50 text-left">
                    <th className="px-2 py-1">Metric</th>
                    <th className="px-2 py-1">Value</th>
                </tr>
                </thead>
                <tbody>
                <tr className="border-t"><td className="px-2 py-1">Images uploaded</td><td className="px-2 py-1 font-mono">{batchResult.params?.images}</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Processed (success)</td><td className="px-2 py-1 font-mono">{successItems.length}</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Failed</td><td className="px-2 py-1 font-mono">{failedItems.length}</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Zero detections</td><td className="px-2 py-1 font-mono">{zeroDetItems.length}</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Total objects</td><td className="px-2 py-1 font-mono">{batchResult.collection.total}</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Total time</td><td className="px-2 py-1 font-mono">{batchResult.collection.inference_ms_total} ms</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Avg per-image time</td><td className="px-2 py-1 font-mono">{timingStats.avg} ms</td></tr>
                <tr className="border-t"><td className="px-2 py-1">Min / Max per-image time</td><td className="px-2 py-1 font-mono">{timingStats.min} / {timingStats.max} ms</td></tr>
                </tbody>
            </table>
            </div>

            {/* Collection per-class */}
            <div>
            <h4 className="font-medium mb-2">Collection per-class</h4>
            <table className="w-full border rounded-xl text-xs leading-tight">
                <thead><tr className="bg-slate-50 text-left"><th className="px-2 py-1">Class</th><th className="px-2 py-1">Count</th></tr></thead>
                <tbody>
                {withAllClasses(batchResult.collection.counts).map(([k, v]) => (
                    <tr key={k} className="border-t">
                    <td className={`px-2 py-1 font-mono ${classColor(k)}`}>{k}</td>
                    <td className="px-2 py-1 font-mono">{v}</td>
                    </tr>
                ))}
                <tr className="border-t font-semibold bg-slate-50/50">
                    <td className="px-2 py-1">Total</td>
                    <td className="px-2 py-1 font-mono">{sumCounts(batchResult.collection.counts)}</td>
                </tr>
                </tbody>
            </table>
            </div>

            {/* Failed images */}
            {failedItems.length > 0 && (
            <div>
                <h4 className="font-medium mb-2">Failed images</h4>
                <table className="w-full border rounded-xl text-xs leading-tight">
                <thead><tr className="bg-slate-50 text-left"><th className="px-2 py-1">Image</th><th className="px-2 py-1">Error</th></tr></thead>
                <tbody>
                    {failedItems.map((it, i) => (
                    <tr key={i} className="border-t">
                        <td className="px-2 py-1 font-mono">{it.name}</td>
                        <td className="px-2 py-1">{it.error}</td>
                    </tr>
                    ))}
                </tbody>
                </table>
            </div>
            )}

            {/* Per-image rows (Original | Annotated) */}
            <div className="space-y-4">
                {pagedItems.map((it, idx) => {
                const before = previews.get(it.name) || null;
                const after  = it.image_b64 ? `data:image/jpeg;base64,${it.image_b64}` : (before ?? '');

                // NEW: global index across the whole batch (not just the page)
                const globalIndex = (page - 1) * pageSize + idx + 1;

                return (
                    <div key={`${page}-${idx}`} className="card p-4">
                    <div className="flex items-center justify-between mb-2">
                        <div className="text-sm text-slate-700">
                        <span className="font-mono mr-2">#{globalIndex}</span>{it.name}
                        </div>
                        {it.image && (
                        <div className="text-xs text-slate-500">
                            {it.image.width}×{it.image.height}
                        </div>
                        )}
                    </div>
                    {it.error ? (
                    <div className="text-red-600 text-sm">Error: {it.error}</div>
                    ) : (
                    <>
                        <div className="grid gap-4 md:grid-cols-2">
                        {/* Left: Original */}
                        <div>
                            <div className="text-xs text-slate-500 mb-1">Original</div>
                            <div className="relative inline-block">
                            {before ? (
                                <img
                                src={before}
                                alt={`${it.name} original`}
                                className="block max-w-full object-contain rounded-lg border"
                                style={{ maxHeight: '40vh' }}
                                />
                            ) : (
                                <div className="h-40 grid place-items-center rounded-lg border text-xs text-slate-500 bg-white">
                                Original not available
                                </div>
                            )}
                            </div>
                        </div>

                        {/* Right: Annotated with tooltips */}
                        <div>
                            <div className="text-xs text-slate-500 mb-1">Annotated</div>
                            <div className="w-full flex justify-start">
                            <AfterWithOverlay
                                src={after}
                                detections={it.detections}
                                original={it.image}
                                showBorders={!it.image_b64}
                                maxHeight="40vh"
                            />
                            </div>
                        </div>
                        </div>

                        {/* Per-image per-class table */}
                        <table className="mt-3 w-full border rounded-xl text-xs leading-tight">
                        <thead><tr className="bg-slate-50 text-left"><th className="px-2 py-1">Class</th><th className="px-2 py-1">Count</th></tr></thead>
                        <tbody>
                            {withAllClasses(it.counts).map(([k, v]) => (
                            <tr key={k} className="border-t">
                                <td className={`px-2 py-1 font-mono ${classColor(k)}`}>{k}</td>
                                <td className="px-2 py-1 font-mono">{v}</td>
                            </tr>
                            ))}
                            <tr className="border-t font-semibold bg-slate-50/50">
                            <td className="px-2 py-1">Total</td>
                            <td className="px-2 py-1 font-mono">{sumCounts(it.counts)}</td>
                            </tr>
                        </tbody>
                        </table>

                        <div className="mt-2 text-[12px] text-slate-600">
                        time <span className="font-mono">{it.inference_ms}</span> ms · total{' '}
                        <span className="font-mono">{it.total}</span>
                        </div>
                    </>
                    )}
                </div>
                );
            })}
            </div>

            {/* Pagination — moved to the bottom */}
            <div className="flex items-center justify-between pt-2">
            <div className="text-sm text-slate-600">
                Page <span className="font-mono">{page}</span> / <span className="font-mono">{totalPages}</span>
            </div>
            <div className="flex gap-2">
                <button
                type="button"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                className="px-3 py-1.5 rounded border bg-white disabled:opacity-50"
                disabled={page <= 1}
                >
                Prev
                </button>
                <button
                type="button"
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                className="px-3 py-1.5 rounded border bg-white disabled:opacity-50"
                disabled={page >= totalPages}
                >
                Next
                </button>
            </div>
            </div>
        </section>
        )}

    </div>
  );
}
