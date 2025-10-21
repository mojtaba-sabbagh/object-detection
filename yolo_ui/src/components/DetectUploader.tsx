import React, { useMemo, useRef, useState, useEffect } from 'react';
import axios from 'axios';
import type { ApiResult } from '../types';
import OverlayBoxes from './OverlayBoxes';

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? '';

export default function DetectUploader(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<ApiResult | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>): void {
    const f = e.target.files?.[0] ?? null;
    setResult(null);
    setError('');
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function onSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault();
    if (!file) return setError('Please select an image first.');

    const form = new FormData();
    form.append('image', file);

    setLoading(true);
    setError('');
    try {
      const { data } = await axios.post<ApiResult>(
        `${API_BASE_URL}/api/detect/?conf=0.25&imgsz=640`,
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(data);
      setTimeout(() => containerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    } catch (err: unknown) {
      const msg = (err as any)?.response?.data?.error ?? 'Upload failed';
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  }

  const countsRows = useMemo(
    () => (Object.entries(result?.counts ?? {}).length ? Object.entries(result!.counts) : [['—', 0]] as any),
    [result]
  );

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <header className="mb-6 flex items-center justify-between">
        <h1 className="w-full text-2xl text-center font-yekan md:text-3xl font-bold tracking-tight">سامانه تشخیص و شمارش پسیل پسته</h1>
        <a
          className="text-sm text-emerald-700 hover:text-emerald-900 underline"
          href="https://github.com/ultralytics/ultralytics"
          target="_blank"
          rel="noreferrer"
        >
          Powered by Ultralytics
        </a>
      </header>

      <section className="card p-5 mt-10">
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-[1fr_auto] items-center">
          <input
            type="file"
            accept="image/*"
            onChange={onFileChange}
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

        {preview && (
            <div className="mt-8">
              <h2 className="card-header mb-2">Preview</h2>

              {/* center the image; wrapper sizes to the image, not full width */}
              <div className="w-full flex justify-center">
                <div className="relative inline-block">
                  <img
                    ref={imgRef}
                    src={preview}
                    alt="preview"
                    className="block max-h-[70vh] max-w-full object-contain"
                  />
                  {/* Overlay sized to the image element */}
                  <OverlayBoxes
                    imgEl={imgRef.current}
                    original={result?.image}
                    detections={result?.detections ?? []}
                  />
                </div>
              </div>
            </div>
          )}

      </section>

      {error && <div className="mt-4 text-red-600">{error}</div>}

      {result && (
        <section ref={containerRef} className="mt-8 grid gap-6 md:grid-cols-5">
          <div className="md:col-span-2 card p-5">
            <h2 className="text-lg font-semibold mb-3">Summary</h2>
            <div className="grid grid-cols-2 gap-3">
              <div className="card p-3">
                <div className="card-header">Width</div>
                <div className="card-value">{result.image?.width ?? '—'}</div>
              </div>
              <div className="card p-3">
                <div className="card-header">Height</div>
                <div className="card-value">{result.image?.height ?? '—'}</div>
              </div>
              <div className="card p-3">
                <div className="card-header">Time</div>
                <div className="card-value">{result.inference_ms} ms</div>
              </div>
              <div className="card p-3">
                <div className="card-header">Total</div>
                <div className="card-value">{result.total}</div>
              </div>
            </div>

            <div className="mt-5">
              <h3 className="font-medium mb-2">Per-class counts</h3>
              <table className="w-full overflow-hidden rounded-xl border text-sm">
                <thead>
                  <tr className="bg-slate-50 text-left">
                    <th className="p-2">Class</th>
                    <th className="p-2">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {countsRows.map(([cls, count]: [string, number]) => (
                    <tr key={String(cls)} className="border-t">
                      <td className="p-2 font-mono">{String(cls)}</td>
                      <td className="p-2">{String(count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="md:col-span-3 card p-5">
            <h3 className="text-lg font-semibold mb-3">Detections</h3>
            <div className="grid gap-3 sm:grid-cols-2">
              {(result.detections ?? []).map((d, idx) => (
                <div key={idx} className="card p-3">
                  <div className="text-xs text-slate-500">#{idx + 1}</div>
                  <div className="font-mono text-sm">
                    {d.class_name} — conf {d.confidence.toFixed(3)}
                  </div>
                  <div className="text-xs text-slate-600">
                    bbox: x1 {d.bbox.x1.toFixed(1)}, y1 {d.bbox.y1.toFixed(1)},
                    x2 {d.bbox.x2.toFixed(1)}, y2 {d.bbox.y2.toFixed(1)}
                  </div>
                </div>
              ))}
              {(!result.detections || result.detections.length === 0) && (
                <div className="text-slate-600">No objects detected.</div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
