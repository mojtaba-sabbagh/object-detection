// src/components/BatchUploader.tsx
import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

type Counts = Record<string, number>;
type Item = {
  name: string;
  image?: { width: number; height: number };
  inference_ms?: number;
  counts?: Counts;
  total?: number;
  error?: string;
};

const classColor = (k: string) =>
  k === '0' ? 'text-yellow-600'
: k === '1' ? 'text-orange-600'
: k === '2' ? 'text-black'
: 'text-emerald-700';


export default function BatchUploader() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [result, setResult] = useState<{
    params: any;
    items: Item[];
    collection: { counts: Counts; total: number; inference_ms_total: number };
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!files || files.length === 0) return setErr('Please select images.');
    const form = new FormData();
    Array.from(files).forEach(f => form.append('images', f));
    setLoading(true); setErr('');
    try {
      const { data } = await axios.post(
        `${API_BASE_URL}/api/detect/batch/?conf=0.25&imgsz=640`,
        form, { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(data);
    } catch (e: any) {
      setErr(e?.response?.data?.error || 'Batch request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card p-5 space-y-4">
      <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-[1fr_auto] items-center">
        <input type="file" accept="image/*" multiple onChange={e => setFiles(e.target.files)} className="block w-full rounded-xl border p-2" />
        <button disabled={loading} className="px-4 py-2 rounded-xl bg-emerald-600 text-white disabled:opacity-50">
          {loading ? 'Processing…' : 'Run Batch Detection'}
        </button>
      </form>

      {err && <div className="text-red-600">{err}</div>}

      {result && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold">Collection summary</h3>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="card p-3"><div className="text-sm text-slate-500">Images</div><div className="font-mono">{result.params?.images}</div></div>
              <div className="card p-3"><div className="text-sm text-slate-500">Total objects</div><div className="font-mono">{result.collection.total}</div></div>
              <div className="card p-3"><div className="text-sm text-slate-500">Total time</div><div className="font-mono">{result.collection.inference_ms_total} ms</div></div>
            </div>
            <table className="mt-4 w-full border rounded-xl text-sm">
              <thead><tr className="bg-slate-50 text-left"><th className="p-2">Class</th><th className="p-2">Count</th></tr></thead>
              <tbody>
              {Object.entries(result.collection.counts || {}).map(([k,v]) => (
                <tr key={k} className="border-t"><td className="p-2 font-mono">{k}</td><td className="p-2">{v}</td></tr>
              ))}
              </tbody>
            </table>
          </div>

            <div>
            <h3 className="text-lg font-semibold">Per-image results</h3>
            <div className="mt-2 space-y-3">
                {result.items.map((it, idx) => (
                <div key={idx} className="card p-4">
                    <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-700">{it.name}</div>
                    {it.image && (
                        <div className="text-xs text-slate-500">
                        {it.image.width}×{it.image.height}
                        </div>
                    )}
                    </div>

                    {it.error ? (
                    <div className="mt-2 text-red-600 text-sm">Error: {it.error}</div>
                    ) : (
                    <>
                        <div className="mt-1 text-sm text-slate-700">
                        time <span className="font-mono">{it.inference_ms}</span> ms · total{' '}
                        <span className="font-mono">{it.total}</span>
                        </div>

                        {/* per-image class stats */}
                        <table className="mt-3 w-full border rounded-xl text-sm border-separate border-spacing-y-2">
                        <thead>
                            <tr className="bg-slate-50 text-left">
                            <th className="p-2">Class</th>
                            <th className="p-2">Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(it.counts ?? {}).map(([k, v]) => (
                            <tr key={k} className="border-t">
                                <td className={`p-2 font-mono ${classColor(k)}`}>{k}</td>
                                <td className="p-2">{v}</td>
                            </tr>
                            ))}
                        </tbody>
                        </table>
                    </>
                    )}
                </div>
                ))}
            </div>
            </div>

        </div>
      )}
    </div>
  );
}
