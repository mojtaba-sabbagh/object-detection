import React from 'react';
import DetectUploader from './components/DetectUploader';
import BatchUploader from './components/BatchUploader';

export default function App(): JSX.Element {
  return (
    <div className="min-h-screen bg-slate-50 font-yekan">
      <header className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">YOLO Object Detection</h1>
        <p className="text-slate-600 mt-1">
          Upload a single image for detection, or run a batch to get per-image and collection stats.
        </p>

        {/* Quick anchors */}
        <nav className="mt-4 flex gap-4 text-sm">
          <a href="#single" className="text-emerald-700 hover:underline">Single image</a>
          <a href="#batch" className="text-emerald-700 hover:underline">Batch</a>
        </nav>
      </header>

      <main className="max-w-6xl mx-auto p-6 pt-0 space-y-10">
        {/* Single uploader section */}
        <section id="single" className="space-y-4">
          <h2 className="text-lg font-semibold">Single image</h2>
          <DetectUploader />
        </section>

        {/* Batch uploader section */}
        <section id="batch" className="space-y-4">
          <h2 className="text-lg font-semibold">Batch</h2>
          <BatchUploader />
        </section>
      </main>
    </div>
  );
}
