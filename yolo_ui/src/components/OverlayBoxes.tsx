// src/components/OverlayBoxes.tsx
import React, { useEffect, useMemo, useState } from 'react';
import type { Detection } from '../types';

interface Props {
  imgEl: HTMLImageElement | null;
  original?: { width: number; height: number } | undefined;
  detections: Detection[];
  readyTick?: number;
  showBorders?: boolean;   // draw borders or not (default true)
  showTooltip?: boolean;   // show class + confidence on hover (default true)
}

export default function OverlayBoxes({
  imgEl,
  original,
  detections,
  readyTick = 0,
  showBorders = true,
  showTooltip = true,
}: Props): JSX.Element | null {
  const [dims, setDims] = useState({ w: 0, h: 0 });

  const updateDims = () => {
    if (!imgEl) return;
    const w = imgEl.clientWidth || imgEl.naturalWidth || 0;
    const h = imgEl.clientHeight || imgEl.naturalHeight || 0;
    setDims({ w, h });
  };

  useEffect(() => {
    updateDims();
    window.addEventListener('resize', updateDims);
    return () => window.removeEventListener('resize', updateDims);
  }, [imgEl, readyTick]);

  useEffect(() => { updateDims(); }, [detections]);

  const { scaleX, scaleY } = useMemo(() => {
    if (!original?.width || !original?.height || !dims.w || !dims.h) return { scaleX: 1, scaleY: 1 };
    return { scaleX: dims.w / original.width, scaleY: dims.h / original.height };
  }, [original?.width, original?.height, dims.w, dims.h]);

  if (!imgEl || !original) return null;

  const colorClassFor = (clsName: string) => {
    switch (clsName) {
      case '0': return 'border-yellow-400';
      case '1': return 'border-orange-500';
      case '2': return 'border-black';
      default:  return 'border-emerald-500';
    }
  };

  return (
    <div
      className="absolute left-0 top-0 z-10"
      style={{ width: dims.w, height: dims.h, pointerEvents: 'auto' }} // allow hover for tooltips
    >
      {detections.map((d, i) => {
        const x = d.bbox.x1 * scaleX;
        const y = d.bbox.y1 * scaleY;
        const w = d.bbox.width * scaleX;
        const h = d.bbox.height * scaleY;

        // Tooltip text: "<class> (XX.X%)"
        const tip =
          typeof d.confidence === 'number'
            ? `${d.class_name} ${(d.confidence * 100).toFixed(1)}%`
            : d.class_name;

        return (
          <div
            key={i}
            title={showTooltip ? tip : undefined}                 // ← tooltip only
            className={`absolute rounded-md shadow-sm ${showBorders ? `border-2 ${colorClassFor(d.class_name)}` : 'border-0'}`}
            style={{ left: x, top: y, width: w, height: h }}
            aria-label={d.class_name}
          />
        );
      })}
    </div>
  );
}
