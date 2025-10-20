import React, { useEffect, useMemo, useState } from 'react';
import type { Detection } from '../types';

interface Props {
  imgEl: HTMLImageElement | null;
  original?: { width: number; height: number } | undefined;
  detections: Detection[];
  readyTick?: number; // bumps when the <img> fires onLoad
}

export default function OverlayBoxes({ imgEl, original, detections, readyTick = 0 }: Props): JSX.Element | null {
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
    // Re-run when imgEl changes or image finishes loading or detections update
  }, [imgEl, readyTick]);

  useEffect(() => {
    // When detections change, ensure overlay dimensions are still correct
    updateDims();
  }, [detections]);

  const { scaleX, scaleY } = useMemo(() => {
    if (!original?.width || !original?.height || !dims.w || !dims.h) {
      return { scaleX: 1, scaleY: 1 };
    }
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
      className="absolute left-0 top-0 z-10"  // ← ensure overlay is above the <img>
      style={{ width: dims.w, height: dims.h }}
    >
      {detections.map((d, i) => {
        const x = d.bbox.x1 * scaleX;
        const y = d.bbox.y1 * scaleY;
        const w = d.bbox.width * scaleX;
        const h = d.bbox.height * scaleY;
        const borderClass = colorClassFor(d.class_name);

        return (
          <div
            key={i}
            title={d.class_name}
            className={`absolute rounded-md shadow-sm border-2 ${borderClass}`}
            style={{ left: x, top: y, width: w, height: h, pointerEvents: 'auto' }}
            aria-label={d.class_name}
          />
        );
      })}
    </div>
  );
}
