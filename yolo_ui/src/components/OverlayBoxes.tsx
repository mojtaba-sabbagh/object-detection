import React, { useEffect, useMemo, useState } from 'react';
import type { Detection } from '../types';

interface Props {
  imgEl: HTMLImageElement | null;
  original?: { width: number; height: number } | undefined;
  detections: Detection[];
}

export default function OverlayBoxes({ imgEl, original, detections }: Props): JSX.Element | null {
  const [dims, setDims] = useState({ w: 0, h: 0 });

  useEffect(() => {
    if (!imgEl) return;
    const update = () => setDims({ w: imgEl.clientWidth, h: imgEl.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(imgEl);
    window.addEventListener('resize', update);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', update);
    };
  }, [imgEl]);

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
      className="absolute left-0 top-0"
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
