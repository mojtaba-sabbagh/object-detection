import React, { useRef, useState } from 'react';
import OverlayBoxes from './OverlayBoxes';
import type { Detection } from '../types';

interface Props {
  beforeSrc?: string | null;                        // local preview (File URL) or data URL
  afterAnnotatedSrc?: string | null;                // "data:image/jpeg;base64,..." from API
  detections?: Detection[];
  original?: { width: number; height: number } | undefined;
  title?: string;
}

export default function BeforeAfter({
  beforeSrc,
  afterAnnotatedSrc,
  detections = [],
  original,
  title,
}: Props) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [tick, setTick] = useState(0);

  // 'After' strategy:
  // 1) If annotated image provided by backend -> show that.
  // 2) Else overlay boxes over the 'before' image to simulate 'after'.
  const afterSrc = afterAnnotatedSrc || beforeSrc;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {/* BEFORE */}
      <div>
        {title && <div className="text-xs text-slate-500 mb-1">{title} — Before</div>}
        <div className="relative inline-block max-w-full">
          {beforeSrc ? (
            <img
              src={beforeSrc}
              alt="before"
              className="block max-h-[40vh] max-w-full object-contain rounded-lg border"
            />
          ) : (
            <div className="text-xs text-slate-500">No preview available</div>
          )}
        </div>
      </div>

      {/* AFTER */}
      <div>
        {title && <div className="text-xs text-slate-500 mb-1">{title} — After</div>}
        <div className="relative inline-block max-w-full">
          {afterSrc ? (
            <>
              <img
                ref={imgRef}
                src={afterSrc}
                alt="after"
                className="block max-h-[40vh] max-w-full object-contain rounded-lg border"
                onLoad={() => setTick(t => t + 1)}
              />
              {!afterAnnotatedSrc && (
                <OverlayBoxes
                  imgEl={imgRef.current}
                  original={original}
                  detections={detections}
                  readyTick={tick}
                />
              )}
            </>
          ) : (
            <div className="text-xs text-slate-500">No result image</div>
          )}
        </div>
      </div>
    </div>
  );
}
