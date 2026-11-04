import { useRef, useState } from "react";
import { previewUrl } from "../api";

export type Box = [number, number, number, number];

/** Draw one box prompt on a decoded source frame. Coordinates are source pixels (half-open). */
export default function BoxPrompt({
  runId, frameCount, width, height, frame, setFrame, box, setBox,
}: {
  runId: string;
  frameCount: number;
  width: number;
  height: number;
  frame: number;
  setFrame: (f: number) => void;
  box: Box | null;
  setBox: (b: Box | null) => void;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const [cur, setCur] = useState<[number, number] | null>(null);
  const [pending, setPending] = useState(frame);

  const toPx = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [
      Math.max(0, Math.min(width, ((e.clientX - r.left) / r.width) * width)),
      Math.max(0, Math.min(height, ((e.clientY - r.top) / r.height) * height)),
    ] as [number, number];
  };
  const shown: Box | null =
    drag && cur
      ? [Math.min(drag[0], cur[0]), Math.min(drag[1], cur[1]), Math.max(drag[0], cur[0]), Math.max(drag[1], cur[1])]
      : box;

  return (
    <div className="col">
      <div className="media" style={{ cursor: "crosshair" }}>
        <img src={previewUrl(runId, frame)} alt={`source frame ${frame}`} draggable={false} />
        <svg
          ref={ref}
          className="overlay"
          style={{ pointerEvents: "auto" }}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture?.(e.pointerId);
            const p = toPx(e);
            setDrag(p);
            setCur(p);
          }}
          onPointerMove={(e) => drag && setCur(toPx(e))}
          onPointerUp={() => {
            if (shown && shown[2] - shown[0] >= 3 && shown[3] - shown[1] >= 3)
              setBox(shown.map((v) => Math.round(v)) as Box);
            setDrag(null);
            setCur(null);
          }}
        >
          {shown && (
            <rect x={shown[0]} y={shown[1]} width={shown[2] - shown[0]} height={shown[3] - shown[1]} fill="rgba(57,135,229,0.12)"
              stroke="#3987e5" strokeWidth={Math.max(1, width / 500)} vectorEffect="non-scaling-stroke" />
          )}
        </svg>
        <span className="tag">frame {frame}</span>
      </div>
      <div className="row">
        <span className="tiny muted">Prompt frame</span>
        <input type="range" min={0} max={frameCount - 1} value={pending} onChange={(e) => setPending(+e.target.value)}
          onPointerUp={() => setFrame(pending)} onKeyUp={() => setFrame(pending)} style={{ flex: 1 }} aria-label="prompt frame" />
        <span className="timecode">{pending}</span>
      </div>
      <div className="row tiny muted">
        {box ? <span className="mono">box [{box.join(", ")}] on frame {frame}</span> : <span>Drag on the frame to draw a box around the object.</span>}
        <span className="grow" />
        {box && (
          <button className="btn sm ghost" onClick={() => setBox(null)}>
            clear box
          </button>
        )}
      </div>
    </div>
  );
}
