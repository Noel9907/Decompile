import { useEffect, useMemo, useRef, useState } from "react";
import { frameUrl } from "../api";
import type { Analysis, Perception } from "../types";

export interface Layers {
  masks: boolean;
  boxes: boolean;
  outline: boolean;
  path: boolean;
  program: boolean;
}

/** Exact decoded source frames of the reconstruction window with the measurement overlaid in source pixels. */
export default function FrameInspector({
  runId, rid, analysis, perception, index, setIndex, layers, width, height, autoPlay = false, compact = false,
}: {
  autoPlay?: boolean;
  compact?: boolean;
  runId: string;
  rid: string;
  analysis: Analysis;
  perception: Perception | null;
  index: number;
  setIndex: (i: number) => void;
  layers: Layers;
  width: number;
  height: number;
}) {
  const frames = analysis.frames;
  const pos = Math.max(0, frames.findIndex((f) => f.frame_index === index));
  const f = frames[pos];
  const [playing, setPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState(autoPlay ? 0.5 : 0.25);
  const posRef = useRef(pos);
  posRef.current = pos;

  const dt = useMemo(() => {
    const d = frames.slice(1).map((x, i) => x.t_s - frames[i].t_s).sort((a, b) => a - b);
    return d.length ? d[d.length >> 1] : 1 / 30;
  }, [frames]);

  useEffect(() => {
    if (!playing) return;
    const h = window.setInterval(() => {
      let next = posRef.current + 1;
      if (next >= frames.length) {
        if (!autoPlay) {
          setPlaying(false);
          return;
        }
        next = 0;
      }
      setIndex(frames[next].frame_index);
    }, (dt * 1000) / speed);
    return () => window.clearInterval(h);
  }, [playing, speed, dt, frames, setIndex, autoPlay]);

  // preload neighbouring frames so stepping is instant
  useEffect(() => {
    for (const k of [pos + 1, pos + 2, pos - 1]) {
      if (frames[k]) new Image().src = frameUrl(runId, rid, frames[k].frame_index);
    }
  }, [pos, frames, runId, rid]);

  const fit = analysis.fitted;
  const fk = fit ? fit.frame_index.indexOf(index) : -1;
  const sw = Math.max(1, width / 560);
  const objs = perception?.frames[String(index)] ?? [];
  const trail = frames.slice(0, pos + 1).filter((x) => x.status === "measured");
  const fitTrail = fit ? fit.frame_index.map((fi, k) => [fi, fit.x[k], fit.y[k]] as const).filter(([fi]) => fi <= index) : [];
  const medD = analysis.report.measurement.median_diameter_px ?? 20;

  return (
    <div>
      <div className="media">
        <img src={frameUrl(runId, rid, index)} alt={`decoded source frame ${index}`} />
        <svg className="overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {layers.masks &&
            objs.map((o) =>
              o.contours.map((c, k) => (
                <polygon
                  key={`${o.id}-${k}`}
                  points={c.map((p) => p.join(",")).join(" ")}
                  fill={o.id === perception?.selected_object ? "rgba(233,231,225,0.22)" : "rgba(125,127,128,0.14)"}
                  stroke={o.id === perception?.selected_object ? "rgba(233,231,225,0.9)" : "rgba(125,127,128,0.8)"}
                  strokeWidth={sw}
                  strokeDasharray={o.id === perception?.selected_object ? undefined : `${3 * sw} ${3 * sw}`}
                />
              )),
            )}
          {layers.boxes &&
            objs.map((o) => (
              <g key={`b${o.id}`}>
                <rect x={o.box[0]} y={o.box[1]} width={o.box[2] - o.box[0]} height={o.box[3] - o.box[1]} fill="none"
                  stroke={o.id === perception?.selected_object ? "#e9e7e1" : "#7d7f80"} strokeWidth={sw} strokeDasharray={`${4 * sw} ${3 * sw}`} />
                <text x={o.box[0]} y={o.box[1] - 4 * sw} fill={o.id === perception?.selected_object ? "#e9e7e1" : "#7d7f80"} fontSize={11 * sw} fontFamily="var(--mono)">
                  SAM #{o.id}
                  {o.mask_confidence !== null ? ` ${o.mask_confidence.toFixed(2)}` : ""}
                </text>
              </g>
            ))}
          {layers.path && trail.length > 1 && (
            <polyline points={trail.map((x) => `${x.x},${x.y}`).join(" ")} fill="none" stroke="var(--series-measured)" strokeWidth={1.5 * sw} opacity={0.8} />
          )}
          {layers.program && fitTrail.length > 1 && (
            <polyline points={fitTrail.map(([, x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="var(--series-fitted)" strokeWidth={1.5 * sw} strokeDasharray={`${3 * sw} ${2 * sw}`} />
          )}
          {layers.program && fit && fk >= 0 && (
            <circle cx={fit.x[fk]} cy={fit.y[fk]} r={(fit.d?.[fk] ?? medD) / 2 + 3 * sw} fill="none" stroke="var(--series-fitted)" strokeWidth={1.5 * sw} />
          )}
          {layers.outline && f.status === "measured" && f.outline && (
            <ellipse cx={f.x!} cy={f.y!} rx={f.outline.semi_major_px} ry={f.outline.semi_minor_px}
              transform={`rotate(${f.outline.angle_deg} ${f.x} ${f.y})`} fill="none" stroke="var(--series-measured)" strokeWidth={1.5 * sw} />
          )}
          {layers.outline && f.status === "measured" && (
            <g stroke="#fff" strokeWidth={sw}>
              <line x1={f.x! - 6 * sw} x2={f.x! + 6 * sw} y1={f.y!} y2={f.y!} />
              <line x1={f.x!} x2={f.x!} y1={f.y! - 6 * sw} y2={f.y! + 6 * sw} />
            </g>
          )}
        </svg>
        <span className="tag r" style={{ color: f.status === "measured" ? "#7fd17f" : f.status === "uncertain" ? "#fbc85a" : "#ea7f7f" }}>
          {f.status}
        </span>
      </div>
      {!compact && <div className="transport">
        <button className="btn sm" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "pause" : "play"} style={{ width: 64 }}>
          {playing ? "Pause" : "Play"}
        </button>
        <button className="btn sm" onClick={() => pos > 0 && setIndex(frames[pos - 1].frame_index)} aria-label="previous frame">
          ◀
        </button>
        <button className="btn sm" onClick={() => pos < frames.length - 1 && setIndex(frames[pos + 1].frame_index)} aria-label="next frame">
          ▶
        </button>
        <input type="range" min={0} max={frames.length - 1} value={pos} onChange={(e) => setIndex(frames[+e.target.value].frame_index)} aria-label="frame" />
        <select className="input" style={{ height: 28, width: 80 }} value={speed} onChange={(e) => setSpeed(+e.target.value)} aria-label="playback speed">
          {[0.1, 0.25, 0.5, 1].map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
        <span className="timecode">
          frame {index} · t = {f.t_s.toFixed(4)} s (PTS {f.t_exact.num}/{f.t_exact.den})
        </span>
      </div>}
      {compact && (
        <div className="transport">
          <button className="btn sm" style={{ width: 70 }} onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
          <input type="range" min={0} max={frames.length - 1} value={pos} onChange={(e) => { setPlaying(false); setIndex(frames[+e.target.value].frame_index); }} aria-label="frame" />
          <span className="timecode">frame {index} · {f.t_s.toFixed(3)} s</span>
        </div>
      )}
    </div>
  );
}
