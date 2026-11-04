import { useEffect, useMemo, useRef, useState } from "react";

export interface MeasuredPoint {
  t: number;
  v: number;
  s?: number | null;
  frame: number;
}
export interface Line {
  key: string;
  label: string;
  color: string;
  dash?: boolean;
  t: number[];
  v: number[];
}
export interface Mark {
  t: number;
  label: string;
  color: string;
  dash?: boolean;
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function interp(ts: number[], vs: number[], t: number): number | null {
  if (!ts.length || t < ts[0] - 1e-9 || t > ts[ts.length - 1] + 1e-9) return null;
  let lo = 0;
  let hi = ts.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (ts[m] <= t) lo = m;
    else hi = m;
  }
  const f = ts[hi] === ts[lo] ? 0 : (t - ts[lo]) / (ts[hi] - ts[lo]);
  return vs[lo] + f * (vs[hi] - vs[lo]);
}

export default function Chart({
  title, unit, invert = false, points = [], sigmaMult = 1, lines = [], gaps = [], marks = [], bands = [],
  cursorT, onPick, height = 220, measuredColor = "var(--series-measured)",
}: {
  title: string;
  unit: string;
  invert?: boolean;
  points?: MeasuredPoint[];
  sigmaMult?: number;
  lines?: Line[];
  gaps?: { t: number; status: string }[];
  marks?: Mark[];
  bands?: { t0: number; t1: number; label: string }[];
  cursorT?: number | null;
  onPick?: (frame: number, t: number) => void;
  height?: number;
  measuredColor?: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(640);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const M = { l: 58, r: 14, t: 10, b: 30 };
  const iw = w - M.l - M.r;
  const ih = height - M.t - M.b;

  const { x0, x1, y0, y1 } = useMemo(() => {
    const ts: number[] = [];
    const vs: number[] = [];
    for (const p of points) {
      ts.push(p.t);
      vs.push(p.v - (p.s ?? 0) * sigmaMult, p.v + (p.s ?? 0) * sigmaMult);
    }
    for (const l of lines) {
      ts.push(...l.t);
      vs.push(...l.v);
    }
    for (const g of gaps) ts.push(g.t);
    for (const b of bands) ts.push(b.t0, b.t1);
    let a = Math.min(...ts), b = Math.max(...ts), c = Math.min(...vs), d = Math.max(...vs);
    if (!Number.isFinite(a)) [a, b] = [0, 1];
    if (!Number.isFinite(c)) [c, d] = [0, 1];
    const pad = (d - c) * 0.08 || 1;
    return { x0: a, x1: b === a ? a + 1 : b, y0: c - pad, y1: d + pad };
  }, [points, lines, gaps, bands, sigmaMult]);

  const X = (t: number) => M.l + ((t - x0) / (x1 - x0)) * iw;
  const Y = (v: number) => (invert ? M.t + ((v - y0) / (y1 - y0)) * ih : M.t + ih - ((v - y0) / (y1 - y0)) * ih);
  const tInv = (px: number) => x0 + ((px - M.l) / iw) * (x1 - x0);
  const path = (l: Line) => l.t.map((t, i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(l.v[i]).toFixed(1)}`).join("");
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90)));
  const yt = niceTicks(y0, y1, 4);

  const near = hover === null ? null : points.reduce<MeasuredPoint | null>((best, p) => (!best || Math.abs(p.t - hover) < Math.abs(best.t - hover) ? p : best), null);
  const snapT = near && Math.abs(X(near.t) - X(hover!)) < 14 ? near.t : hover;

  return (
    <div ref={wrap} style={{ position: "relative" }}>
      <div className="row tiny" style={{ marginBottom: 4 }}>
        <span className="eyebrow">{title}</span>
        <span className="muted mono">{unit}</span>
      </div>
      <svg
        className="chart"
        width={w}
        height={height}
        role="img"
        aria-label={title}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = e.clientX - r.left;
          setHover(px >= M.l && px <= M.l + iw ? tInv(px) : null);
        }}
        onMouseLeave={() => setHover(null)}
        onClick={() => near && onPick?.(near.frame, near.t)}
        style={{ cursor: onPick ? "pointer" : "default" }}
      >
        <g className="grid">
          {yt.map((v) => (
            <line key={v} x1={M.l} x2={M.l + iw} y1={Y(v)} y2={Y(v)} />
          ))}
        </g>
        {bands.map((b, i) => (
          <g key={i}>
            <rect x={X(b.t0)} y={M.t} width={Math.max(0, X(b.t1) - X(b.t0))} height={ih} fill="url(#hatch)" />
            <text x={X(b.t0) + 6} y={M.t + 12} fill="var(--ink-3)" fontSize="10.5">
              {b.label}
            </text>
          </g>
        ))}
        <defs>
          <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(217,89,38,0.16)" strokeWidth="3" />
          </pattern>
        </defs>
        <g className="axis">
          {yt.map((v) => (
            <text key={v} x={M.l - 8} y={Y(v) + 3.5} textAnchor="end">
              {Math.abs(v) >= 1000 ? v.toFixed(0) : +v.toFixed(2)}
            </text>
          ))}
          {xt.map((t) => (
            <text key={t} x={X(t)} y={M.t + ih + 18} textAnchor="middle">
              {t.toFixed(2)}
            </text>
          ))}
          <line x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} stroke="#343b42" />
        </g>
        {/* frames without a coordinate: drawn as ticks on the time axis, never interpolated */}
        {gaps.map((g, i) => (
          <g key={i}>
            <line x1={X(g.t)} x2={X(g.t)} y1={M.t} y2={M.t + ih} stroke={g.status === "missing" ? "rgba(208,59,59,0.35)" : "rgba(250,178,25,0.28)"} strokeDasharray="2 3" />
            <path d={`M${X(g.t) - 3.5},${M.t + ih - 1}l3.5,-6l3.5,6z`} fill={g.status === "missing" ? "var(--status-critical)" : "var(--status-warn)"} />
          </g>
        ))}
        {marks.map((m, i) => (
          <g key={i}>
            <line x1={X(m.t)} x2={X(m.t)} y1={M.t} y2={M.t + ih} stroke={m.color} strokeWidth={1} strokeDasharray={m.dash ? "4 3" : undefined} opacity={0.8} />
          </g>
        ))}
        {lines.map((l) => (
          <path key={l.key} d={path(l)} fill="none" stroke={l.color} strokeWidth={2} strokeDasharray={l.dash ? "6 4" : undefined} strokeLinejoin="round" />
        ))}
        {points.map((p) => {
          const s = (p.s ?? 0) * sigmaMult;
          return (
            <g key={p.frame}>
              {s > 0 && <line x1={X(p.t)} x2={X(p.t)} y1={Y(p.v - s)} y2={Y(p.v + s)} stroke={measuredColor} strokeWidth={1.2} opacity={0.7} />}
              <circle cx={X(p.t)} cy={Y(p.v)} r={3.6} fill={measuredColor} stroke="var(--surface-1)" strokeWidth={2} />
            </g>
          );
        })}
        {cursorT !== undefined && cursorT !== null && cursorT >= x0 && cursorT <= x1 && (
          <line x1={X(cursorT)} x2={X(cursorT)} y1={M.t} y2={M.t + ih} stroke="var(--ink-1)" strokeWidth={1} opacity={0.6} />
        )}
        {snapT !== null && <line x1={X(snapT)} x2={X(snapT)} y1={M.t} y2={M.t + ih} stroke="var(--ink-2)" strokeWidth={1} strokeDasharray="2 2" />}
      </svg>
      {snapT !== null && (
        <div className="tooltip" style={{ left: Math.min(X(snapT) + 12, w - 190), top: 24 }}>
          <div className="tt-row">
            <span className="l">t</span>
            <span>{snapT.toFixed(4)} s</span>
          </div>
          {near && near.t === snapT && (
            <>
              <div className="tt-row">
                <span className="l">frame</span>
                <span>{near.frame}</span>
              </div>
              <div className="tt-row">
                <span className="l">measured</span>
                <span>
                  {near.v.toFixed(2)}
                  {near.s ? ` ± ${near.s.toFixed(2)}` : ""}
                </span>
              </div>
            </>
          )}
          {lines.map((l) => {
            const v = interp(l.t, l.v, snapT);
            return v === null ? null : (
              <div className="tt-row" key={l.key}>
                <span className="l">{l.label}</span>
                <span>{v.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
