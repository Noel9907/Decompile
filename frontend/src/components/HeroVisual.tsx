import { useEffect, useState } from "react";
import { api, frameUrl } from "../api";
import type { Analysis, Counterfactual } from "../types";
import { validPart } from "../types";

export interface Featured {
  runId: string;
  rid: string;
  analysis: Analysis;
  filename: string;
}

const path = (xs: number[], ys: number[]) => xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join("");

/** The landing visual: a real reconstruction replayed in four beats (track, measure, fit, rewrite). */
export default function HeroVisual({ featured }: { featured: Featured | null }) {
  const [cf, setCf] = useState<Counterfactual | null>(null);
  const [cycle, setCycle] = useState(0);
  useEffect(() => {
    if (!featured?.analysis.program) return;
    api.counterfactual(featured.runId, featured.rid, { restitution: null, gravity_scale: 0.165, extend_s: 0.6 }).then(setCf).catch(() => setCf(null));
  }, [featured]);
  useEffect(() => {
    const h = window.setInterval(() => setCycle((c) => c + 1), 9000);
    return () => window.clearInterval(h);
  }, []);

  if (!featured)
    return (
      <div className="window">
        <div className="bar"><i /> <i /> <i /><span className="mono" style={{ marginLeft: 8 }}>bounce.mp4 → decompiled.py</span></div>
        <div style={{ aspectRatio: "16 / 9", display: "grid", placeItems: "center" }} className="muted small">
          Decompile a video to see it here.
        </div>
      </div>
    );

  const a = featured.analysis;
  const [W, H] = a.report.video.displayed_size;
  const measured = a.frames.filter((f) => f.status === "measured");
  const fd = a.fitted_dense;
  const c = cf ? validPart(cf.dense.counterfactual) : null;
  const first = a.frames[0].frame_index;
  const pp = a.program_parameters;
  const sw = W / 500;

  return (
    <div className="window" key={cycle}>
      <div className="bar">
        <i /> <i /> <i />
        <span className="mono" style={{ marginLeft: 8, color: "var(--ink-2)" }}>{featured.filename}</span>
        <span className="spacer" />
        <span className="tiny">real footage · real SAM 3.1 output</span>
      </div>
      <div className="media" style={{ borderRadius: 0 }}>
        <img src={frameUrl(featured.runId, featured.rid, first)} alt="first frame of the reconstructed clip" style={{ filter: "saturate(0.8) brightness(0.8)" }} />
        <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {measured.map((f, i) => (
            <circle key={f.frame_index} className="pop" style={{ animationDelay: `${0.3 + i * 0.07}s`, transformBox: "fill-box", transformOrigin: "center" }}
              cx={f.x!} cy={f.y!} r={3.2 * sw} fill="var(--series-measured)" stroke="#000" strokeWidth={sw} />
          ))}
          {fd && (
            <path className="draw" pathLength={1} style={{ animationDelay: `${0.5 + measured.length * 0.07}s` }}
              d={path(fd.x, fd.y)} fill="none" stroke="var(--series-fitted)" strokeWidth={2.2 * sw} strokeLinecap="round" />
          )}
          {c && (
            <path className="draw" pathLength={1} style={{ animationDelay: `${1.6 + measured.length * 0.07}s`, animationDuration: "3s" }}
              d={path(c.x, c.y)} fill="none" stroke="var(--series-cf)" strokeWidth={2 * sw} strokeDasharray={`${6 * sw} ${4 * sw}`} />
          )}
        </svg>
        <div className="type-in" style={{ position: "absolute", left: 16, bottom: 16, animationDelay: `${1.0 + measured.length * 0.07}s` }}>
          <div className="window" style={{ borderRadius: 12, background: "var(--surface-1)", opacity: 0.95, backdropFilter: "blur(8px)" }}>
            <pre className="code" style={{ padding: "10px 0", fontSize: 11.5, lineHeight: 1.65 }}>
              <span className="ln"><span className="tok-a">restitution</span><span className="tok-p">=</span><span className="tok-n">{pp?.restitution?.toFixed(3)}</span><span className="tok-p">,</span>  <span className="tok-c"># measured at the bounce</span>{"\n"}</span>
              <span className="ln"><span className="tok-a">g</span><span className="tok-p">=</span><span className="tok-n">{(pp?.gravity_magnitude_diameters_s2 ?? pp?.gravity_magnitude_px_s2)?.toFixed(1)}</span><span className="tok-p">,</span>  <span className="tok-c"># {pp?.model === "projective" ? "ball diameters / s²" : "px / s²"}</span>{"\n"}</span>
              <span className="ln"><span className="tok-f">simulate</span><span className="tok-p">(</span><span className="tok-a">program</span><span className="tok-p">)</span>  <span className="tok-c"># fit {a.report.reconstruction_consistency?.rms_px.toFixed(2)} px rms</span>{"\n"}</span>
            </pre>
          </div>
        </div>
        <div className="legend type-in" style={{ position: "absolute", right: 16, top: 14, animationDelay: "0.2s", background: "rgba(8,9,11,0.72)", color: "#f2f1ec", padding: "6px 10px", borderRadius: 8, backdropFilter: "blur(6px)" }}>
          <span className="k"><span className="mk" style={{ background: "var(--series-measured)" }} />measured</span>
          <span className="k"><span className="sw" style={{ background: "var(--series-fitted)" }} />decompiled</span>
          {c && <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-cf)" }} />on the Moon</span>}
        </div>
      </div>
    </div>
  );
}
