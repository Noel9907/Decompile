import { useEffect, useRef, useState } from "react";
import { api, frameUrl, useAsync } from "../api";
import type { Analysis, Run } from "../types";
import Chart from "./Chart";

export interface Heldout {
  fitted_window: [number, number];
  predicted_window: [number, number];
  frames: { frame_index: number; t_s: number; status: string; measured: [number, number] | null; predicted: [number, number] | null; predicted_diameter_px: number | null; error_px?: number }[];
  dense: { t: number[]; x: number[]; y: number[] };
  summary: {
    n_compared: number; median_error_px: number | null; max_error_px: number | null; median_diameter_px: number | null;
    predicted_next_bounce_s: number | null; observed_next_bounce: { frame: number; t_s: number } | null; first_frames_error_px: number[];
  };
  label: string;
}

/** The fitted program predicts the part of the video it never saw; compared with Decompile's measurements there. */
export default function PredictionCard({ run, rid, analysis }: { run: Run; rid: string; analysis: Analysis }) {
  const h = useAsync(() => api.predict(run.id, rid) as Promise<Heldout>, [run.id, rid]);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(true);
  const kRef = useRef(0);
  kRef.current = k;
  const data = h.data;
  useEffect(() => {
    if (!data || !playing) return;
    const id = window.setInterval(() => setK((x) => (x + 1) % data.frames.length), 140);
    return () => window.clearInterval(id);
  }, [data, playing]);

  if (h.loading) return <div className="card card-pad"><div className="h3">Did it predict the future?</div><p className="small muted" style={{ marginTop: 8 }}>Measuring the frames after the fitted window and running the program forward…</p><div className="progress indeterminate" style={{ marginTop: 12 }}><div /></div></div>;
  if (h.error || !data) return <div className="card card-pad"><div className="h3">Did it predict the future?</div><p className="small muted" style={{ marginTop: 8 }}>{h.error ?? "No later frames to predict."}</p></div>;

  const [W, H] = analysis.report.video.displayed_size;
  const sw = Math.max(1, W / 560);
  const f = data.frames[k];
  const D = data.summary.median_diameter_px ?? 50;
  const pbT = data.summary.predicted_next_bounce_s ?? Infinity;
  const first = data.frames.filter((x) => x.error_px != null && x.t_s < pbT).map((x) => x.error_px!);
  const ob = data.summary.observed_next_bounce;
  const pb = data.summary.predicted_next_bounce_s;
  const fitted = analysis.frames.filter((x) => x.status === "measured");
  const held = data.frames.filter((x) => x.measured);
  const fd = analysis.fitted_dense;

  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <div className="card-head">
        <span className="t" style={{ fontSize: 17 }}>Did it predict the future?</span>
        <span className="s">The program only ever saw frames {data.fitted_window[0]}–{data.fitted_window[1]}. These are frames {data.predicted_window[0]}–{data.predicted_window[1]}.</span>
      </div>
      <div className="card-body">
        <div className="metrics well" style={{ marginBottom: 18 }}>
          <div className="metric">
            <span className="label">Frames before the next bounce</span>
            <span className="value">{first.length ? `${Math.min(...first).toFixed(0)}–${Math.max(...first).toFixed(0)}` : "—"}<span className="u">px off</span></span>
            <span className="note">≈ {first.length ? (Math.max(...first) / D).toFixed(2) : "—"} ball widths, on frames it never saw</span>
          </div>
          <div className="metric">
            <span className="label">Next bounce</span>
            <span className="value">{pb ? pb.toFixed(3) : "—"}<span className="u">s predicted</span></span>
            <span className="note">{ob && pb ? `real ball (SAM 3.1 track) lands at frame ${ob.frame}: ${(Math.abs(pb - ob.t_s) * 1000).toFixed(0)} ms apart, about ${Math.max(1, Math.round(Math.abs(pb - ob.t_s) * 30))} frame` : "not visible in the track"}</span>
          </div>
          <div className="metric">
            <span className="label">After the 2nd bounce</span>
            <span className="value">{data.summary.median_error_px?.toFixed(0) ?? "—"}<span className="u">px</span></span>
            <span className="note">median over {data.summary.n_compared} later frames: the real 2nd bounce lost more energy than the 1st</span>
          </div>
        </div>
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div>
            <div className="media">
              <img src={frameUrl(run.id, rid, f.frame_index)} alt={`later frame ${f.frame_index}`} />
              <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                <polyline points={data.frames.slice(0, k + 1).filter((x) => x.predicted).map((x) => `${x.predicted![0]},${x.predicted![1]}`).join(" ")} fill="none" stroke="var(--series-fitted)" strokeWidth={2 * sw} strokeDasharray={`${5 * sw} ${3 * sw}`} />
                {f.predicted && <circle cx={f.predicted[0]} cy={f.predicted[1]} r={(f.predicted_diameter_px ?? D) / 2 + 3 * sw} fill="none" stroke="var(--series-fitted)" strokeWidth={2 * sw} strokeDasharray={`${5 * sw} ${3 * sw}`} />}
                {f.measured && <g stroke="#fff" strokeWidth={1.5 * sw}><line x1={f.measured[0] - 7 * sw} x2={f.measured[0] + 7 * sw} y1={f.measured[1]} y2={f.measured[1]} /><line x1={f.measured[0]} x2={f.measured[0]} y1={f.measured[1] - 7 * sw} y2={f.measured[1] + 7 * sw} /></g>}
              </svg>
              <span className="tag">real · frame {f.frame_index}</span>
              <span className="tag r" style={{ color: "#8ff0c8" }}>{f.error_px != null ? `${f.error_px.toFixed(1)} px off` : f.status === "measured" ? "" : `${f.status}: no measurement`}</span>
            </div>
            <div className="transport">
              <button className="btn sm" style={{ width: 70 }} onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
              <input type="range" min={0} max={data.frames.length - 1} value={k} onChange={(e) => { setPlaying(false); setK(+e.target.value); }} aria-label="later frame" />
            </div>
            <div className="legend" style={{ marginTop: 8 }}>
              <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-fitted)" }} />predicted by the program</span>
              <span className="k">✛ measured in the real frame</span>
            </div>
          </div>
          <div>
            <Chart title="Height over the whole video" unit="px, oriented like the image" invert height={260}
              points={[...fitted.map((x) => ({ t: x.t_s, v: x.y!, s: x.sigma_y, frame: x.frame_index })),
                ...held.map((x) => ({ t: x.t_s, v: x.measured![1], s: null, frame: x.frame_index }))]}
              lines={[
                ...(fd ? [{ key: "fit", label: "fitted (frames it saw)", color: "var(--series-fitted)", t: fd.t, v: fd.y }] : []),
                { key: "pred", label: "predicted (never seen)", color: "var(--series-fitted)", dash: true, t: data.dense.t, v: data.dense.y },
              ]}
              marks={[{ t: data.frames[0].t_s, label: "fit ends", color: "var(--ink-3)", dash: true }]}
              cursorT={f.t_s} onPick={(fr) => { const i = data.frames.findIndex((x) => x.frame_index === fr); if (i >= 0) { setPlaying(false); setK(i); } }} />
            <p className="tiny muted" style={{ marginTop: 10 }}>{data.label}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
