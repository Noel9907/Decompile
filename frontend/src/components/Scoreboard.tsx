import { useEffect, useState } from "react";
import { previewUrl } from "../api";
import { Badge } from "../ui";

interface Row {
  run_id: string;
  filename: string;
  status: "decompiled" | "refused" | "pending";
  reason: string | null;
  recon_id?: string;
  model?: string;
  restitution?: number;
  fit_rms_px?: number;
  ball_diameter_cm?: number | null;
  median_diameter_px?: number | null;
  prediction?: { pre_bounce_error_px: [number, number] | null; median_error_px: number | null; next_bounce_error_ms: number | null; n_compared: number };
}

/** Every clip on this server, with its held-out prediction, or the reason it was refused. */
export default function Scoreboard() {
  const [rows, setRows] = useState<Row[] | null>(null);
  useEffect(() => {
    fetch("/api/scoreboard").then((r) => r.json()).then(setRows).catch(() => setRows([]));
  }, []);
  if (!rows) return null;
  const shown = rows.filter((r) => r.status !== "pending");
  if (!shown.length) return null;
  const done = shown.filter((r) => r.status === "decompiled" && r.prediction);
  const ms = done.map((r) => r.prediction!.next_bounce_error_ms).filter((v): v is number => v != null);
  return (
    <section className="block" id="scoreboard">
      <div className="row wrap" style={{ alignItems: "flex-end", marginBottom: 18 }}>
        <div className="grow">
          <div className="eyebrow">Tested on real videos</div>
          <h2 className="h1" style={{ marginTop: 10 }}>It predicts frames it never saw.</h2>
          <p className="lede" style={{ marginTop: 10 }}>
            Each clip is fitted on one bounce only. The program then predicts the rest of the video, and is checked against what the camera
            actually recorded. Clips it cannot explain are refused, not faked.
          </p>
        </div>
        {ms.length > 0 && (
          <div className="metric" style={{ textAlign: "right" }}>
            <span className="label">Next-bounce timing error</span>
            <span className="value">{Math.max(...ms).toFixed(0)}<span className="u">ms worst, {done.length} clip{done.length > 1 ? "s" : ""}</span></span>
          </div>
        )}
      </div>
      <div className="card" style={{ overflow: "hidden" }}>
        <table className="data">
          <thead>
            <tr>
              <th style={{ width: 92 }}></th>
              <th>Clip</th>
              <th>Result</th>
              <th>Bounciness</th>
              <th>Fit</th>
              <th>Unseen frames, before next bounce</th>
              <th>Next bounce</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const p = r.prediction;
              const D = r.median_diameter_px ?? 50;
              return (
                <tr key={r.run_id} className="click" onClick={() => (window.location.hash = `/runs/${r.run_id}/${r.status === "decompiled" ? "result" : "process"}`)}>
                  <td><div style={{ width: 76, aspectRatio: "16/9", borderRadius: 6, background: `#000 url(${previewUrl(r.run_id, 0)}) center/cover` }} /></td>
                  <td className="text" style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.filename}</td>
                  <td className="text">
                    {r.status === "decompiled" ? <Badge tone="good">decompiled</Badge> : <span title={r.reason ?? ""}><Badge tone="warn">refused</Badge> <span className="tiny muted">{(r.reason ?? "").slice(0, 70)}</span></span>}
                  </td>
                  <td>{r.restitution != null ? `${Math.round(r.restitution * 100)}%` : "—"}</td>
                  <td>{r.fit_rms_px != null ? `${r.fit_rms_px.toFixed(2)} px` : "—"}</td>
                  <td>{p?.pre_bounce_error_px ? `${p.pre_bounce_error_px[0].toFixed(0)}–${p.pre_bounce_error_px[1].toFixed(0)} px (${(p.pre_bounce_error_px[1] / D).toFixed(2)} ball widths)` : r.status === "decompiled" ? "—" : ""}</td>
                  <td>{p?.next_bounce_error_ms != null ? `${p.next_bounce_error_ms.toFixed(0)} ms off` : r.status === "decompiled" ? "not in view" : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="tiny muted" style={{ marginTop: 10 }}>
        "Unseen" frames come after the fitted window and were never used by the fit. Their positions are Decompile's own measurements of
        those frames, not independent ground truth.
      </p>
    </section>
  );
}
