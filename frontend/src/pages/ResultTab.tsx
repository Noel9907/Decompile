import { useState } from "react";
import { api, useAsync } from "../api";
import { navigate } from "../App";
import CodeView from "../components/CodeView";
import FrameInspector from "../components/FrameInspector";
import PredictionCard from "../components/PredictionCard";
import type { Analysis, Run } from "../types";
import { Badge, fmt } from "../ui";

export default function ResultTab({ run, rid, analysis }: { run: Run; rid: string; analysis: Analysis }) {
  const rep = analysis.report;
  const has = !!analysis.program;
  const src = useAsync(() => (has ? api.source(run.id, rid) : Promise.resolve(null)), [run.id, rid, has]);
  const per = useAsync(() => api.perception(run.id, rid), [run.id, rid]);
  const review = useAsync(() => (has ? api.explain(run.id, rid) : Promise.resolve(null)), [run.id, rid, has]);
  const [index, setIndex] = useState(analysis.frames[0]?.frame_index ?? 0);
  const [W, H] = rep.video.displayed_size;
  const p = rep.physics;
  const pp = analysis.program_parameters;
  const c = rep.measurement.status_counts;
  const rc = rep.reconstruction_consistency;
  const projective = pp?.model === "projective";
  const go = (t: string) => navigate(`/runs/${run.id}/${t}`);

  if (!has)
    return (
      <div className="card card-pad fade-in" style={{ maxWidth: 860 }}>
        <div className="eyebrow">Result</div>
        <h2 className="h1" style={{ marginTop: 12 }}>Tracked and measured, but not decompiled.</h2>
        <p className="lede" style={{ marginTop: 12 }}>
          SAM 3.1 tracked the object and {c.measured} frames were measured, but the physics checks did not accept this stretch of video, so
          Decompile did not invent a program.
        </p>
        <div className="callout warn" style={{ marginTop: 20 }}>
          <strong>Why:</strong> {(analysis.physics_status?.reasons ?? []).join("; ")}
        </div>
        <p className="small muted" style={{ marginTop: 16 }}>Most often the window has no bounce, two bounces, or too few clean frames. Pick a window around exactly one bounce.</p>
        <div className="row" style={{ marginTop: 22 }}>
          <button className="btn primary" onClick={() => go("process")}>Try another window</button>
          <button className="btn" onClick={() => go("reconstruction")}>Inspect the measurements</button>
        </div>
      </div>
    );

  return (
    <div className="stack fade-in">
      <div className="row wrap" style={{ alignItems: "flex-end", gap: 24 }}>
        <div className="grow" style={{ minWidth: 300 }}>
          <h2 className="h1">This video, decompiled.</h2>
          <p className="lede" style={{ marginTop: 10, maxWidth: 760 }}>
            Left: the real footage with what Decompile measured. Right: the physics it recovered, as a program that runs on the Decompile engine and
            reproduces the motion.
          </p>
        </div>
        <button className="btn primary lg" onClick={() => go("simulation")}>Ask “what if…?” →</button>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        <div className="metrics">
          <div className="metric">
            <span className="label">Bounciness</span>
            <span className="value">{pp?.restitution != null ? Math.round(pp.restitution * 100) : "—"}<span className="u">%</span></span>
            <span className="note">of its speed kept at the bounce (e = {fmt.num(pp?.restitution, 3)})</span>
          </div>
          <div className="metric">
            <span className="label">Gravity</span>
            <span className="value">{projective ? fmt.num(pp?.gravity_magnitude_diameters_s2, 0) : fmt.num(pp?.gravity_magnitude_px_s2, 0)}<span className="u">{projective ? "D/s²" : "px/s²"}</span></span>
            <span className="note">{projective && p.implied_ball_diameter_m_if_g_9_81 ? `⇒ a ${(p.implied_ball_diameter_m_if_g_9_81 * 100).toFixed(1)} cm ball on Earth` : "in image units"}</span>
          </div>
          <div className="metric">
            <span className="label">Bounce at</span>
            <span className="value">{fmt.num(projective ? p.contact_t_s : p.kinematics?.contact?.t_s, 3)}<span className="u">s</span></span>
            <span className="note">between measured frames {(projective ? p.contact_between_measured_frames : p.kinematics?.contact?.between_frames)?.join("–")}</span>
          </div>
          <div className="metric">
            <span className="label">Fit error</span>
            <span className="value">{fmt.num(rc?.rms_px, 2)}<span className="u">px</span></span>
            <span className="note">{fmt.num(rc?.rms_normalised, 2)}σ over {rc?.n} frames · gate: {p.adequacy ?? p.kinematics?.adequacy}</span>
          </div>
        </div>
      </div>

      <div className="card card-pad review">
        <div className="row wrap" style={{ gap: 10 }}>
          <span className="spark-mark" aria-hidden>✦</span>
          <span className="h3">Muse Spark's review of this fit</span>
          <span className="grow" />
          {review.data && <Badge tone={review.data.confidence === "high" ? "good" : review.data.confidence === "medium" ? "info" : "warn"}>{review.data.confidence} confidence</Badge>}
          {review.data && <span className="tiny muted mono">{review.data.model} · {review.data.latency_s.toFixed(1)} s</span>}
        </div>
        {review.loading && (
          <div style={{ marginTop: 14 }}>
            <p className="small muted">Muse Spark is reading the measurements and checking whether the physics makes sense…</p>
            <div className="progress indeterminate" style={{ marginTop: 10 }}><div /></div>
          </div>
        )}
        {review.error && <p className="small muted" style={{ marginTop: 10 }}>Review unavailable: {review.error}</p>}
        {review.data && (
          <div className="grid-2" style={{ marginTop: 14, gap: 28 }}>
            <div>
              <p style={{ fontSize: 19, fontWeight: 550, letterSpacing: "-0.015em", lineHeight: 1.35 }}>{review.data.headline}</p>
              <p className="ink2" style={{ marginTop: 10 }}>{review.data.explanation}</p>
              <p className="small" style={{ marginTop: 12 }}><span className="muted">Material: </span>{review.data.material_guess}</p>
              <p className="small" style={{ marginTop: 6 }}><span className="muted">Sanity check: </span>{review.data.plausibility}</p>
            </div>
            <div>
              <div className="tiny muted" style={{ fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>Critique</div>
              <ul className="small ink2" style={{ margin: "8px 0 0", paddingLeft: 18, display: "grid", gap: 6 }}>
                {review.data.critique.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
              <p className="tiny muted" style={{ marginTop: 10 }}>Commentary only. Muse Spark did not change any measured number.</p>
            </div>
          </div>
        )}
      </div>

      <div className="grid-2" style={{ alignItems: "start" }}>
        <div className="card" style={{ padding: 14 }}>
          <FrameInspector runId={run.id} rid={rid} analysis={analysis} perception={per.data} index={index} setIndex={setIndex}
            layers={{ masks: true, boxes: false, outline: true, path: true, program: true }} width={W} height={H} autoPlay compact />
          <div className="legend" style={{ marginTop: 12 }}>
            <span className="k"><span className="sw" style={{ background: "var(--ink-4)" }} />SAM 3.1 mask</span>
            <span className="k"><span className="sw" style={{ background: "var(--series-measured)" }} />measured outline &amp; path</span>
            <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-fitted)" }} />decompiled program</span>
          </div>
        </div>
        {src.data ? (
          <CodeView code={src.data.display} exact={src.data.exact} filename={src.data.filename} maxHeight={560} />
        ) : (
          <div className="skeleton" style={{ height: 480 }} />
        )}
      </div>

      <PredictionCard run={run} rid={rid} analysis={analysis} />

    </div>
  );
}
