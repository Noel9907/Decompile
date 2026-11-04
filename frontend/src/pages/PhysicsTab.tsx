import { useMemo } from "react";
import Chart from "../components/Chart";
import type { Analysis, Run } from "../types";
import { Badge, Panel, Readout, StatusIcon, fmt } from "../ui";

export const CATEGORIES = [
  { key: "fitted_reconstruction", label: "Fitted reconstruction", desc: "Program fitted to the observations it is compared with — a consistency check." },
  { key: "held_out_prediction", label: "Held-out prediction", desc: "Not produced by the web app (CLI: tools/real_video_demo.py prereg-sam / run / reveal)." },
  { key: "counterfactual", label: "Hypothetical counterfactual", desc: "What-if runs of the fitted program. Never observed evidence." },
];

export default function PhysicsTab({ run, analysis }: { run: Run; analysis: Analysis }) {
  const p = analysis.report.physics;
  const pp = analysis.program_parameters;
  const rc = analysis.report.reconstruction_consistency;
  const has = !!analysis.program;
  const projective = pp?.model === "projective";
  const adequacy: string | undefined = p.adequacy ?? p.kinematics?.adequacy;
  const adequacyReasons: string[] = p.adequacy_reasons ?? p.kinematics?.adequacy_reasons ?? [];
  const assumptions: string[] = (analysis.program?.provenance?.assumptions as string[]) ?? [];

  const residuals = useMemo(() => {
    if (!analysis.fitted) return null;
    const fit = analysis.fitted;
    const at = new Map(fit.frame_index.map((fi, k) => [fi, k]));
    const rows = analysis.frames.filter((f) => f.status === "measured" && at.has(f.frame_index));
    return {
      x: rows.map((f) => ({ t: f.t_s, v: f.x! - fit.x[at.get(f.frame_index)!], s: f.sigma_x, frame: f.frame_index })),
      y: rows.map((f) => ({ t: f.t_s, v: f.y! - fit.y[at.get(f.frame_index)!], s: f.sigma_y, frame: f.frame_index })),
    };
  }, [analysis]);

  return (
    <div className="stack">
      <div className="row wrap">
        {CATEGORIES.map((c) => {
          const active = c.key === "fitted_reconstruction" && has;
          return (
            <div key={c.key} className={`callout ${active ? "good" : ""}`} style={{ flex: 1, minWidth: 260, opacity: active ? 1 : 0.55 }}>
              <div className="row" style={{ gap: 8 }}>
                <StatusIcon status={active ? "done" : "skipped"} size={14} />
                <strong>{c.label}</strong>
                {active && <Badge tone="good">this result</Badge>}
              </div>
              <div className="tiny" style={{ marginTop: 4 }}>
                {c.desc}
              </div>
            </div>
          );
        })}
      </div>

      {!has ? (
        <Panel title="No executable scene program">
          <div className="callout bad" role="alert">
            <strong>Reconstruction stopped before a scene program.</strong> The physics gates did not accept this trajectory, so no program was
            produced and nothing was fabricated.
            <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {(analysis.physics_status?.reasons ?? []).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
          <p className="small muted" style={{ marginTop: 12 }}>
            The measurements remain available in the Reconstruction tab. Typical causes: the window holds no contact or more than one, too few
            measured frames, or an inadequate fit. Try a window around a single bounce.
          </p>
          {p.kinematics && (
            <dl className="kv" style={{ marginTop: 12 }}>
              <dt>Kinematic model</dt>
              <dd>{p.kinematics.selected_model}</dd>
              <dt>Adequacy</dt>
              <dd>{p.kinematics.adequacy}</dd>
              {(p.kinematics.adequacy_reasons ?? []).map((r: string, i: number) => (
                <Pair key={i} k="reason" v={r} />
              ))}
            </dl>
          )}
        </Panel>
      ) : (
        <>
          <div className="readouts">
            <Readout label="Restitution e" value={fmt.num(pp!.restitution, 3)} note="−v_n⁺ / v_n⁻ at the contact" />
            {projective ? (
              <Readout label="|g|" value={fmt.num(pp!.gravity_magnitude_diameters_s2, 1)} unit="D/s²" note="ball diameters per s²" />
            ) : (
              <Readout label="|a|" value={fmt.num(pp!.gravity_magnitude_px_s2, 1)} unit="px/s²" note="uniform acceleration" />
            )}
            <Readout label="Contact time" value={fmt.num(projective ? p.contact_t_s : p.kinematics?.contact?.t_s, 4)} unit="s" note="source PTS clock" />
            <Readout label="Fit RMS" value={fmt.num(rc?.rms_px, 2)} unit="px" note={`${fmt.num(rc?.rms_normalised, 2)} σ over ${rc?.n ?? "—"} frames`} />
            <Readout
              label="Adequacy gate"
              value={<span style={{ color: adequacy === "adequate" ? "#7fd17f" : "#fbc85a", fontSize: 16 }}>{adequacy ?? "—"}</span>}
              note={adequacyReasons.join("; ") || "fit residuals consistent with σ"}
            />
            {projective && <Readout label="Implied ball Ø" value={fmt.num((p.implied_ball_diameter_m_if_g_9_81 ?? 0) * 100, 1)} unit="cm" note="if g = 9.81 m/s²" />}
          </div>

          <div className="grid-2">
            <Panel title="Scene program parameters" sub={`${pp!.model} · ${projective ? "units: ball diameters (D), seconds" : "units: px, s"}`}>
              <dl className="kv">
                <dt>Model</dt>
                <dd>{projective ? p.selected : p.kinematics?.selected_model}</dd>
                <dt>Program start t₀</dt>
                <dd>{fmt.num(pp!.t0_s, 4)} s</dd>
                {projective ? (
                  <>
                    <dt>Initial position p₀</dt>
                    <dd>{fmt.vec(pp!.initial_position_diameters, 3)} D</dd>
                    <dt>Initial velocity v₀</dt>
                    <dd>{fmt.vec(pp!.initial_velocity_diameters_s, 3)} D/s</dd>
                    <dt>Gravity vector g</dt>
                    <dd>{fmt.vec(pp!.gravity_vector_diameters_s2, 2)} D/s²</dd>
                    <dt>Floor normal</dt>
                    <dd>{fmt.vec(pp!.floor_normal, 3)}</dd>
                    <dt>Tangential impulse ratio</dt>
                    <dd>{fmt.vec(pp!.tangential_impulse_ratio, 4)}</dd>
                    <dt>Focal length (from prior)</dt>
                    <dd>{fmt.num(pp!.focal_length_px, 1)} px</dd>
                    <dt>Camera HFOV prior</dt>
                    <dd>{p.camera_hfov_prior_deg}° (not a calibration)</dd>
                  </>
                ) : (
                  <>
                    <dt>Initial position</dt>
                    <dd>{fmt.vec(pp!.initial_position_px, 2)} px</dd>
                    <dt>Initial velocity</dt>
                    <dd>{fmt.vec(pp!.initial_velocity_px_s, 2)} px/s</dd>
                    <dt>Acceleration</dt>
                    <dd>{fmt.vec(pp!.acceleration_px_s2, 2)} px/s²</dd>
                    <dt>Contact boundary</dt>
                    <dd>
                      {pp!.contact_boundary ? `${pp!.contact_boundary.axis} = ${pp!.contact_boundary.position_px.toFixed(2)} px (inferred)` : "—"}
                    </dd>
                  </>
                )}
                <dt>Program SHA-256</dt>
                <dd>{String(p.program_sha256).slice(0, 16)}…</dd>
              </dl>
              {projective && p.verdicts && (
                <>
                  <div className="divider" />
                  <div className="eyebrow" style={{ marginBottom: 6 }}>
                    Model-comparison verdicts
                  </div>
                  <dl className="kv">
                    {Object.entries(p.verdicts as Record<string, string>).map(([k, v]) => (
                      <Pair key={k} k={k.replace(/_/g, " ")} v={v} />
                    ))}
                  </dl>
                </>
              )}
            </Panel>
            <Panel title="Fit residuals" sub="measured − scene program, with ±1σ measurement bars">
              {residuals && (
                <div className="col">
                  <Chart title="x residual" unit="px" height={150} points={residuals.x} lines={[{ key: "z", label: "zero", color: "var(--ink-4)", t: [residuals.x[0]?.t ?? 0, residuals.x[residuals.x.length - 1]?.t ?? 1], v: [0, 0] }]} />
                  <Chart title="y residual" unit="px" height={150} points={residuals.y} lines={[{ key: "z", label: "zero", color: "var(--ink-4)", t: [residuals.y[0]?.t ?? 0, residuals.y[residuals.y.length - 1]?.t ?? 1], v: [0, 0] }]} />
                </div>
              )}
              <p className="tiny muted" style={{ marginTop: 8 }}>
                {rc?.note}
              </p>
            </Panel>
          </div>
        </>
      )}

      <div className="grid-2">
        <Panel title="Assumptions" sub="encoded in the scene program">
          {assumptions.length ? (
            <ul className="small ink2" style={{ margin: 0, paddingLeft: 18 }}>
              {assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          ) : (
            <p className="small muted">{has ? "Image-space program: uniform acceleration, one inferred axis-aligned contact boundary, restitution in [0, 1], no drag." : "—"}</p>
          )}
          <div className="divider" />
          <div className="eyebrow" style={{ marginBottom: 6 }}>
            Claims recorded by the engine
          </div>
          <ul className="small ink2" style={{ margin: 0, paddingLeft: 18 }}>
            {analysis.report.claims.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </Panel>
        <Panel title="Limitations" sub="what this result does not show">
          <ul className="small ink2" style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 6 }}>
            {LIMITS.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
          <p className="tiny muted" style={{ marginTop: 10 }}>
            Run {run.id} · {analysis.report.label}
          </p>
        </Panel>
      </div>
    </div>
  );
}

const LIMITS = [
  "One body, one contact in the selected window. Multi-object and multi-contact physics are not supported.",
  "Static camera assumed; lens distortion and rolling shutter not modelled.",
  "Projective model: focal length from a field-of-view prior, not a calibration; metric scale is unknown (units are ball diameters).",
  "Fitted, not predicted: agreement with the measurements is a consistency check.",
  "Measurement σ is per frame; parameter uncertainty is not propagated into the simulated paths.",
];

function Pair({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}
