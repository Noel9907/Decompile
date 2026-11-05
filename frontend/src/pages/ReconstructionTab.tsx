import { useCallback, useEffect, useMemo, useState } from "react";
import { api, useAsync, videoUrl } from "../api";
import Chart from "../components/Chart";
import FrameInspector, { type Layers } from "../components/FrameInspector";
import VideoPlayer from "../components/VideoPlayer";
import type { Analysis, FrameRow, Run } from "../types";
import { Badge, Panel, Readout, STATUS_COLOR, StatusIcon, fmt } from "../ui";

export function gapsOf(frames: FrameRow[]) {
  return frames.filter((f) => f.status !== "measured").map((f) => ({ t: f.t_s, status: f.status }));
}

export default function ReconstructionTab({ run, rid, analysis }: { run: Run; rid: string; analysis: Analysis }) {
  const perception = useAsync(() => api.perception(run.id, rid), [run.id, rid]);
  const ft = useAsync(() => api.frameTable(run.id), [run.id]);
  const frames = analysis.frames;
  const [index, setIndex] = useState(frames[0]?.frame_index ?? 0);
  const [layers, setLayers] = useState<Layers>({ masks: true, boxes: true, outline: true, path: true, program: true });
  const [sigmaMult, setSigmaMult] = useState(3);
  const rep = analysis.report;
  const c = rep.measurement.status_counts;
  const cur = frames.find((f) => f.frame_index === index) ?? frames[0];
  const W = rep.video.displayed_size[0];
  const H = rep.video.displayed_size[1];
  useEffect(() => setIndex(frames[0]?.frame_index ?? 0), [rid, frames]);

  const measured = useMemo(() => frames.filter((f) => f.status === "measured"), [frames]);
  const pts = (key: "x" | "y" | "diameter_px", s: "sigma_x" | "sigma_y" | "sigma_diameter_px") =>
    measured.map((f) => ({ t: f.t_s, v: f[key] as number, s: f[s], frame: f.frame_index }));
  const fd = analysis.fitted_dense;
  const fitted = (k: "x" | "y" | "d") =>
    fd && fd[k] ? [{ key: "fit", label: "scene program", color: "var(--series-fitted)", t: fd.t, v: fd[k] as number[] }] : [];
  const evs = (fd?.events ?? []).map((e) => ({ t: e.t_s, label: "simulated contact", color: "var(--series-fitted)", dash: true }));
  const pick = useCallback((fr: number) => setIndex(fr), []);
  const toggle = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }));
  const selected = rep.sam.object_id;
  const samRec = perception.data?.frames[String(index)]?.find((o) => o.id === selected);

  return (
    <div className="stack">
      <div className="readouts">
        <Readout label="SAM object" value={`#${selected}`} note={`of ${rep.sam.objects_in_result.length} in result · ${rep.sam.model}`} />
        <Readout label="Window" value={`${rep.clip.start_frame}–${rep.clip.end_frame_exclusive - 1}`} note={`${rep.clip.frames} frames`} />
        <Readout label="Measured" value={c.measured} note={<span style={{ color: "var(--good-ink)" }}>accepted with coordinate</span>} />
        <Readout label="Uncertain" value={c.uncertain} note="no coordinate, not interpolated" />
        <Readout label="Missing" value={c.missing} note="no usable mask / outline" />
        <Readout label="Median σ" value={fmt.num(rep.measurement.median_sigma_px, 2)} unit="px" note="per-frame 1σ centre" />
        <Readout label="Median diameter" value={fmt.num(rep.measurement.median_diameter_px, 1)} unit="px" note="apparent" />
      </div>

      <div className="grid-2">
        <Panel title="A · Original video" sub="as uploaded; frame number from the exact PTS table">
          <VideoPlayer src={videoUrl(run.id)} times={ft.data?.t_s} tag="observed" />
        </Panel>
        <Panel
          title="B · Perception · C · Measurement"
          sub="exact decoded frames"
          right={
            <div className="row tiny">
              {(["masks", "boxes", "outline", "path", "program"] as (keyof Layers)[]).map((k) => (
                <label key={k} className="row" style={{ gap: 4 }}>
                  <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
                  {k === "program" ? "scene program" : k}
                </label>
              ))}
            </div>
          }
        >
          <FrameInspector runId={run.id} rid={rid} analysis={analysis} perception={perception.data} index={index} setIndex={setIndex} layers={layers} width={W} height={H} />
          <div className="legend" style={{ marginTop: 10 }}>
            <span className="k"><span className="sw" style={{ background: "var(--ink-4)" }} />SAM mask &amp; box (selected)</span>
            <span className="k"><span className="sw dash" style={{ borderColor: "#7d7f80" }} />other SAM objects</span>
            <span className="k"><span className="sw" style={{ background: "var(--series-measured)" }} />measured outline &amp; path</span>
            {analysis.fitted && <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-fitted)" }} />scene program</span>}
          </div>
        </Panel>
      </div>

      <Panel title="Frame status" sub="click a frame · every frame in the window is shown, including rejected ones">
        <div className="barcode" role="list">
          {frames.map((f) => (
            <button key={f.frame_index} role="listitem" className={f.frame_index === index ? "cur" : ""} onClick={() => setIndex(f.frame_index)}
              title={`frame ${f.frame_index}: ${f.status}${f.reason ? ` — ${f.reason}` : ""}`} style={{ background: STATUS_COLOR[f.status], opacity: f.status === "measured" ? 0.75 : 1 }}
              aria-label={`frame ${f.frame_index} ${f.status}`} />
          ))}
        </div>
        <div className="row" style={{ marginTop: 14, alignItems: "flex-start", gap: 24 }}>
          <div className="row" style={{ gap: 8 }}>
            <StatusIcon status={cur.status} />
            <div>
              <div style={{ fontWeight: 600 }}>
                Frame {cur.frame_index} — {cur.status}
              </div>
              <div className="tiny muted">{cur.reason ?? (cur.status === "measured" ? "outline measured in the original pixels" : "")}</div>
            </div>
          </div>
          <dl className="kv grow" style={{ gridTemplateColumns: "auto 1fr auto 1fr auto 1fr" }}>
            <dt>t (PTS)</dt>
            <dd>{cur.t_s.toFixed(5)} s</dd>
            <dt>centre</dt>
            <dd>{cur.x !== null ? `${cur.x.toFixed(2)}, ${cur.y!.toFixed(2)}` : "— (no coordinate)"}</dd>
            <dt>σ (x, y)</dt>
            <dd>{cur.sigma_x !== null ? `${cur.sigma_x.toFixed(2)}, ${cur.sigma_y!.toFixed(2)} px` : "—"}</dd>
            <dt>diameter</dt>
            <dd>{cur.diameter_px !== null ? `${cur.diameter_px.toFixed(2)} ± ${fmt.num(cur.sigma_diameter_px, 2)} px` : "—"}</dd>
            <dt>SAM confidence</dt>
            <dd>{fmt.num(cur.confidence, 3)}</dd>
            <dt>SAM box</dt>
            <dd>{samRec ? `[${samRec.box.join(", ")}]` : "—"}</dd>
            <dt>mask area</dt>
            <dd>{cur.mask_area_px ?? "—"} px</dd>
            <dt>IoU outline/mask</dt>
            <dd>{fmt.num(cur.iou_outline_mask, 3)}</dd>
            <dt>centre − mask centroid</dt>
            <dd>{fmt.num(cur.centre_offset_from_mask_px, 2)} px</dd>
          </dl>
        </div>
      </Panel>

      <Panel
        title="D · Trajectory"
        sub="measured centres with error bars · exact source timestamps"
        right={
          <label className="row tiny">
            error bars
            <select className="input" style={{ height: 26 }} value={sigmaMult} onChange={(e) => setSigmaMult(+e.target.value)}>
              <option value={1}>±1σ</option>
              <option value={3}>±3σ</option>
            </select>
          </label>
        }
      >
        <div className="legend" style={{ marginBottom: 10 }}>
          <span className="k"><span className="mk" style={{ background: "var(--series-measured)" }} />measured (SAM-guided outline)</span>
          {fd && <span className="k"><span className="sw" style={{ background: "var(--series-fitted)" }} />scene program (fitted)</span>}
          <span className="k"><StatusIcon status="uncertain" size={12} />uncertain frame</span>
          <span className="k"><StatusIcon status="missing" size={12} />missing frame</span>
          {fd && <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-fitted)" }} />simulated contact</span>}
        </div>
        <div className="grid-2">
          <Chart title="x(t)" unit="px →" points={pts("x", "sigma_x")} sigmaMult={sigmaMult} lines={fitted("x")} gaps={gapsOf(frames)} marks={evs} cursorT={cur.t_s} onPick={pick} />
          <Chart title="y(t)" unit="px, image +y down (chart oriented like the image)" invert points={pts("y", "sigma_y")} sigmaMult={sigmaMult} lines={fitted("y")} gaps={gapsOf(frames)} marks={evs} cursorT={cur.t_s} onPick={pick} />
        </div>
        <div style={{ marginTop: 12 }}>
          <Chart title="apparent diameter D(t)" unit="px (depth cue for the projective model)" height={160} points={pts("diameter_px", "sigma_diameter_px")} sigmaMult={sigmaMult}
            lines={fitted("d")} gaps={gapsOf(frames)} marks={evs} cursorT={cur.t_s} onPick={pick} />
        </div>
      </Panel>

      <div className="grid-2">
        <EventsPanel analysis={analysis} />
        <Panel title="Measurement table" sub={`${frames.length} frames`} flush>
          <div className="scroll" style={{ maxHeight: 360 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Frame</th>
                  <th>t (s)</th>
                  <th>Status</th>
                  <th>x</th>
                  <th>y</th>
                  <th>σx</th>
                  <th>σy</th>
                  <th>D</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {frames.map((f) => (
                  <tr key={f.frame_index} className={`click ${f.frame_index === index ? "sel" : ""}`} onClick={() => setIndex(f.frame_index)}>
                    <td>{f.frame_index}</td>
                    <td>{f.t_s.toFixed(4)}</td>
                    <td className="text">
                      <span className="row" style={{ gap: 4 }}>
                        <StatusIcon status={f.status} size={12} />
                        {f.status}
                      </span>
                    </td>
                    <td>{fmt.num(f.x, 2)}</td>
                    <td>{fmt.num(f.y, 2)}</td>
                    <td>{fmt.num(f.sigma_x, 2)}</td>
                    <td>{fmt.num(f.sigma_y, 2)}</td>
                    <td>{fmt.num(f.diameter_px, 1)}</td>
                    <td className="text tiny muted" style={{ maxWidth: 240 }}>{f.reason ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function EventsPanel({ analysis }: { analysis: Analysis }) {
  const p = analysis.report.physics;
  const projective = p.model === "projective";
  const k = p.kinematics;
  const contact = projective
    ? p.contact_t_s !== undefined && p.contact_t_s !== null
      ? { t: p.contact_t_s as number, between: p.contact_between_measured_frames as number[] }
      : null
    : k?.contact
      ? { t: k.contact.t_s as number, between: k.contact.between_frames as number[] }
      : null;
  const byIdx = new Map(analysis.frames.map((f) => [f.frame_index, f]));
  const lo = contact ? byIdx.get(contact.between[0])?.t_s : undefined;
  const hi = contact ? byIdx.get(contact.between[1])?.t_s : undefined;
  return (
    <Panel title="E · Physical events" sub="detected in the measured trajectory">
      {!contact ? (
        <div className="callout warn">
          <strong>No contact was established.</strong>{" "}
          {(analysis.physics_status?.reasons ?? p.reasons ?? []).join("; ") || "The kinematic analysis did not find a supported contact."}
        </div>
      ) : (
        <div className="col">
          <div className="row">
            <Badge tone="info">contact (bounce)</Badge>
            <span className="mono">t = {contact.t.toFixed(4)} s</span>
          </div>
          <dl className="kv">
            <dt>Evidence</dt>
            <dd>velocity discontinuity fitted inside the measured gap</dd>
            <dt>Bracketing measured frames</dt>
            <dd>
              {contact.between[0]} → {contact.between[1]}
            </dd>
            <dt>Interval (uncertainty bound)</dt>
            <dd>{lo !== undefined && hi !== undefined ? `${lo.toFixed(4)} – ${hi.toFixed(4)} s (${((hi - lo) * 1000).toFixed(1)} ms)` : "—"}</dd>
            {projective && (
              <>
                <dt>Model selected</dt>
                <dd>{p.selected}</dd>
                {Object.entries((p.verdicts ?? {}) as Record<string, string>).map(([kk, v]) => (
                  <FragmentKV key={kk} k={kk.replace(/_/g, " ")} v={v} />
                ))}
              </>
            )}
            {!projective &&
              k?.contact?.axes?.map((a: any) => (
                <FragmentKV key={a.axis} k={`v_${a.axis} before → after`} v={`${a.v_before_px_s.toFixed(1)} → ${a.v_after_px_s.toFixed(1)} px/s ${a.reversal ? "(reversal)" : ""}`} />
              ))}
            {!projective && k?.additional_discontinuity && <FragmentKV k="second discontinuity" v={`ΔBIC ${k.additional_discontinuity.delta_bic.toFixed(1)}`} />}
          </dl>
          <p className="tiny muted">
            The contact time is an estimate between two measured frames; frames inside the gap have no coordinate and are never interpolated.
            Contact timing is not independently validated.
          </p>
        </div>
      )}
    </Panel>
  );
}

function FragmentKV({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}
