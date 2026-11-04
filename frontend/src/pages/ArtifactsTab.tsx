import { useEffect, useMemo, useState } from "react";
import { api, fileUrl, useAsync } from "../api";
import type { Artifact, Run } from "../types";
import { Badge, Empty, Panel, fmt } from "../ui";

const CATS: [string, string][] = [
  ["input", "Input"],
  ["perception", "Perception (SAM 3.1)"],
  ["measurement", "Measurement and trajectory"],
  ["physics", "Physical scene program"],
  ["simulation", "Simulation"],
  ["counterfactual", "Counterfactuals (hypothetical)"],
  ["report", "Reports and logs"],
];

export default function ArtifactsTab({ run, rid }: { run: Run; rid: string | null }) {
  const arts = useAsync(() => api.artifacts(run.id), [run.id, run.updated_utc]);
  const report = useAsync(() => api.report(run.id, rid ?? undefined), [run.id, rid, run.updated_utc]);
  const [md, setMd] = useState<string | null>(null);
  const [scope, setScope] = useState<"active" | "all">("active");

  const list = useMemo(() => {
    const a = arts.data ?? [];
    if (scope === "all" || !rid) return a;
    return a.filter((x) => !x.path.startsWith("recon/") || x.scope === rid);
  }, [arts.data, scope, rid]);

  useEffect(() => {
    setMd(null);
    const p = list.find((a) => a.path === `recon/${rid}/report.md`);
    if (p) api.text(fileUrl(run.id, p.path)).then(setMd).catch(() => setMd(null));
  }, [list, rid, run.id]);

  const r = report.data;
  return (
    <div className="stack">
      <div className="split">
        <Panel title="Run summary" sub={rid ? `reconstruction ${rid}` : "no reconstruction yet"}
          right={r && <a className="btn sm" href={`/api/runs/${run.id}/report${rid ? `?recon_id=${rid}` : ""}`} download={`${run.id}_run_summary.json`}>Summary JSON</a>}>
          {!r ? (
            <div className="skeleton" style={{ height: 200 }} />
          ) : (
            <div className="col">
              <div className="row wrap">
                <Badge tone={r.status === "scene_program_fitted" ? "good" : r.status === "not_reconstructed" ? "" : "warn"}>{String(r.status).replace(/_/g, " ")}</Badge>
                {r.result_category && <Badge tone="info">{String(r.result_category).replace(/_/g, " ")}</Badge>}
                {r.sam?.source && <Badge>{r.sam.source === "imported" ? "SAM result imported" : "SAM live request"}</Badge>}
              </div>
              <dl className="kv">
                <dt>Input video</dt>
                <dd>
                  {r.input_video.filename} · {r.input_video.frame_count} frames · {r.input_video.display_width}×{r.input_video.display_height}
                </dd>
                <dt>Selected object</dt>
                <dd>{r.selected_object ? `SAM #${r.selected_object}` : "—"}</dd>
                <dt>Tracking</dt>
                <dd>{r.tracking ? `${r.tracking.n_frames} frames (${r.tracking.first_frame}–${r.tracking.last_frame}), median conf ${fmt.num(r.tracking.median_confidence, 2)}` : "—"}</dd>
                <dt>Measurement</dt>
                <dd>
                  {r.measurement
                    ? `measured ${r.measurement.status_counts.measured} · uncertain ${r.measurement.status_counts.uncertain} · missing ${r.measurement.status_counts.missing} · median σ ${fmt.num(r.measurement.median_sigma_px, 2)} px`
                    : "—"}
                </dd>
                <dt>Physics</dt>
                <dd>
                  {r.physics
                    ? r.physics.program_sha256
                      ? `${r.physics.model}, e = ${fmt.num(r.physics.restitution, 3)}, adequacy ${r.physics.adequacy ?? r.physics.kinematics?.adequacy}`
                      : `no program: ${(r.physics.reasons ?? []).join("; ")}`
                    : "—"}
                </dd>
                <dt>Consistency</dt>
                <dd>{r.reconstruction_consistency ? `RMS ${fmt.num(r.reconstruction_consistency.rms_px, 2)} px (${fmt.num(r.reconstruction_consistency.rms_normalised, 2)} σ)` : "—"}</dd>
                <dt>{r.sam?.source === "imported" ? "SAM usage (recorded in the imported result)" : "SAM usage (reported by the API)"}</dt>
                <dd>
                  {r.sam?.usage?.video_frames_processed !== undefined ? `${r.sam.usage.video_frames_processed} video frames processed` : "—"}
                  {r.sam?.source === "imported" ? " · not charged to this run" : r.sam?.attempts?.length ? ` · actual ${fmt.usd(r.sam.attempts.at(-1)?.actual_cost_usd)}` : ""}
                </dd>
              </dl>
              {r.result_category_note && <div className="callout info small">{r.result_category_note}</div>}
              {(r.warnings as string[]).map((w, i) => (
                <div className="callout warn small" key={i}>
                  {w}
                </div>
              ))}
              <details>
                <summary className="small ink2" style={{ cursor: "pointer" }}>
                  Limitations ({r.limitations.length})
                </summary>
                <ul className="small ink2" style={{ paddingLeft: 18 }}>
                  {(r.limitations as string[]).map((l, i) => (
                    <li key={i}>{l}</li>
                  ))}
                </ul>
              </details>
            </div>
          )}
        </Panel>
        <Panel title="Human-readable report" sub="report.md written by the engine">
          {md ? <pre className="mdview">{md}</pre> : <Empty>No report yet.</Empty>}
        </Panel>
      </div>

      <Panel
        title="Artifacts"
        sub="only files that exist on the server are listed"
        right={
          rid && (
            <div className="row tiny">
              <label className="row" style={{ gap: 4 }}>
                <input type="radio" checked={scope === "active"} onChange={() => setScope("active")} />
                active reconstruction
              </label>
              <label className="row" style={{ gap: 4 }}>
                <input type="radio" checked={scope === "all"} onChange={() => setScope("all")} />
                all
              </label>
            </div>
          )
        }
        flush
      >
        {arts.error && <div className="callout bad">{arts.error}</div>}
        {!arts.data ? (
          <div className="panel-body">
            <div className="skeleton" style={{ height: 120 }} />
          </div>
        ) : (
          CATS.map(([cat, title]) => {
            const items = list.filter((a) => a.category === cat);
            if (!items.length) return null;
            return (
              <div key={cat}>
                <div className="panel-head" style={{ background: "var(--surface-2)" }}>
                  <span className="eyebrow">{title}</span>
                  {cat === "counterfactual" && <Badge tone="hypo">hypothetical</Badge>}
                </div>
                <table className="data">
                  <tbody>
                    {items.map((a) => (
                      <ArtifactRow key={a.path} run={run} a={a} />
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })
        )}
      </Panel>
    </div>
  );
}

function ArtifactRow({ run, a }: { run: Run; a: Artifact }) {
  return (
    <tr>
      <td className="text" style={{ width: "34%" }}>{a.label}</td>
      <td className="tiny muted">{a.path}</td>
      <td style={{ width: 90, textAlign: "right" }}>{fmt.bytes(a.size_bytes)}</td>
      <td style={{ width: 170, textAlign: "right" }}>
        <a className="btn sm ghost" href={fileUrl(run.id, a.path)} target="_blank" rel="noreferrer">
          View
        </a>{" "}
        <a className="btn sm" href={fileUrl(run.id, a.path, true)}>
          Download
        </a>
      </td>
    </tr>
  );
}
