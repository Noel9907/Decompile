import { useEffect, useMemo, useState } from "react";
import { api, useAsync, videoUrl } from "../api";
import BoxPrompt, { type Box } from "../components/BoxPrompt";
import VideoPlayer from "../components/VideoPlayer";
import type { Health, Plan, Run, RunEvent } from "../types";
import { Badge, Panel, RECON_ORDER, StageList, StatusIcon, TRACKING_ORDER, fmt } from "../ui";

export default function ProcessTab({
  run, health, events, refresh, activeRecon, onOpenRecon,
}: {
  run: Run;
  health: Health | null;
  events: RunEvent[];
  refresh: () => void;
  activeRecon: string | null;
  onOpenRecon: (rid: string) => void;
}) {
  const ft = useAsync(() => (run.state === "ingesting" || run.state === "rejected" ? Promise.resolve(null) : api.frameTable(run.id)), [run.id, run.state === "ingesting"]);
  const latestRecon = run.reconstructions[run.reconstructions.length - 1];
  const logs = events.filter((e) => e.type === "log" && e.level !== "debug").slice(-6);
  const jobErr = run.job?.status === "failed" ? run.job.error : null;

  return (
    <div className="stack">
      <div className="split">
        <Panel title="Input video" sub={run.state === "ingesting" ? "validating…" : "uploaded to the Decompile server"}>
          <div className="col">
            <VideoPlayer src={videoUrl(run.id)} times={ft.data?.t_s} />
            {run.video && (
              <dl className="kv">
                <dt>File</dt>
                <dd>{run.original_filename}</dd>
                <dt>Size</dt>
                <dd>{fmt.bytes(run.video.size_bytes)}</dd>
                <dt>Duration</dt>
                <dd>{fmt.num(run.video.duration_s, 3)} s</dd>
                <dt>Resolution (displayed)</dt>
                <dd>{run.video.display_width ? `${run.video.display_width}×${run.video.display_height}` : `${run.video.width}×${run.video.height}`}</dd>
                <dt>Frames (decoded)</dt>
                <dd>{run.video.frame_count ?? run.video.decoded_frame_count ?? "—"}</dd>
                <dt>Codec · nominal fps</dt>
                <dd>
                  {run.video.codec} · {fmt.fps(run.video.nominal_fps)}
                </dd>
                <dt>Time base</dt>
                <dd>{run.video.time_base ?? "—"}</dd>
                <dt>SHA-256</dt>
                <dd title={run.video.sha256}>{run.video.sha256.slice(0, 16)}…</dd>
              </dl>
            )}
          </div>
        </Panel>

        <div className="stack">
          <Suitability run={run} />
          <Panel title="Pipeline" sub="actual stage status reported by the backend">
            <StageList stages={run.stages} order={TRACKING_ORDER} />
            {latestRecon && (
              <>
                <div className="divider" />
                <div className="row tiny muted" style={{ marginBottom: 4 }}>
                  <span className="eyebrow">Reconstruction {latestRecon.id}</span>
                  <span className="grow" />
                  <span className="mono">{latestRecon.params.physics_model}</span>
                </div>
                <StageList stages={latestRecon.stages} order={RECON_ORDER} />
              </>
            )}
            {jobErr && (
              <div className="callout bad" role="alert" style={{ marginTop: 12 }}>
                <strong>{run.job?.kind?.replace("_", " ")} stopped.</strong> {jobErr}
                {run.job?.details?.length ? (
                  <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                    {run.job.details.slice(0, 6).map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )}
            {logs.length > 0 && (
              <pre className="mdview" style={{ marginTop: 12, maxHeight: 120 }}>
                {logs.map((l) => `${String(l.t).slice(11, 19)}  ${l.message}`).join("\n")}
              </pre>
            )}
          </Panel>
        </div>
      </div>

      {run.state !== "ingesting" && run.state !== "rejected" && (
        <SamPanel run={run} health={health} refresh={refresh} frameCount={ft.data?.frame_count ?? run.video?.frame_count ?? 0} />
      )}
      {run.sam.status === "completed" && (
        <ReconstructPanel run={run} health={health} refresh={refresh} activeRecon={activeRecon} onOpenRecon={onOpenRecon} />
      )}
    </div>
  );
}

function Suitability({ run }: { run: Run }) {
  if (run.state === "ingesting")
    return (
      <Panel title="Suitability">
        <div className="progress indeterminate">
          <div />
        </div>
        <p className="small muted" style={{ marginTop: 10 }}>
          Probing, decoding every frame and validating presentation timestamps…
        </p>
      </Panel>
    );
  if (run.state === "rejected")
    return (
      <Panel title="Suitability">
        <div className="callout bad" role="alert">
          <strong>This video cannot be processed.</strong>
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {run.rejection_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      </Panel>
    );
  const s = run.suitability;
  if (!s) return null;
  return (
    <Panel
      title="Suitability"
      right={<Badge tone={s.suitable ? "good" : "bad"}>{s.suitable ? "suitable for the implemented path" : "not suitable"}</Badge>}
    >
      <ul className="stages">
        {s.checks.map((c) => (
          <li key={c.label}>
            <StatusIcon status={c.ok ? "done" : c.level === "required" ? "failed" : "warn"} />
            <div>
              <div className="title">{c.label}</div>
              <div className="detail">{c.detail}</div>
            </div>
            <span className="time">{c.level}</span>
          </li>
        ))}
      </ul>
      {s.warnings.map((w, i) => (
        <div key={i} className="callout warn small" style={{ marginTop: 8 }}>
          {w}
        </div>
      ))}
      <p className="tiny muted" style={{ marginTop: 10 }}>
        {s.scope}
      </p>
    </Panel>
  );
}

// ------------------------------------------------------------------ SAM

function SamPanel({ run, health, refresh, frameCount }: { run: Run; health: Health | null; refresh: () => void; frameCount: number }) {
  const [mode, setMode] = useState<"concept" | "box">("concept");
  const [text, setText] = useState("ball");
  const [frame, setFrame] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const plan = run.plan;
  const sam = run.sam;
  const done = sam.status === "completed";
  const running = sam.status === "running";
  const liveOk = !!health?.live_sam_enabled && !!health?.api_key_configured;
  const W = run.video?.display_width ?? 0;
  const H = run.video?.display_height ?? 0;

  const makePlan = async () => {
    setBusy(true);
    setErr(null);
    try {
      const prompt = mode === "concept" ? { kind: "concept", text } : { kind: "objects", frame, box, points: [] };
      if (mode === "box" && !box) throw new Error("Draw a box around the object first.");
      await api.plan(run.id, prompt);
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const doImport = async (f: File | undefined) => {
    if (!f) return;
    setBusy(true);
    setErr(null);
    try {
      await api.importSam(run.id, f);
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel
      title="Step 1 · Perception with Meta SAM 3.1"
      sub="segmentation and identity tracking (the only billed step)"
      right={
        done ? (
          <Badge tone="good">{sam.source === "imported" ? "SAM result imported" : "SAM tracking completed"}</Badge>
        ) : running ? (
          <Badge tone="info">SAM request running</Badge>
        ) : sam.status === "failed" || sam.status === "interrupted" ? (
          <Badge tone="bad">SAM {sam.status}</Badge>
        ) : (
          <Badge>no request made</Badge>
        )
      }
    >
      {!done && (
        <div className="grid-2">
          <div className="col">
            <div className="row">
              <button className={`btn sm ${mode === "concept" ? "primary" : ""}`} onClick={() => setMode("concept")} disabled={running}>
                Concept phrase
              </button>
              <button className={`btn sm ${mode === "box" ? "primary" : ""}`} onClick={() => setMode("box")} disabled={running || !W}>
                Box on a frame
              </button>
            </div>
            {mode === "concept" ? (
              <label className="field">
                <span>Concept prompt (one short noun phrase)</span>
                <input className="input" value={text} maxLength={60} onChange={(e) => setText(e.target.value)} placeholder="tennis ball" />
              </label>
            ) : (
              <BoxPrompt runId={run.id} frameCount={frameCount} width={W} height={H} frame={frame}
                setFrame={(f) => { setFrame(f); setBox(null); }} box={box} setBox={setBox} />
            )}
            <div className="row">
              <button className="btn" onClick={makePlan} disabled={busy || running} data-testid="make-plan">
                Estimate cost — no request
              </button>
              <span className="tiny muted">Builds and validates the exact request offline.</span>
            </div>
          </div>
          <div className="col">
            {plan ? <PlanCard plan={plan} /> : <div className="empty">Create a plan to see the frames SAM will process and the estimated cost.</div>}
            {plan && (
              <div className="col">
                {!liveOk && (
                  <div className="callout warn small">
                    {health?.live_sam_enabled
                      ? "The server has no MODEL_API_KEY, so it cannot make a live SAM request."
                      : "Live SAM requests are disabled on this server (start it with DECOMPILE_LIVE_SAM=1). Nothing will be sent."}
                  </div>
                )}
                <button className="btn danger" disabled={!liveOk || running || busy} onClick={() => setConfirming(true)} data-testid="run-sam">
                  Review and run SAM 3.1 (billed)…
                </button>
              </div>
            )}
            <div className="divider" />
            <div className="col">
              <div className="small ink2">Already have a SAM result for this exact video?</div>
              <label className="btn sm" style={{ alignSelf: "flex-start" }}>
                Import sam_tracking.json
                <input type="file" accept="application/json,.json" hidden onChange={(e) => doImport(e.target.files?.[0])} disabled={busy || running} />
              </label>
              <span className="tiny muted">
                Produced by <code>tools/sam_demo.py track</code>. Verified against this video's SHA-256 and frame table. No API call, no
                charge.
              </span>
            </div>
          </div>
        </div>
      )}
      {err && (
        <div className="callout bad" role="alert" style={{ marginTop: 12 }}>
          {err}
        </div>
      )}
      <SamOutcome run={run} />
      {confirming && plan && (
        <ConfirmModal
          plan={plan}
          run={run}
          onCancel={() => setConfirming(false)}
          onConfirmed={() => {
            setConfirming(false);
            refresh();
          }}
        />
      )}
    </Panel>
  );
}

export function PlanCard({ plan }: { plan: Plan }) {
  const c = plan.cost;
  const prompt = plan.prompt.kind === "concept" ? `concept “${plan.prompt.text}”` : `object prompt ${plan.prompt.input_text ?? ""}`;
  return (
    <div className="col" data-testid="plan-card">
      <div className="readouts">
        <div className="readout">
          <span className="label">Frames processed</span>
          <span className="value">{plan.expected_processed_frames}</span>
          <span className="note">every frame of the file</span>
        </div>
        <div className="readout">
          <span className="label">Billed frames</span>
          <span className="value">{c.expected_billed_frames}</span>
          <span className="note">minimum {c.billing_minimum_frames} per video</span>
        </div>
        <div className="readout">
          <span className="label">Estimated cost</span>
          <span className="value">{fmt.usd(c.expected_cost_usd)}</span>
          <span className="note">${c.usd_per_1000_frames.toFixed(2)} / 1,000 frames</span>
        </div>
      </div>
      <dl className="kv">
        <dt>Prompt</dt>
        <dd>{prompt}</dd>
        <dt>Media</dt>
        <dd>
          {plan.media.displayed_width}×{plan.media.displayed_height}, PTS {plan.media.first_t_s.toFixed(3)}–{plan.media.last_t_s.toFixed(3)} s
        </dd>
        <dt>Plan hash</dt>
        <dd>{plan.sha256.slice(0, 16)}…</dd>
      </dl>
      <p className="tiny muted">
        Billing rule: SAM bills every processed frame with a minimum of {c.billing_minimum_frames} frames per video request. {c.caveat}. Source:{" "}
        {c.pricing_source}.
      </p>
      {plan.warnings.map((w, i) => (
        <div key={i} className="callout warn small">
          {w}
        </div>
      ))}
    </div>
  );
}

export function ConfirmModal({ plan, run, onCancel, onConfirmed }: { plan: Plan; run: Run; onCancel: () => void; onConfirmed: () => void }) {
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const prior = run.sam.attempts.filter((a) => a.request_sent);
  return (
    <div className="modal-back" role="dialog" aria-modal="true" aria-label="Confirm billed SAM request">
      <div className="modal">
        <header className="panel-head">
          <h3>Confirm billed request</h3>
        </header>
        <div className="panel-body">
          <p className="ink2">
            Decompile will upload <strong>{run.original_filename}</strong> ({fmt.bytes(run.video?.size_bytes)}) to Meta's Model API and request SAM 3.1
            segmentation of <strong>{plan.expected_processed_frames} frames</strong>.
          </p>
          <div className="readouts">
            <div className="readout">
              <span className="label">Billed frames (est.)</span>
              <span className="value">{plan.cost.expected_billed_frames}</span>
            </div>
            <div className="readout">
              <span className="label">Estimated cost</span>
              <span className="value">{fmt.usd(plan.cost.expected_cost_usd)}</span>
            </div>
          </div>
          {prior.length > 0 && (
            <div className="callout warn small">
              {prior.length} earlier request(s) for this run were sent and may already have been billed. Confirming makes a new request.
            </div>
          )}
          <p className="tiny muted">
            One request, never retried automatically after it is sent. The API key stays on the server. The response's
            usage.video_frames_processed is recorded as the authoritative billed amount.
          </p>
          <label className="row small">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} data-testid="ack" />
            I approve a billed SAM 3.1 request of about {fmt.usd(plan.cost.expected_cost_usd)}.
          </label>
          {err && <div className="callout bad small">{err}</div>}
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <button className="btn ghost" onClick={onCancel} disabled={busy}>
              Cancel
            </button>
            <button
              className="btn danger"
              disabled={!ack || busy}
              data-testid="confirm-sam"
              onClick={async () => {
                setBusy(true);
                setErr(null);
                try {
                  await api.confirmSam(run.id, plan);
                  onConfirmed();
                } catch (e) {
                  setErr((e as Error).message);
                  setBusy(false);
                }
              }}
            >
              Upload and run SAM 3.1
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function SamOutcome({ run }: { run: Run }) {
  const sam = run.sam;
  if (!sam.attempts.length && !sam.result && !sam.warnings.length) return null;
  return (
    <div className="col" style={{ marginTop: 16 }}>
      {sam.attempts.map((a) => (
        <div key={a.n} className={`callout ${a.status === "completed" ? "good" : a.status === "running" ? "info" : "bad"} small`}>
          <strong>
            Live request #{a.n}: {a.status}
          </strong>
          {a.usage && (
            <>
              {" "}
              · usage <code>{JSON.stringify(a.usage)}</code>
            </>
          )}
          {a.actual_cost_usd !== undefined && a.actual_cost_usd !== null && <> · actual cost {fmt.usd(a.actual_cost_usd)} (from reported usage)</>}
          {a.error && <div>{a.error}</div>}
          {a.billing_note && <div>{a.billing_note}</div>}
        </div>
      ))}
      {sam.warnings.map((w, i) => (
        <div key={i} className="callout warn small">
          {w}
        </div>
      ))}
      {sam.result && (
        <dl className="kv">
          <dt>Model · outcome</dt>
          <dd>
            {sam.result.model} · {sam.result.outcome.status}
          </dd>
          <dt>Prompt</dt>
          <dd>{JSON.stringify(sam.result.prompt)}</dd>
          <dt>Objects · frames with records</dt>
          <dd>
            {sam.result.summary.n_objects} · {sam.result.summary.n_frames_with_records}
          </dd>
          <dt>Reported usage</dt>
          <dd>{sam.result.usage ? JSON.stringify(sam.result.usage) : "—"}</dd>
          {typeof sam.result.provenance?.response_id === "string" && (
            <>
              <dt>Response id</dt>
              <dd>{sam.result.provenance.response_id as string}</dd>
            </>
          )}
          <dt>Result SHA-256</dt>
          <dd>{sam.result.sha256.slice(0, 16)}…</dd>
        </dl>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ reconstruction setup

function ReconstructPanel({
  run, health, refresh, activeRecon, onOpenRecon,
}: {
  run: Run;
  health: Health | null;
  refresh: () => void;
  activeRecon: string | null;
  onOpenRecon: (rid: string) => void;
}) {
  const tracks = useAsync(() => api.tracks(run.id), [run.id, run.sam.result?.sha256]);
  const n = run.video?.frame_count ?? 0;
  const maxWin = health?.max_window_frames ?? 300;
  const ids = useMemo(() => Object.keys(tracks.data?.objects ?? {}), [tracks.data]);
  const [obj, setObj] = useState("most_frames");
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState<number>(Math.min(n, maxWin));
  const [model, setModel] = useState<"projective" | "image_space">("projective");
  const [hfov, setHfov] = useState(66);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const running = run.job?.status === "running" && run.job.kind === "reconstruction";

  useEffect(() => setEnd(Math.min(n, maxWin)), [n, maxWin]);

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api.reconstruct(run.id, { sam_object: obj, start_frame: start, end_frame: end >= n ? null : end, physics_model: model, hfov_deg: hfov });
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const lane = (oid: string) => {
    const fr = tracks.data!.objects[oid];
    return (
      <svg width="100%" height="14" viewBox={`0 0 ${n} 14`} preserveAspectRatio="none" aria-label={`frames with object ${oid}`}>
        <rect x={start} y={0} width={Math.max(0, end - start)} height={14} fill="rgba(233,231,225,0.06)" />
        {fr.map((r) => (
          <rect key={r.f} x={r.f + 0.08} y={3} width={0.84} height={8} fill={r.mask ? "#3987e5" : "#565a5d"} />
        ))}
      </svg>
    );
  };

  return (
    <Panel title="Step 2 · Select the object and reconstruct" sub="offline: measurement → physics → scene program → simulation (no API call)">
      {tracks.error && <div className="callout bad">{tracks.error}</div>}
      {tracks.data && (
        <div className="grid-2">
          <div className="col">
            <table className="data">
              <thead>
                <tr>
                  <th></th>
                  <th>SAM object</th>
                  <th>Frames</th>
                  <th>Range</th>
                  <th>Median conf.</th>
                  <th style={{ width: "38%" }}>Presence over the video</th>
                </tr>
              </thead>
              <tbody>
                <tr className="click" onClick={() => setObj("most_frames")}>
                  <td>
                    <input type="radio" checked={obj === "most_frames"} readOnly />
                  </td>
                  <td className="text" colSpan={5}>
                    Automatic — the object with the most frames in the window <span className="muted">(deterministic rule)</span>
                  </td>
                </tr>
                {ids.map((id) => {
                  const s = tracks.data!.summary[id];
                  return (
                    <tr key={id} className={`click ${obj === id ? "sel" : ""}`} onClick={() => setObj(id)}>
                      <td>
                        <input type="radio" checked={obj === id} readOnly aria-label={`object ${id}`} />
                      </td>
                      <td>#{id}</td>
                      <td>{s.n_frames}</td>
                      <td>
                        {s.first_frame}–{s.last_frame}
                      </td>
                      <td>{fmt.num(s.median_confidence, 2)}</td>
                      <td>{lane(id)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="tiny muted">Blue ticks: frames where SAM returned a mask for the object. Shaded: the reconstruction window.</p>
          </div>
          <div className="col">
            <div className="row">
              <label className="field grow">
                <span>Window start frame</span>
                <input className="input" type="number" min={0} max={n - 1} value={start} onChange={(e) => setStart(Math.max(0, +e.target.value))} />
              </label>
              <label className="field grow">
                <span>Window end (exclusive)</span>
                <input className="input" type="number" min={1} max={n} value={end} onChange={(e) => setEnd(Math.min(n, +e.target.value))} />
              </label>
            </div>
            <p className="tiny muted">
              {end - start} frames (4–{maxWin}). For the implemented physics the window should contain exactly one bounce.
            </p>
            <div className="row">
              <label className="field grow">
                <span>Physics model</span>
                <select className="input" value={model} onChange={(e) => setModel(e.target.value as any)}>
                  <option value="projective">Projective (perspective-aware, 3-D)</option>
                  <option value="image_space">Image space (pixels, 2-D)</option>
                </select>
              </label>
              {model === "projective" && (
                <label className="field" style={{ width: 150 }}>
                  <span>Camera HFOV prior (°)</span>
                  <input className="input" type="number" min={20} max={140} value={hfov} onChange={(e) => setHfov(+e.target.value)} />
                </label>
              )}
            </div>
            <button className="btn primary" onClick={submit} disabled={busy || running} data-testid="reconstruct">
              {running ? "Reconstructing…" : "Reconstruct (offline, no API call)"}
            </button>
            {err && <div className="callout bad small">{err}</div>}
          </div>
        </div>
      )}
      {run.reconstructions.length > 0 && (
        <>
          <div className="divider" />
          <table className="data">
            <thead>
              <tr>
                <th>Id</th>
                <th>Object</th>
                <th>Window</th>
                <th>Model</th>
                <th>Status</th>
                <th>Outcome</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {[...run.reconstructions].reverse().map((r) => (
                <tr key={r.id} className={r.id === activeRecon ? "sel" : ""}>
                  <td>{r.id}</td>
                  <td>{r.params.sam_object}</td>
                  <td>
                    {r.params.start_frame}–{r.params.end_frame ?? "end"}
                  </td>
                  <td>{r.params.physics_model}</td>
                  <td className="text">
                    <span className="row" style={{ gap: 6 }}>
                      <StatusIcon status={r.status === "completed" ? "done" : r.status} size={14} />
                      {r.status}
                    </span>
                  </td>
                  <td className="text">
                    {r.outcome === "scene_program" ? (
                      <Badge tone="good">scene program</Badge>
                    ) : r.outcome === "measurements_only" ? (
                      <Badge tone="warn">measurements only</Badge>
                    ) : r.error ? (
                      <span className="tiny" style={{ color: "#ea7f7f" }}>{r.error}</span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {r.status === "completed" && (
                      <button className="btn sm" onClick={() => onOpenRecon(r.id)}>
                        Open
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}
