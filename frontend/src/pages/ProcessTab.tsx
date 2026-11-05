import { useEffect, useMemo, useState } from "react";
import { api, useAsync, videoUrl } from "../api";
import { navigate } from "../App";
import BoxPrompt, { type Box } from "../components/BoxPrompt";
import VideoPlayer from "../components/VideoPlayer";
import type { Health, Plan, Run, RunEvent } from "../types";
import { Badge, RECON_ORDER, StageList, StatusIcon, TRACKING_ORDER, fmt } from "../ui";

type Tracks = Awaited<ReturnType<typeof api.tracks>>;

/** Suggest a window around the first bounce, from SAM box centres (a UI hint for the window; physics comes later). */
export function suggestWindow(tracks: Tracks, objectId: string | null, maxWin: number): { start: number; end: number; bounce: number } | null {
  const ids = Object.keys(tracks.objects);
  const oid = objectId && tracks.objects[objectId] ? objectId : ids.sort((a, b) => tracks.objects[b].length - tracks.objects[a].length)[0];
  if (!oid) return null;
  const pts = tracks.objects[oid].map((r) => ({ f: r.f, y: (r.box[1] + r.box[3]) / 2 }));
  if (pts.length < 8) return null;
  const n = tracks.frame_count;
  // bounces = local maxima of y (image y grows downwards) with a clear rise on both sides
  const bounces: number[] = [];
  for (let i = 2; i < pts.length - 2; i++) {
    const y = pts[i].y;
    if (y >= pts[i - 1].y && y >= pts[i + 1].y && y - pts[i - 2].y > 2 && y - pts[i + 2].y > 2) bounces.push(i);
  }
  if (!bounces.length) return null;
  // window = between the neighbouring bounces (4-frame margin); choose the bounce best covered on both sides
  let best: { start: number; end: number; bounce: number; score: number } | null = null;
  bounces.forEach((bi, k) => {
    const bf = pts[bi].f;
    const s0 = k > 0 ? pts[bounces[k - 1]].f + 4 : pts[0].f;
    const e0 = k < bounces.length - 1 ? pts[bounces[k + 1]].f - 3 : pts[pts.length - 1].f + 1;
    const score = Math.min(bf - s0, e0 - bf);
    if (!best || score > best.score) best = { start: s0, end: e0, bounce: bf, score };
  });
  const chosen = best as unknown as { start: number; end: number; bounce: number; score: number };
  let start = Math.max(0, chosen.start);
  let end = Math.min(n, chosen.end);
  const bounce = chosen.bounce;
  if (end - start > maxWin) end = start + maxWin;
  return { start, end, bounce };
}

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
  const logs = events.filter((e) => e.type === "log" && e.level !== "debug").slice(-5);
  const jobErr = run.job?.status === "failed" ? run.job.error : null;
  const samDone = run.sam.status === "completed";

  return (
    <div className="stack">
      <NextStep run={run} onOpenRecon={onOpenRecon} />
      <div className="side-r">
        <div className="col" style={{ gap: 20 }}>
          {run.state !== "ingesting" && run.state !== "rejected" && (
            <SamStep run={run} health={health} refresh={refresh} frameCount={ft.data?.frame_count ?? run.video?.frame_count ?? 0} />
          )}
          {samDone && <ReconstructStep run={run} health={health} refresh={refresh} activeRecon={activeRecon} onOpenRecon={onOpenRecon} />}
        </div>
        <div className="col sticky" style={{ gap: 20 }}>
          <div className="card" style={{ padding: 14 }}>
            <VideoPlayer src={videoUrl(run.id)} times={ft.data?.t_s} />
            {run.video && (
              <dl className="kv" style={{ marginTop: 14, padding: "0 4px" }}>
                <dt>Size</dt><dd>{fmt.bytes(run.video.size_bytes)}</dd>
                <dt>Duration</dt><dd>{fmt.num(run.video.duration_s, 2)} s</dd>
                <dt>Resolution</dt><dd>{run.video.display_width ?? run.video.width}×{run.video.display_height ?? run.video.height}</dd>
                <dt>Frames</dt><dd>{run.video.frame_count ?? run.video.decoded_frame_count ?? "—"} @ {fmt.fps(run.video.nominal_fps)} fps</dd>
              </dl>
            )}
            <Suitability run={run} />
          </div>
          <div className="card card-pad">
            <div className="h3" style={{ marginBottom: 6 }}>Progress</div>
            <StageList stages={run.stages} order={TRACKING_ORDER} />
            {latestRecon && (
              <>
                <div className="divider" />
                <div className="tiny muted" style={{ marginBottom: 2 }}>Decompile ({latestRecon.id})</div>
                <StageList stages={latestRecon.stages} order={RECON_ORDER} />
              </>
            )}
            {jobErr && (
              <div className="callout bad small" role="alert" style={{ marginTop: 12 }}>
                <strong>Stopped.</strong> {jobErr}
              </div>
            )}
            {logs.length > 0 && run.job?.status === "running" && (
              <pre className="mdview" style={{ marginTop: 12, maxHeight: 100, fontSize: 11 }}>{logs.map((l) => `${l.message}`).join("\n")}</pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function NextStep({ run, onOpenRecon }: { run: Run; onOpenRecon: (rid: string) => void }) {
  const latest = run.reconstructions[run.reconstructions.length - 1];
  const running = run.job?.status === "running";
  let title: string, body: string, tone = "", action: React.ReactNode = null, busy = false;
  if (run.state === "ingesting") {
    [title, body, busy] = ["Checking your video…", "Decoding every frame and validating its timestamps. This takes a few seconds.", true];
  } else if (run.state === "rejected") {
    [title, body, tone] = ["This video can't be used", run.rejection_reasons.join("; "), "bad"];
    action = <button className="btn primary" onClick={() => navigate("/")}>Upload another video</button>;
  } else if (run.sam.status === "running") {
    [title, body, busy] = ["SAM 3.1 is tracking your video…", "Meta's model is segmenting the object in every frame. Usually under a minute.", true];
  } else if (run.sam.status !== "completed") {
    [title, body] = ["Step 1 of 2 · Tell SAM 3.1 what to track", "Name the object (for example “ball”) or draw a box around it, then start tracking."];
  } else if (running && run.job?.kind === "reconstruction") {
    [title, body] = ["Decompiling…", "Measuring every frame, fitting physics and building the program. About 30–60 seconds."];
    busy = true;
  } else if (latest?.outcome === "scene_program") {
    [title, body, tone] = ["Done. Your video is decompiled.", "The physics is now a program you can read, run and rewrite.", "good"];
    action = <button className="btn primary lg" onClick={() => onOpenRecon(latest.id)}>See the result →</button>;
  } else if (latest && latest.status !== "running") {
    [title, body, tone] = ["Not decompiled. Try another window.", latest.error ?? "The physics checks did not accept this window. Choose a window around exactly one bounce.", "warn"];
  } else {
    [title, body] = ["Step 2 of 2 · Decompile the bounce", "We suggest a window around the first bounce. Check it and press Decompile."];
  }
  return (
    <div className={`card card-pad ${tone === "good" ? "" : ""}`} style={{ borderColor: tone === "good" ? "rgba(12,163,12,0.35)" : tone === "bad" ? "rgba(208,59,59,0.4)" : tone === "warn" ? "rgba(250,178,25,0.3)" : undefined }}>
      <div className="row wrap" style={{ gap: 20 }}>
        <div className="grow" style={{ minWidth: 280 }}>
          <div className="h2">{title}</div>
          <p className="ink2" style={{ marginTop: 6 }}>{body}</p>
        </div>
        {action}
      </div>
      {busy && <div className="progress indeterminate" style={{ marginTop: 16 }}><div /></div>}
    </div>
  );
}

function Suitability({ run }: { run: Run }) {
  const s = run.suitability;
  if (!s) return null;
  const failed = s.checks.filter((c) => !c.ok);
  return (
    <details className="more" style={{ marginTop: 14, borderTop: "1px solid var(--line)" }}>
      <summary style={{ padding: "12px 4px" }}>
        <StatusIcon status={s.suitable ? "done" : "failed"} size={15} />
        {s.suitable ? "Video checks passed" : `${failed.length} check(s) failed`}
      </summary>
      <div style={{ padding: "0 4px 6px" }}>
        <ul className="stages">
          {s.checks.map((c) => (
            <li key={c.label}>
              <StatusIcon status={c.ok ? "done" : c.level === "required" ? "failed" : "warn"} size={15} />
              <div><div className="title small">{c.label}</div><div className="detail">{c.detail}</div></div>
              <span />
            </li>
          ))}
        </ul>
        <p className="tiny muted" style={{ marginTop: 8 }}>{s.scope}</p>
      </div>
    </details>
  );
}

// ------------------------------------------------------------------ Step 1: SAM

function SamStep({ run, health, refresh, frameCount }: { run: Run; health: Health | null; refresh: () => void; frameCount: number }) {
  const [mode, setMode] = useState<"concept" | "box">("concept");
  const [text, setText] = useState("ball");
  const [frame, setFrame] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const sam = run.sam;
  const done = sam.status === "completed";
  const running = sam.status === "running";
  const liveOk = !!health?.live_sam_enabled && !!health?.api_key_configured;
  const W = run.video?.display_width ?? 0;
  const H = run.video?.display_height ?? 0;

  const start = async () => {
    setBusy(true);
    setErr(null);
    try {
      if (mode === "box" && !box) throw new Error("Draw a box around the object first.");
      const prompt = mode === "concept" ? { kind: "concept", text } : { kind: "objects", frame, box, points: [] };
      await api.plan(run.id, prompt);
      refresh();
      setConfirming(true);
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

  if (done)
    return (
      <div className="card card-pad">
        <div className="row">
          <span className="n-badge"><StatusIcon status="done" /></span>
          <div className="grow">
            <div className="h3">1 · Tracked with SAM 3.1</div>
            <div className="small muted" style={{ marginTop: 2 }}>
              {sam.result?.summary.n_objects} object(s) found in {sam.result?.summary.n_frames_with_records} frames · prompt “{String(sam.result?.prompt?.text ?? "object prompt")}”
              {sam.source === "imported" ? " · imported result (no new charge)" : ""}
              {sam.attempts.at(-1)?.actual_cost_usd != null ? ` · cost ${fmt.usd(sam.attempts.at(-1)!.actual_cost_usd)}` : ""}
            </div>
          </div>
          <Badge tone="good">done</Badge>
        </div>
      </div>
    );

  return (
    <div className="card">
      <div className="card-head">
        <span className="t">1 · What should SAM 3.1 track?</span>
        <span className="grow" />
        <div className="seg">
          <button className={mode === "concept" ? "on" : ""} onClick={() => setMode("concept")} disabled={running}>Name it</button>
          <button className={mode === "box" ? "on" : ""} onClick={() => setMode("box")} disabled={running || !W}>Draw a box</button>
        </div>
      </div>
      <div className="card-body">
        {mode === "concept" ? (
          <form className="row" onSubmit={(e) => { e.preventDefault(); start(); }}>
            <input className="input" style={{ flex: 1, height: 48, fontSize: 16 }} value={text} maxLength={60} onChange={(e) => setText(e.target.value)}
              placeholder="e.g. tennis ball" aria-label="object to track" disabled={running} />
          </form>
        ) : (
          <BoxPrompt runId={run.id} frameCount={frameCount} width={W} height={H} frame={frame} setFrame={(f) => { setFrame(f); setBox(null); }} box={box} setBox={setBox} />
        )}
        <div className="row wrap" style={{ marginTop: 16 }}>
          <button className="btn primary lg" onClick={start} disabled={busy || running || !liveOk} data-testid="run-sam">
            {running ? "Tracking…" : "Track with SAM 3.1"}
          </button>
          <span className="small muted">
            {liveOk ? `You'll see the exact cost (about ${fmt.usd(Math.max(50, frameCount) * 0.0002)}) and confirm before anything is sent.` : "Live SAM is disabled on this server."}
          </span>
        </div>
        {err && <div className="callout bad small" style={{ marginTop: 14 }}>{err}</div>}
        <SamOutcome run={run} />
      </div>
      <details className="more">
        <summary>Already have a SAM result for this exact video? Import it (no API call)</summary>
        <div className="inner">
          <label className="btn sm">
            Choose sam_tracking.json
            <input type="file" accept="application/json,.json" hidden onChange={(e) => doImport(e.target.files?.[0])} disabled={busy || running} />
          </label>
          <p className="tiny muted" style={{ marginTop: 8 }}>From <code>tools/sam_demo.py track</code>. Checked against this video's SHA-256 and frame table.</p>
        </div>
      </details>
      {confirming && run.plan && (
        <ConfirmModal plan={run.plan} run={run} onCancel={() => setConfirming(false)} onConfirmed={() => { setConfirming(false); refresh(); }} />
      )}
    </div>
  );
}

export function PlanCard({ plan }: { plan: Plan }) {
  const c = plan.cost;
  return (
    <div className="col" data-testid="plan-card">
      <div className="readouts">
        <div className="readout"><span className="label">Frames processed</span><span className="value">{plan.expected_processed_frames}</span></div>
        <div className="readout"><span className="label">Billed frames</span><span className="value">{c.expected_billed_frames}</span><span className="note">minimum {c.billing_minimum_frames}</span></div>
        <div className="readout"><span className="label">Estimated cost</span><span className="value">{fmt.usd(c.expected_cost_usd)}</span></div>
      </div>
    </div>
  );
}

export function ConfirmModal({ plan, run, onCancel, onConfirmed }: { plan: Plan; run: Run; onCancel: () => void; onConfirmed: () => void }) {
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const prior = run.sam.attempts.filter((a) => a.request_sent);
  const prompt = plan.prompt.kind === "concept" ? `“${plan.prompt.text}”` : "your box prompt";
  return (
    <div className="modal-back" role="dialog" aria-modal="true" aria-label="Confirm billed SAM request">
      <div className="modal">
        <header className="panel-head" style={{ paddingTop: 22 }}>
          <h3 style={{ fontSize: 18 }}>Track {prompt} with SAM 3.1?</h3>
        </header>
        <div className="panel-body">
          <p className="ink2">
            Decompile will upload <strong>{run.original_filename}</strong> ({fmt.bytes(run.video?.size_bytes)}) to Meta's Model API and segment all{" "}
            <strong>{plan.expected_processed_frames} frames</strong>.
          </p>
          <div className="readouts">
            <div className="readout">
              <span className="label">Billed frames (est.)</span>
              <span className="value">{plan.cost.expected_billed_frames}</span>
              <span className="note">minimum {plan.cost.billing_minimum_frames} per video</span>
            </div>
            <div className="readout">
              <span className="label">Estimated cost</span>
              <span className="value">{fmt.usd(plan.cost.expected_cost_usd)}</span>
              <span className="note">${plan.cost.usd_per_1000_frames.toFixed(2)} per 1,000 frames</span>
            </div>
          </div>
          {plan.warnings.map((w, i) => <div key={i} className="callout warn small">{w}</div>)}
          {prior.length > 0 && (
            <div className="callout warn small">
              {prior.length} earlier request(s) for this run were sent and may already have been billed. Confirming makes a new request.
            </div>
          )}
          <p className="tiny muted">One request, never retried automatically. Your API key stays on the server. The billed amount reported by the API is recorded.</p>
          <label className="row small">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} data-testid="ack" />
            I approve a billed SAM 3.1 request of about {fmt.usd(plan.cost.expected_cost_usd)}.
          </label>
          {err && <div className="callout bad small">{err}</div>}
          <div className="row" style={{ justifyContent: "flex-end" }}>
            <button className="btn ghost" onClick={onCancel} disabled={busy}>Cancel</button>
            <button
              className="btn primary"
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
  const failed = run.sam.attempts.filter((a) => a.status === "failed" || a.status === "interrupted");
  if (!failed.length && !run.sam.warnings.length) return null;
  return (
    <div className="col" style={{ marginTop: 14 }}>
      {failed.map((a) => (
        <div key={a.n} className="callout bad small">
          <strong>Request #{a.n} {a.status}.</strong> {a.error} {a.billing_note}
        </div>
      ))}
      {run.sam.warnings.map((w, i) => <div key={i} className="callout warn small">{w}</div>)}
    </div>
  );
}

// ------------------------------------------------------------------ Step 2: window + decompile

function ReconstructStep({
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
  const [obj, setObj] = useState<string | null>(null);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState<number>(Math.min(n, maxWin));
  const [model, setModel] = useState<"projective" | "image_space">("projective");
  const [hfov, setHfov] = useState(66);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const running = run.job?.status === "running" && run.job.kind === "reconstruction";
  const suggestion = useMemo(() => (tracks.data ? suggestWindow(tracks.data, obj, maxWin) : null), [tracks.data, obj, maxWin]);

  useEffect(() => {
    if (!tracks.data) return;
    if (obj === null) setObj(ids.sort((a, b) => tracks.data!.objects[b].length - tracks.data!.objects[a].length)[0] ?? null);
  }, [tracks.data, ids, obj]);
  useEffect(() => {
    if (suggestion) {
      setStart(suggestion.start);
      setEnd(suggestion.end);
    } else setEnd(Math.min(n, maxWin));
  }, [suggestion, n, maxWin]);

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api.reconstruct(run.id, { sam_object: obj ?? "most_frames", start_frame: start, end_frame: end >= n ? null : end, physics_model: model, hfov_deg: hfov });
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const plot = () => {
    if (!tracks.data || !obj) return null;
    const pts = tracks.data.objects[obj].map((r) => ({ f: r.f, y: (r.box[1] + r.box[3]) / 2 }));
    const H = tracks.data.height;
    const w = 1000, h = 150;
    const X = (f: number) => (f / Math.max(1, n - 1)) * w;
    const Y = (y: number) => (y / H) * h;
    return (
      <svg viewBox={`0 0 ${w} ${h + 22}`} width="100%" style={{ display: "block" }} aria-label="object height over the video">
        <rect x={X(start)} y={0} width={Math.max(2, X(end - 1) - X(start))} height={h} fill="rgba(57,135,229,0.10)" stroke="rgba(57,135,229,0.5)" rx={6} />
        <line x1={0} x2={w} y1={h} y2={h} stroke="rgba(255,255,255,0.1)" />
        <polyline points={pts.map((p) => `${X(p.f)},${Y(p.y)}`).join(" ")} fill="none" stroke="var(--ink-1)" strokeWidth={2} strokeLinejoin="round" />
        {pts.map((p) => <circle key={p.f} cx={X(p.f)} cy={Y(p.y)} r={2.6} fill={p.f >= start && p.f < end ? "var(--series-measured)" : "var(--ink-4)"} />)}
        {suggestion && <g><line x1={X(suggestion.bounce)} x2={X(suggestion.bounce)} y1={0} y2={h} stroke="var(--series-fitted)" strokeDasharray="4 4" /><text x={X(suggestion.bounce) + 6} y={14} fill="var(--series-fitted)" fontSize="13">bounce</text></g>}
        <text x={0} y={h + 18} fill="var(--ink-3)" fontSize="12">frame 0</text>
        <text x={w} y={h + 18} fill="var(--ink-3)" fontSize="12" textAnchor="end">frame {n - 1}</text>
      </svg>
    );
  };

  return (
    <div className="card">
      <div className="card-head">
        <span className="t">2 · Choose the bounce and decompile</span>
        <span className="s">offline · no API cost</span>
      </div>
      <div className="card-body">
        {tracks.error && <div className="callout bad">{tracks.error}</div>}
        {tracks.data && (
          <div className="col" style={{ gap: 16 }}>
            {ids.length > 1 && (
              <div className="row wrap small">
                <span className="muted">SAM found {ids.length} objects. Decompile:</span>
                <div className="seg">
                  {ids.map((id) => (
                    <button key={id} className={obj === id ? "on" : ""} onClick={() => setObj(id)}>
                      #{id} · {tracks.data!.objects[id].length} frames
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="well" style={{ padding: "14px 16px 8px" }}>
              <div className="row tiny muted" style={{ marginBottom: 8 }}>
                <span>Height of the tracked object over the video (from SAM 3.1)</span>
                <span className="grow" />
                {suggestion ? <span style={{ color: "var(--info-ink)" }}>Suggested window around the bounce at frame {suggestion.bounce}</span> : <span>No bounce found automatically: set the window by hand.</span>}
              </div>
              {plot()}
            </div>
            <div className="row wrap" style={{ alignItems: "flex-end" }}>
              <label className="field" style={{ width: 130 }}>
                <span>From frame</span>
                <input className="input" type="number" min={0} max={n - 1} value={start} onChange={(e) => setStart(Math.max(0, +e.target.value))} />
              </label>
              <label className="field" style={{ width: 130 }}>
                <span>To frame (excl.)</span>
                <input className="input" type="number" min={1} max={n} value={end} onChange={(e) => setEnd(Math.min(n, +e.target.value))} />
              </label>
              <span className="small muted" style={{ paddingBottom: 10 }}>{end - start} frames · one bounce works best</span>
              <span className="grow" />
              <button className="btn primary lg" onClick={submit} disabled={busy || running} data-testid="reconstruct">
                {running ? "Decompiling…" : "Decompile"}
              </button>
            </div>
            {err && <div className="callout bad small">{err}</div>}
          </div>
        )}
      </div>
      <details className="more">
        <summary>Advanced: physics model and camera</summary>
        <div className="inner row wrap">
          <label className="field">
            <span>Physics model</span>
            <select className="input" value={model} onChange={(e) => setModel(e.target.value as "projective" | "image_space")}>
              <option value="projective">Projective 3-D (perspective-aware)</option>
              <option value="image_space">Image-space 2-D (pixels)</option>
            </select>
          </label>
          {model === "projective" && (
            <label className="field" style={{ width: 200 }}>
              <span>Camera field of view prior (°)</span>
              <input className="input" type="number" min={20} max={140} value={hfov} onChange={(e) => setHfov(+e.target.value)} />
            </label>
          )}
        </div>
      </details>
      {run.reconstructions.length > 0 && (
        <details className="more">
          <summary>Previous attempts ({run.reconstructions.length})</summary>
          <div className="inner">
            <table className="data">
              <thead><tr><th>Id</th><th>Window</th><th>Model</th><th>Outcome</th><th></th></tr></thead>
              <tbody>
                {[...run.reconstructions].reverse().map((r) => (
                  <tr key={r.id} className={r.id === activeRecon ? "sel" : ""}>
                    <td>{r.id}</td>
                    <td>{r.params.start_frame}–{r.params.end_frame ?? "end"}</td>
                    <td>{r.params.physics_model}</td>
                    <td className="text">
                      {r.outcome === "scene_program" ? <Badge tone="good">decompiled</Badge> : r.outcome === "measurements_only" ? <Badge tone="warn">measured, no program</Badge> : r.status === "running" ? <Badge tone="info">running</Badge> : <span className="tiny" style={{ color: "var(--bad-ink)" }}>{r.error ?? r.status}</span>}
                    </td>
                    <td>{r.status === "completed" && <button className="btn sm" onClick={() => onOpenRecon(r.id)}>Open</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
