import { useEffect, useMemo, useRef, useState } from "react";
import { api, fileUrl, frameUrl, useAsync } from "../api";
import Chart from "../components/Chart";
import CodeView from "../components/CodeView";
import { validPart, type Analysis, type AskResult, type Counterfactual, type Health, type Run, type Sim } from "../types";
import { Badge, Empty, fmt } from "../ui";
import { gapsOf } from "./ReconstructionTab";

function at(sim: Sim, t: number): { x: number; y: number; d?: number } | null {
  const ts = sim.t;
  if (!ts.length || t < ts[0] - 1e-9 || t > ts[ts.length - 1] + 1e-9) return null;
  let lo = 0;
  let hi = ts.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (ts[m] <= t) lo = m;
    else hi = m;
  }
  const f = ts[hi] === ts[lo] ? 0 : (t - ts[lo]) / (ts[hi] - ts[lo]);
  const L = (a: number[]) => a[lo] + f * (a[hi] - a[lo]);
  return { x: L(sim.x), y: L(sim.y), d: sim.d ? L(sim.d) : undefined };
}

const SUGGESTIONS = ["What if this happened on the Moon?", "What if there was no floor?", "Throw it twice as hard", "Make it a superball", "Make it a lump of clay", "Add a strong wind"];

export default function SimulationTab({ run, rid, analysis, health }: { run: Run; rid: string; analysis: Analysis; health?: Health | null }) {
  const pp = analysis.program_parameters;
  const fittedE: number = pp?.restitution ?? 0.5;
  const [e, setE] = useState<number>(Math.round(Math.max(0, Math.min(1, fittedE * 0.45)) * 100) / 100);
  const [g, setG] = useState(1);
  const [ext, setExt] = useState(0.6);
  const [vs, setVs] = useState(1);
  const [floorOn, setFloorOn] = useState(true);
  const [cf, setCf] = useState<Counterfactual | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [q, setQ] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [askErr, setAskErr] = useState<string | null>(null);
  const reqId = useRef(0);
  const fittedSrc = useAsync(() => (analysis.program ? api.source(run.id, rid) : Promise.resolve(null)), [run.id, rid]);

  useEffect(() => {
    if (!analysis.program) return;
    const my = ++reqId.current;
    setBusy(true);
    const h = window.setTimeout(async () => {
      const t0 = performance.now();
      try {
        const r = await api.counterfactual(run.id, rid, { restitution: e, gravity_scale: g === 1 ? null : g, extend_s: ext, velocity_scale: vs === 1 ? null : vs, remove_floor: !floorOn });
        if (my !== reqId.current) return;
        setCf(r);
        setErr(null);
        setVideo(null);
        setLatency(performance.now() - t0);
      } catch (x) {
        if (my === reqId.current) setErr((x as Error).message);
      } finally {
        if (my === reqId.current) setBusy(false);
      }
    }, 220);
    return () => window.clearTimeout(h);
  }, [e, g, ext, vs, floorOn, run.id, rid, analysis.program]);

  const ask = async (question: string) => {
    if (!question.trim()) return;
    setQ(question);
    setAsking(true);
    setAskErr(null);
    setAnswer(null);
    try {
      const r = await api.ask(run.id, rid, question);
      setAnswer(r);
      const it = r.interpretation;
      if (it.supported) {
        setE(it.restitution ?? Math.round(fittedE * 1000) / 1000);
        setG(it.gravity_scale ?? 1);
        setVs(it.velocity_scale ?? 1);
        setFloorOn(!it.remove_floor);
        setExt(Math.min(2, Math.max(0.6, it.extend_s)));
      }
    } catch (x) {
      setAskErr((x as Error).message);
    } finally {
      setAsking(false);
    }
  };

  if (!analysis.program)
    return (
      <Empty>
        <div className="callout bad" style={{ textAlign: "left", maxWidth: 720, margin: "0 auto" }}>
          <strong>No executable scene program for this reconstruction</strong>, so there is nothing to simulate. Reasons:{" "}
          {(analysis.physics_status?.reasons ?? []).join("; ")}
        </div>
      </Empty>
    );

  const sparkOn = health?.spark_available !== false;
  return (
    <div className="stack fade-in">
      <div>
        <h2 className="h1">What if…?</h2>
        <p className="lede" style={{ marginTop: 10, maxWidth: 820 }}>
          Change the physics of the decompiled scene and watch what <em>would</em> have happened. Everything in orange is a simulation of the
          edited program, not footage.
        </p>
      </div>

      <div className="card card-pad">
        <form className="row" onSubmit={(x) => { x.preventDefault(); ask(q); }}>
          <input className="input" style={{ flex: 1, height: 52, fontSize: 16, paddingLeft: 18, borderRadius: 14 }} value={q}
            onChange={(x) => setQ(x.target.value)} placeholder="Ask in plain English, e.g. “what if this happened on the Moon?”" disabled={!sparkOn || asking}
            aria-label="what-if question" />
          <button className="btn primary lg" type="submit" disabled={!sparkOn || asking || !q.trim()}>{asking ? "Muse Spark is thinking…" : "Ask Muse Spark"}</button>
        </form>
        <div className="row wrap" style={{ marginTop: 12, gap: 8 }}>
          {SUGGESTIONS.map((s) => (
            <button key={s} className="btn sm" disabled={!sparkOn || asking} onClick={() => ask(s)}>{s}</button>
          ))}
        </div>
        {!sparkOn && <p className="tiny muted" style={{ marginTop: 10 }}>Muse Spark is not configured on this server; use the controls below.</p>}
        {asking && <div className="progress indeterminate" style={{ marginTop: 14 }}><div /></div>}
        {askErr && <div className="callout bad small" style={{ marginTop: 14 }}>{askErr}</div>}
        {answer && (
          <div className={`callout ${answer.interpretation.supported ? "hypo" : "warn"} fade-in`} style={{ marginTop: 14 }}>
            <div className="row wrap" style={{ gap: 10 }}>
              <strong style={{ fontSize: 15 }}>{answer.interpretation.supported ? answer.interpretation.headline : "Decompile can't simulate that yet"}</strong>
              <span className="grow" />
              <span className="tiny muted mono">{answer.interpretation.model} · {answer.interpretation.latency_s.toFixed(1)} s</span>
            </div>
            <div style={{ marginTop: 6 }}>{answer.interpretation.supported ? answer.interpretation.reasoning : answer.interpretation.unsupported_reason}</div>
            {answer.interpretation.supported && (
              <div className="row wrap tiny" style={{ marginTop: 8, gap: 14 }}>
                <span>restitution → <b className="mono">{answer.interpretation.restitution ?? "unchanged"}</b></span>
                <span>gravity → <b className="mono">{answer.interpretation.gravity_scale ? `× ${answer.interpretation.gravity_scale}` : "unchanged"}</b></span>
                {answer.interpretation.velocity_scale ? <span>throw speed → <b className="mono">× {answer.interpretation.velocity_scale}</b></span> : null}
                {answer.interpretation.remove_floor ? <span>floor → <b className="mono">removed</b></span> : null}
                {answer.interpretation.assumptions.map((a, i) => <span key={i} className="muted">· {a}</span>)}
              </div>
            )}
            <div className="tiny muted" style={{ marginTop: 8 }}>Muse Spark only chooses the parameters. The motion is computed by Decompile's deterministic simulator.</div>
          </div>
        )}
      </div>

      <div className="side">
        <div className="card card-pad sticky">
          <div className="row">
            <span className="h3">Physics controls</span>
            <span className="grow" />
            <span className="tiny muted mono">{busy ? "simulating…" : latency !== null ? `${latency.toFixed(0)} ms` : ""}</span>
          </div>
          <div className="ctl">
            <div className="top"><span className="name">Bounciness</span><span className="val">{e.toFixed(2)}</span></div>
            <input type="range" min={0} max={1} step={0.01} value={e} onChange={(x) => setE(+x.target.value)} aria-label="restitution" />
            <div className="hint"><span>clay 0</span><span>measured {fittedE.toFixed(2)}</span><span>1 perfect</span></div>
          </div>
          <div className="ctl">
            <div className="top"><span className="name">Gravity</span><span className="val">×{g.toFixed(2)}</span></div>
            <input type="range" min={0.1} max={2.6} step={0.005} value={g} onChange={(x) => setG(+x.target.value)} aria-label="gravity scale" />
            <div className="hint"><span>Moon 0.17</span><span>Earth 1</span><span>Jupiter 2.5</span></div>
          </div>
          <div className="ctl">
            <div className="top"><span className="name">Throw speed</span><span className="val">×{vs.toFixed(2)}</span></div>
            <input type="range" min={0.2} max={3} step={0.05} value={vs} onChange={(x) => setVs(+x.target.value)} aria-label="throw speed" />
            <div className="hint"><span>gentle</span><span>as filmed 1</span><span>3× harder</span></div>
          </div>
          <div className="ctl">
            <label className="row" style={{ justifyContent: "space-between", cursor: "pointer" }}>
              <span className="name">Floor</span>
              <span className="seg">
                <button type="button" className={floorOn ? "on" : ""} onClick={() => setFloorOn(true)}>On</button>
                <button type="button" className={!floorOn ? "on" : ""} onClick={() => setFloorOn(false)}>Removed</button>
              </span>
            </label>
          </div>
          <div className="ctl">
            <div className="top"><span className="name">Keep simulating</span><span className="val">+{ext.toFixed(1)} s</span></div>
            <input type="range" min={0} max={2} step={0.1} value={ext} onChange={(x) => setExt(+x.target.value)} aria-label="extend horizon" />
            <div className="hint"><span>after the video ends</span></div>
          </div>
          <div className="row wrap" style={{ marginTop: 6 }}>
            <button className="btn sm" onClick={() => { setE(Math.round(fittedE * 1000) / 1000); setG(1); setVs(1); setFloorOn(true); setAnswer(null); }}>Reset to measured</button>
          </div>
          {err && <div className="callout bad small" style={{ marginTop: 12 }}>{err}</div>}
        </div>
        <div className="col" style={{ gap: 20 }}>
          {cf ? <Theatre run={run} rid={rid} analysis={analysis} cf={cf} /> : <div className="skeleton" style={{ height: 420 }} />}
        </div>
      </div>

      {cf && (
        <div className="grid-2" style={{ alignItems: "start" }}>
          {fittedSrc.data && cf.source ? (
            <div className="col" style={{ gap: 10 }}>
              <div className="row small"><span className="h3">The edit, in code</span><span className="muted">changed lines are marked ~</span></div>
              <CodeView code={cf.source.display} exact={cf.source.exact} compare={fittedSrc.data.display} filename={`counterfactual_${rid}.py`} maxHeight={520} />
            </div>
          ) : <div />}
          <div className="col" style={{ gap: 20 }}>
            <Compare cf={cf} />
            <div className="card card-pad">
              <div className="row wrap">
                <a className="btn sm" href={fileUrl(run.id, `recon/${rid}/web/counterfactuals/${cf.id}.json`, true)}>Download simulation (JSON)</a>
                <button className="btn sm" disabled={rendering} onClick={async () => {
                  setRendering(true);
                  try {
                    const r = await api.renderCounterfactual(run.id, rid, cf.id);
                    setVideo(fileUrl(run.id, r.path));
                  } catch (x) {
                    setErr((x as Error).message);
                  } finally {
                    setRendering(false);
                  }
                }}>{rendering ? "Rendering…" : "Render as video"}</button>
              </div>
              {video && (
                <div className="media hypo-frame" style={{ marginTop: 14 }}>
                  <video src={video} controls autoPlay muted loop playsInline />
                  <span className="tag hypo">hypothetical</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Theatre({ run, rid, analysis, cf }: { run: Run; rid: string; analysis: Analysis; cf: Counterfactual }) {
  const F = validPart(cf.dense.fitted);
  const C = validPart(cf.dense.counterfactual);
  const t0 = cf.dense.fitted.t[0];
  const t1 = cf.dense.fitted.t[cf.dense.fitted.t.length - 1];
  const notes = [cf.dense.fitted.validity_note, cf.dense.counterfactual.validity_note].filter(Boolean);
  const [t, setT] = useState(t0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0.5);
  const W = analysis.report.video.displayed_size[0];
  const H = analysis.report.video.displayed_size[1];
  const frames = analysis.frames;
  const last = useRef<number | null>(null);

  useEffect(() => { setT(t0); setPlaying(true); }, [cf.id, t0]);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = (now: number) => {
      const dt = last.current === null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setT((x) => {
        const n = x + dt * speed;
        if (n >= t1) {
          setPlaying(false);
          return t1;
        }
        return n;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      last.current = null;
    };
  }, [playing, speed, t1]);

  const observed = t <= cf.observed_until_s + 1e-6;
  const frame = useMemo(() => {
    let best = frames[0];
    for (const f of frames) if (f.t_s <= t + 1e-6) best = f;
    return best;
  }, [frames, t]);
  const fp = at(F, t);
  const cp = at(C, t);
  const medD = analysis.report.measurement.median_diameter_px ?? 20;
  const trail = (s: Sim) => s.t.map((tt, i) => (tt <= t ? `${s.x[i]},${s.y[i]}` : null)).filter(Boolean).join(" ");
  const sw = Math.max(1, W / 560);
  const measured = frames.filter((f) => f.status === "measured");
  const band = cf.observed_until_s < t1 ? [{ t0: cf.observed_until_s, t1, label: "after the video ends" }] : [];
  const marks = [
    ...F.events.map((ev) => ({ t: ev.t_s, label: "contact", color: "var(--series-fitted)", dash: true })),
    ...C.events.slice(0, 12).map((ev) => ({ t: ev.t_s, label: "cf contact", color: "var(--series-cf)", dash: true })),
  ];

  return (
    <>
      <div className="card" style={{ padding: 14 }}>
        <div className="media hypo-frame" style={{ background: "#050607" }}>
          <img src={frameUrl(run.id, rid, frame.frame_index)} alt="" aria-hidden style={{ opacity: 0.32, filter: "grayscale(0.6)" }} />
          <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            <polyline points={trail(F)} fill="none" stroke="var(--series-fitted)" strokeWidth={1.6 * sw} opacity={0.9} />
            <polyline points={trail(C)} fill="none" stroke="var(--series-cf)" strokeWidth={2.2 * sw} strokeDasharray={`${6 * sw} ${4 * sw}`} />
            {fp && <circle cx={fp.x} cy={fp.y} r={(fp.d ?? medD) / 2 + 3 * sw} fill="rgba(25,158,112,0.15)" stroke="var(--series-fitted)" strokeWidth={1.6 * sw} />}
            {cp && <circle cx={cp.x} cy={cp.y} r={(cp.d ?? medD) / 2} fill="rgba(217,89,38,0.28)" stroke="var(--series-cf)" strokeWidth={2.2 * sw} />}
          </svg>
          <span className="tag hypo">simulation · t = {t.toFixed(2)} s{cp ? "" : " · at rest"}</span>
          <div className="pip">
            <div className="media" style={{ borderRadius: 0 }}>
              <img src={frameUrl(run.id, rid, frame.frame_index)} alt={`observed frame ${frame.frame_index}`} style={{ opacity: observed ? 1 : 0.2 }} />
              <span className="tag" style={{ top: 6, left: 6, fontSize: 9.5, padding: "3px 6px" }}>{observed ? "real footage" : "video ended"}</span>
            </div>
          </div>
        </div>
        <div className="transport">
          <button className="btn sm" style={{ width: 70 }} onClick={() => (t >= t1 ? (setT(t0), setPlaying(true)) : setPlaying((p) => !p))}>
            {playing ? "Pause" : t >= t1 ? "Replay" : "Play"}
          </button>
          <input type="range" min={t0} max={t1} step={(t1 - t0) / 1000} value={t} onChange={(x) => { setPlaying(false); setT(+x.target.value); }} aria-label="simulation time" />
          <select className="input" style={{ height: 30, width: 76 }} value={speed} onChange={(x) => setSpeed(+x.target.value)} aria-label="playback speed">
            {[0.1, 0.25, 0.5, 1].map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </div>
        <div className="legend" style={{ marginTop: 10 }}>
          <span className="k"><span className="sw" style={{ background: "var(--series-fitted)" }} />as filmed (decompiled, e = {fmt.num(cf.fitted_parameters.restitution, 2)})</span>
          <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-cf)" }} />what if (e = {fmt.num(cf.counterfactual_parameters.restitution, 2)}{cf.parameters.gravity_scale ? `, gravity ×${cf.parameters.gravity_scale}` : ""}{cf.parameters.velocity_scale ? `, thrown ×${cf.parameters.velocity_scale}` : ""}{cf.parameters.remove_floor ? ", no floor" : ""})</span>
          <Badge tone="hypo">hypothetical</Badge>
        </div>
      </div>
      {notes.length > 0 && (
        <div className="callout warn small">
          <strong>Comes to rest at t = {(C.valid_until_s ?? F.valid_until_s ?? 0).toFixed(2)} s.</strong> {notes[0]}
        </div>
      )}
      <div className="card card-pad">
        <Chart title="Height over time" unit="y in px, oriented like the image" invert height={200}
          points={measured.map((f) => ({ t: f.t_s, v: f.y!, s: f.sigma_y, frame: f.frame_index }))}
          lines={[
            { key: "fit", label: "as filmed", color: "var(--series-fitted)", t: F.t, v: F.y },
            { key: "cf", label: "what if", color: "var(--series-cf)", dash: true, t: C.t, v: C.y },
          ]}
          gaps={gapsOf(frames)} marks={marks} bands={band} cursorT={t} />
      </div>
    </>
  );
}

function Compare({ cf }: { cf: Counterfactual }) {
  const a = cf.fitted_parameters;
  const b = cf.counterfactual_parameters;
  const projective = a.model === "projective";
  const fd = validPart(cf.dense.fitted);
  const cd = validPart(cf.dense.counterfactual);
  const rows: [string, string, string][] = [
    ["Restitution e", fmt.num(a.restitution, 3), fmt.num(b.restitution, 3)],
    projective
      ? ["|g| (D/s²)", fmt.num(a.gravity_magnitude_diameters_s2, 2), fmt.num(b.gravity_magnitude_diameters_s2, 2)]
      : ["|a| (px/s²)", fmt.num(a.gravity_magnitude_px_s2, 2), fmt.num(b.gravity_magnitude_px_s2, 2)],
    ["Bounces in view", String(cf.dense.fitted.events.length), String(cf.dense.counterfactual.events.length)],
    ["First bounce (s)", fmt.num(cf.dense.fitted.events[0]?.t_s, 3), fmt.num(cf.dense.counterfactual.events[0]?.t_s, 3)],
    ["Second bounce (s)", fmt.num(cf.dense.fitted.events[1]?.t_s, 3), fmt.num(cf.dense.counterfactual.events[1]?.t_s, 3)],
    [
      "Last valid position (px)",
      `${fmt.num(fd.x.at(-1), 1)}, ${fmt.num(fd.y.at(-1), 1)}${fd.valid_until_s !== undefined ? " (rest)" : ""}`,
      `${fmt.num(cd.x.at(-1), 1)}, ${fmt.num(cd.y.at(-1), 1)}${cd.valid_until_s !== undefined ? " (rest)" : ""}`,
    ],
  ];
  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <div className="card-head"><span className="t">Fitted vs counterfactual</span><span className="s">same start; only the edited parameters differ</span></div>
      <div className="card-body" style={{ padding: "8px 8px 12px" }}>
        <table className="data compare">
          <thead>
            <tr><th></th><th>As filmed</th><th style={{ color: "var(--hypo-ink)" }}>What if (hypothetical)</th></tr>
          </thead>
          <tbody>
            {rows.map(([k, x, y]) => (
              <tr key={k}><td className="text">{k}</td><td>{x}</td><td className={x !== y ? "delta-up" : ""}>{y}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
