import { useEffect, useMemo, useRef, useState } from "react";
import { api, fileUrl, frameUrl } from "../api";
import Chart from "../components/Chart";
import { validPart, type Analysis, type Counterfactual, type Run, type Sim } from "../types";
import { Badge, Empty, Panel, fmt } from "../ui";
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

export default function SimulationTab({ run, rid, analysis }: { run: Run; rid: string; analysis: Analysis }) {
  const pp = analysis.program_parameters;
  const fittedE: number = pp?.restitution ?? 0.5;
  const [e, setE] = useState<number>(Math.round(Math.max(0, Math.min(1, fittedE * 0.45)) * 100) / 100);
  const [g, setG] = useState(1);
  const [ext, setExt] = useState(0.6);
  const [cf, setCf] = useState<Counterfactual | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const reqId = useRef(0);

  useEffect(() => {
    if (!analysis.program) return;
    const my = ++reqId.current;
    setBusy(true);
    const h = window.setTimeout(async () => {
      const t0 = performance.now();
      try {
        const r = await api.counterfactual(run.id, rid, { restitution: e, gravity_scale: g === 1 ? null : g, extend_s: ext });
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
  }, [e, g, ext, run.id, rid, analysis.program]);

  if (!analysis.program)
    return (
      <Empty>
        <div className="callout bad" style={{ textAlign: "left", maxWidth: 720, margin: "0 auto" }}>
          <strong>No executable scene program for this reconstruction</strong>, so there is nothing to simulate. Reasons:{" "}
          {(analysis.physics_status?.reasons ?? []).join("; ")}
        </div>
      </Empty>
    );

  return (
    <div className="stack">
      <div className="callout hypo hatch">
        <strong>Hypothetical.</strong> Everything orange on this page is a what-if simulation of the fitted scene program with changed physical
        parameters, executed by Decompile's deterministic simulator. It is not observed and not validated. The observed video is shown on the
        left for comparison only.
      </div>

      <Panel
        title="Counterfactual controls"
        sub="each change re-runs the existing scene-program simulator on the server — no SAM call"
        right={
          <span className="tiny muted mono">
            {busy ? "simulating…" : latency !== null ? `simulated in ${latency.toFixed(0)} ms · ${cf?.id}` : ""}
          </span>
        }
      >
        <div className="slider-row">
          <div className="name">
            Restitution e<span className="muted">fitted {fittedE.toFixed(3)} · range 0–1</span>
          </div>
          <input type="range" min={0} max={1} step={0.01} value={e} onChange={(x) => setE(+x.target.value)} aria-label="restitution" />
          <div className="val">{e.toFixed(2)}</div>
        </div>
        <div className="slider-row">
          <div className="name">
            Gravity scale<span className="muted">× fitted gravity vector</span>
          </div>
          <input type="range" min={0.25} max={2.5} step={0.05} value={g} onChange={(x) => setG(+x.target.value)} aria-label="gravity scale" />
          <div className="val">×{g.toFixed(2)}</div>
        </div>
        <div className="slider-row" style={{ borderBottom: "none" }}>
          <div className="name">
            Extend horizon<span className="muted">beyond the last observation</span>
          </div>
          <input type="range" min={0} max={2} step={0.1} value={ext} onChange={(x) => setExt(+x.target.value)} aria-label="extend horizon" />
          <div className="val">+{ext.toFixed(1)} s</div>
        </div>
        <div className="row">
          <button className="btn sm" onClick={() => { setE(Math.round(fittedE * 100) / 100); setG(1); }}>
            Reset to fitted
          </button>
          <button className="btn sm" onClick={() => setE(0.3)}>e = 0.3</button>
          <button className="btn sm" onClick={() => setE(0.95)}>e = 0.95</button>
          <button className="btn sm" onClick={() => setG(0.5)}>half gravity</button>
          <span className="grow" />
          {err && <span className="small" style={{ color: "#ea7f7f" }}>{err}</span>}
        </div>
      </Panel>

      {cf && <Theatre run={run} rid={rid} analysis={analysis} cf={cf} />}

      {cf && (
        <div className="grid-2">
          <Compare cf={cf} />
          <Panel title="Outputs" sub="downloadable, labelled hypothetical">
            <div className="col">
              <a className="btn sm" href={fileUrl(run.id, `recon/${rid}/web/counterfactuals/${cf.id}.json`, true)} style={{ alignSelf: "flex-start" }}>
                Download counterfactual scene program + simulation (JSON)
              </a>
              <div className="row">
                <button
                  className="btn sm"
                  disabled={rendering}
                  onClick={async () => {
                    setRendering(true);
                    try {
                      const r = await api.renderCounterfactual(run.id, rid, cf.id);
                      setVideo(fileUrl(run.id, r.path));
                    } catch (x) {
                      setErr((x as Error).message);
                    } finally {
                      setRendering(false);
                    }
                  }}
                >
                  {rendering ? "Rendering…" : "Render counterfactual video"}
                </button>
                <span className="tiny muted">draws the simulator output over the dimmed source frames (server-side, FFmpeg)</span>
              </div>
              {video && (
                <div className="media hypo-frame">
                  <video src={video} controls autoPlay muted loop playsInline />
                  <span className="tag hypo">hypothetical</span>
                </div>
              )}
            </div>
          </Panel>
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
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.25);
  const W = analysis.report.video.displayed_size[0];
  const H = analysis.report.video.displayed_size[1];
  const frames = analysis.frames;
  const last = useRef<number | null>(null);

  useEffect(() => setT((x) => Math.min(Math.max(x, t0), t1)), [t0, t1]);
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
  const band = cf.observed_until_s < t1 ? [{ t0: cf.observed_until_s, t1, label: "no observation — simulation only" }] : [];
  const lines = (k: "x" | "y") => [
    { key: "fit", label: "fitted program", color: "var(--series-fitted)", t: F.t, v: F[k] },
    { key: "cf", label: "counterfactual", color: "var(--series-cf)", dash: true, t: C.t, v: C[k] },
  ];
  const marks = [
    ...F.events.map((ev) => ({ t: ev.t_s, label: "contact", color: "var(--series-fitted)", dash: true })),
    ...C.events.slice(0, 12).map((ev) => ({ t: ev.t_s, label: "cf contact", color: "var(--series-cf)", dash: true })),
  ];

  return (
    <>
      <div className="grid-2">
        <Panel title="Observed" sub="evidence: decoded source frame" right={<Badge tone="info">observed</Badge>}>
          <div className="media">
            <img src={frameUrl(run.id, rid, frame.frame_index)} alt={`observed frame ${frame.frame_index}`} style={{ opacity: observed ? 1 : 0.25 }} />
            <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
              {observed && frame.status === "measured" && (
                <circle cx={frame.x!} cy={frame.y!} r={(frame.diameter_px ?? medD) / 2} fill="none" stroke="var(--series-measured)" strokeWidth={1.5 * sw} />
              )}
            </svg>
            <span className="tag">{observed ? `frame ${frame.frame_index} · ${frame.status}` : "beyond the video"}</span>
            {!observed && (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }} className="small ink2">
                No observation after t = {cf.observed_until_s.toFixed(3)} s
              </div>
            )}
          </div>
        </Panel>
        <Panel title="Executable simulation" sub="scene program rendered by the simulator" right={<Badge tone="hypo">hypothetical in orange</Badge>}>
          <div className="media hypo-frame" style={{ background: "#07080a" }}>
            <img src={frameUrl(run.id, rid, frame.frame_index)} alt="" aria-hidden style={{ opacity: 0.18, filter: "grayscale(1)" }} />
            <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
              <polyline points={trail(F)} fill="none" stroke="var(--series-fitted)" strokeWidth={1.5 * sw} opacity={0.9} />
              <polyline points={trail(C)} fill="none" stroke="var(--series-cf)" strokeWidth={2 * sw} strokeDasharray={`${6 * sw} ${4 * sw}`} />
              {F.events.filter((ev) => ev.t_s <= t).map((ev, i) => (
                <circle key={`fe${i}`} cx={ev.x_px} cy={ev.y_px} r={3 * sw} fill="var(--series-fitted)" />
              ))}
              {C.events.filter((ev) => ev.t_s <= t).slice(0, 20).map((ev, i) => (
                <path key={`ce${i}`} d={`M${ev.x_px - 4 * sw},${ev.y_px - 4 * sw}l${8 * sw},${8 * sw}M${ev.x_px + 4 * sw},${ev.y_px - 4 * sw}l${-8 * sw},${8 * sw}`} stroke="var(--series-cf)" strokeWidth={1.5 * sw} />
              ))}
              {fp && <circle cx={fp.x} cy={fp.y} r={(fp.d ?? medD) / 2 + 3 * sw} fill="rgba(25,158,112,0.18)" stroke="var(--series-fitted)" strokeWidth={1.5 * sw} />}
              {cp && <circle cx={cp.x} cy={cp.y} r={(cp.d ?? medD) / 2} fill="rgba(217,89,38,0.2)" stroke="var(--series-cf)" strokeWidth={2 * sw} strokeDasharray={`${4 * sw} ${3 * sw}`} />}
            </svg>
            <span className="tag hypo">simulation · t = {t.toFixed(3)} s{cp ? "" : " · at rest"}</span>
          </div>
        </Panel>
      </div>
      <div className="panel" style={{ padding: "4px 16px 12px" }}>
        <div className="transport">
          <button className="btn sm" style={{ width: 64 }} onClick={() => (t >= t1 ? (setT(t0), setPlaying(true)) : setPlaying((p) => !p))}>
            {playing ? "Pause" : "Play"}
          </button>
          <input type="range" min={t0} max={t1} step={(t1 - t0) / 1000} value={t} onChange={(x) => { setPlaying(false); setT(+x.target.value); }} aria-label="simulation time" />
          <select className="input" style={{ height: 28, width: 80 }} value={speed} onChange={(x) => setSpeed(+x.target.value)} aria-label="playback speed">
            {[0.1, 0.25, 0.5, 1].map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
          <span className="timecode">t = {t.toFixed(3)} s</span>
        </div>
        <div className="legend" style={{ marginTop: 8 }}>
          <span className="k"><span className="mk" style={{ background: "var(--series-measured)" }} />measured</span>
          <span className="k"><span className="sw" style={{ background: "var(--series-fitted)" }} />fitted scene program (e = {fmt.num(cf.fitted_parameters.restitution, 3)})</span>
          <span className="k"><span className="sw dash" style={{ borderColor: "var(--series-cf)" }} />counterfactual (e = {fmt.num(cf.counterfactual_parameters.restitution, 2)}{cf.parameters.gravity_scale ? `, g ×${cf.parameters.gravity_scale}` : ""})</span>
        </div>
      </div>
      {notes.length > 0 && (
        <div className="callout warn small">
          <strong>At rest after t = {(C.valid_until_s ?? F.valid_until_s ?? 0).toFixed(3)} s.</strong> {notes[0]}
        </div>
      )}
      <Panel title="Trajectories" sub="measured vs fitted vs counterfactual">
        <div className="grid-2">
          <Chart title="y(t)" unit="px, oriented like the image" invert points={measured.map((f) => ({ t: f.t_s, v: f.y!, s: f.sigma_y, frame: f.frame_index }))}
            lines={lines("y")} gaps={gapsOf(frames)} marks={marks} bands={band} cursorT={t} />
          <Chart title="x(t)" unit="px" points={measured.map((f) => ({ t: f.t_s, v: f.x!, s: f.sigma_x, frame: f.frame_index }))}
            lines={lines("x")} gaps={gapsOf(frames)} marks={marks} bands={band} cursorT={t} />
        </div>
      </Panel>
    </>
  );
}

function Compare({ cf }: { cf: Counterfactual }) {
  const a = cf.fitted_parameters;
  const fd = validPart(cf.dense.fitted);
  const cd = validPart(cf.dense.counterfactual);
  const b = cf.counterfactual_parameters;
  const projective = a.model === "projective";
  const rows: [string, string, string][] = [
    ["Restitution e", fmt.num(a.restitution, 3), fmt.num(b.restitution, 3)],
    projective
      ? ["|g| (D/s²)", fmt.num(a.gravity_magnitude_diameters_s2, 2), fmt.num(b.gravity_magnitude_diameters_s2, 2)]
      : ["|a| (px/s²)", fmt.num(a.gravity_magnitude_px_s2, 2), fmt.num(b.gravity_magnitude_px_s2, 2)],
    projective
      ? ["g vector", fmt.vec(a.gravity_vector_diameters_s2, 1), fmt.vec(b.gravity_vector_diameters_s2, 1)]
      : ["acceleration", fmt.vec(a.acceleration_px_s2, 1), fmt.vec(b.acceleration_px_s2, 1)],
    ["Contacts in horizon", String(cf.dense.fitted.events.length), String(cf.dense.counterfactual.events.length)],
    ["First contact (s)", fmt.num(cf.dense.fitted.events[0]?.t_s, 4), fmt.num(cf.dense.counterfactual.events[0]?.t_s, 4)],
    ["Second contact (s)", fmt.num(cf.dense.fitted.events[1]?.t_s, 4), fmt.num(cf.dense.counterfactual.events[1]?.t_s, 4)],
    [
      "Last valid position (px)",
      `${fmt.num(fd.x.at(-1), 1)}, ${fmt.num(fd.y.at(-1), 1)}${fd.valid_until_s !== undefined ? " (rest)" : ""}`,
      `${fmt.num(cd.x.at(-1), 1)}, ${fmt.num(cd.y.at(-1), 1)}${cd.valid_until_s !== undefined ? " (rest)" : ""}`,
    ],
    ["Termination", cf.dense.fitted.termination ?? "—", cf.dense.counterfactual.termination ?? "—"],
  ];
  return (
    <Panel title="Fitted vs counterfactual" sub="same initial state; only the changed parameters differ" flush>
      <table className="data compare">
        <thead>
          <tr>
            <th>Quantity</th>
            <th>Fitted (from evidence)</th>
            <th style={{ color: "#f0a17f" }}>Counterfactual (hypothetical)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, x, y]) => (
            <tr key={k}>
              <td className="text">{k}</td>
              <td>{x}</td>
              <td className={x !== y ? "delta-up" : ""}>{y}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
