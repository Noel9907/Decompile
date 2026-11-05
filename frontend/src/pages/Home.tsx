import { useEffect, useRef, useState } from "react";
import { api, previewUrl, uploadVideo } from "../api";
import { navigate } from "../App";
import HeroVisual, { type Featured } from "../components/HeroVisual";
import Scoreboard from "../components/Scoreboard";
import type { Health, RunListItem } from "../types";
import { Badge, fmt } from "../ui";

export const STATE_LABEL: Record<string, [string, "good" | "warn" | "bad" | "info" | ""]> = {
  ingesting: ["checking video", "info"],
  rejected: ["video rejected", "bad"],
  uploaded: ["ready to track", ""],
  planned: ["ready to track", ""],
  tracking: ["tracking…", "info"],
  tracking_failed: ["tracking failed", "bad"],
  tracked: ["ready to decompile", "info"],
  reconstructing: ["decompiling…", "info"],
  reconstructed: ["decompiled", "good"],
};

export function validateLocalFile(file: File, maxBytes: number | undefined): string | null {
  if (!/\.mp4$/i.test(file.name)) return `“${file.name}” is not an MP4 file. Decompile currently accepts MP4 video.`;
  if (maxBytes && file.size > maxBytes) return `The file is ${fmt.bytes(file.size)}; this server accepts up to ${fmt.bytes(maxBytes)}.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

const HOW = [
  {
    k: "01", t: "Perceive", d: "Meta SAM 3.1 segments and tracks the object through every frame of your video.",
    icon: <svg width="40" height="40" viewBox="0 0 40 40"><rect x="4" y="8" width="32" height="24" rx="4" fill="none" stroke="#55585e" /><circle cx="22" cy="18" r="5" fill="rgba(57,135,229,0.25)" stroke="#3987e5" strokeWidth="1.5" /><rect x="15.5" y="11.5" width="13" height="13" fill="none" stroke="var(--ink-1)" strokeDasharray="2 2" /></svg>,
  },
  {
    k: "02", t: "Measure", d: "Sub-pixel outline fits give each frame's centre, apparent size and uncertainty. Bad frames are flagged, never invented.",
    icon: <svg width="40" height="40" viewBox="0 0 40 40">{[6, 12, 18, 24, 30].map((x, i) => <g key={x}><line x1={x} x2={x} y1={30 - [14, 20, 22, 18, 10][i] - 3} y2={30 - [14, 20, 22, 18, 10][i] + 3} stroke="#3987e5" /><circle cx={x} cy={30 - [14, 20, 22, 18, 10][i]} r="2.4" fill="#3987e5" /></g>)}<line x1="3" x2="37" y1="33" y2="33" stroke="#55585e" /></svg>,
  },
  {
    k: "03", t: "Decompile", d: "A perspective-aware physics fit recovers gravity, the bounce and its restitution, as an executable program.",
    icon: <svg width="40" height="40" viewBox="0 0 40 40"><path d="M5 30 Q12 2 20 30 Q26 14 33 30" fill="none" stroke="#199e70" strokeWidth="1.8" /><text x="3" y="12" fill="#85868a" fontSize="8" fontFamily="monospace">{"e=0.68"}</text></svg>,
  },
  {
    k: "04", t: "Rewrite reality", d: "Ask “what if this were on the Moon?” Muse Spark edits the program; the simulator plays out what would have happened.",
    icon: <svg width="40" height="40" viewBox="0 0 40 40"><path d="M5 30 Q12 2 20 30" fill="none" stroke="#199e70" strokeWidth="1.6" /><path d="M20 30 Q30 -4 38 24" fill="none" stroke="#d95926" strokeWidth="1.8" strokeDasharray="3 2.5" /></svg>,
  },
];

export default function Home({ health }: { health: Health | null }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ w: number; h: number; d: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [provenance, setProvenance] = useState("real");
  const [progress, setProgress] = useState<number | null>(null);
  const [runs, setRuns] = useState<RunListItem[] | null>(null);
  const [featured, setFeatured] = useState<Featured | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.runs().then(async (rs) => {
      setRuns(rs);
      const best = rs.find((r) => r.has_program);
      if (!best) return;
      const run = await api.run(best.id);
      const rec = [...run.reconstructions].reverse().find((r) => r.outcome === "scene_program");
      if (!rec) return;
      const { analysis } = await api.recon(best.id, rec.id);
      if (analysis) setFeatured({ runId: best.id, rid: rec.id, analysis, filename: run.original_filename });
    }).catch(() => setRuns([]));
  }, []);
  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview]);

  const choose = (f: File | undefined) => {
    if (!f) return;
    const e = validateLocalFile(f, health?.max_upload_bytes);
    setErr(e);
    setMeta(null);
    if (e) {
      setFile(null);
      setPreview(null);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const upload = async () => {
    if (!file) return;
    setErr(null);
    setProgress(0);
    try {
      const run = await uploadVideo(file, provenance, (s, t) => setProgress(s / t));
      navigate(`/runs/${run.id}/process`);
    } catch (e) {
      setErr((e as Error).message);
      setProgress(null);
    }
  };

  const rc = featured?.analysis.report.reconstruction_consistency;
  return (
    <div className="fade-in">
      <section className="hero">
        <div>
          <div className="eyebrow">Inverse graphics with Meta SAM 3.1 + Muse Spark</div>
          <h1 className="display" style={{ marginTop: 22 }}>
            Decompile
            <br />
            reality<span className="dim">.</span>
          </h1>
          <p className="lede" style={{ marginTop: 26 }}>
            Give Decompile a video of something moving. It returns the physics that produced it, as source code you can run, read and
            rewrite.
          </p>
          <div className="cta">
            {featured ? (
              <button className="btn primary lg" onClick={() => navigate(`/runs/${featured.runId}/result`)}>
                See a decompiled video →
              </button>
            ) : null}
            <button className={`btn lg ${featured ? "ghost" : "primary"}`} style={{ border: "1px solid var(--line-strong)" }} onClick={() => document.getElementById("upload")?.scrollIntoView()}>
              Decompile your own video
            </button>
          </div>
          {featured && (
            <div className="proof">
              <div><b>{featured.analysis.program_parameters?.restitution?.toFixed(2)}</b>restitution, from pixels</div>
              <div><b>{rc ? `${rc.rms_px.toFixed(2)} px` : "—"}</b>fit error</div>
              <div><b>{featured.analysis.report.measurement.status_counts.measured}/{featured.analysis.frames.length}</b>frames measured</div>
            </div>
          )}
        </div>
        <HeroVisual featured={featured} />
      </section>

      <Scoreboard />

      <section className="block" id="how">
        <div className="eyebrow" style={{ marginBottom: 18 }}>How it works</div>
        <div className="steps4">
          {HOW.map((s) => (
            <div key={s.k}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                {s.icon}
                <span className="k">{s.k}</span>
              </div>
              <h4>{s.t}</h4>
              <p>{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="block" id="upload">
        <div className="split" style={{ alignItems: "center" }}>
          <div>
            <h2 className="h1">Decompile your own video</h2>
            <p className="lede" style={{ marginTop: 14 }}>Works today on one object, one bounce, filmed from a still camera.</p>
            <ul className="ink2" style={{ marginTop: 22, paddingLeft: 18, display: "grid", gap: 8 }}>
              <li>Hold the phone still (lean it on something).</li>
              <li>Drop or throw a ball so it bounces once on the floor, fully in view.</li>
              <li>Keep the clip short: 2–5 seconds, MP4.</li>
            </ul>
            <p className="small muted" style={{ marginTop: 18 }}>
              Uploading is free. Tracking with SAM 3.1 is a paid API call: you will see the exact cost and confirm it first.
            </p>
          </div>
          <div
            className={`dropzone ${over ? "over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); choose(e.dataTransfer.files?.[0]); }}
            data-testid="dropzone"
          >
            <input ref={input} type="file" accept="video/mp4,.mp4" hidden data-testid="file-input" onChange={(e) => choose(e.target.files?.[0] ?? undefined)} />
            {!file ? (
              <>
                <svg width="48" height="48" viewBox="0 0 48 48" aria-hidden><circle cx="24" cy="24" r="23" fill="none" stroke="var(--line-strong)" /><path d="M24 31V17M17.5 23.5 24 17l6.5 6.5" fill="none" stroke="var(--ink-1)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                <div className="h3">Drop an MP4 here</div>
                <button className="btn primary" onClick={() => input.current?.click()}>Choose video</button>
                <span className="tiny muted">up to {health ? fmt.bytes(health.max_upload_bytes) : "…"}</span>
              </>
            ) : (
              <>
                <video src={preview ?? undefined} controls muted playsInline autoPlay loop
                  onLoadedMetadata={(e) => setMeta({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight, d: e.currentTarget.duration })} />
                <div className="row wrap" style={{ justifyContent: "center" }}>
                  <span className="small">{file.name}</span>
                  <span className="tiny muted">{fmt.bytes(file.size)}{meta ? ` · ${meta.w}×${meta.h} · ${meta.d.toFixed(1)} s` : ""}</span>
                </div>
                <div className="seg">
                  <button className={provenance === "real" ? "on" : ""} onClick={() => setProvenance("real")}>Real footage</button>
                  <button className={provenance === "synthetic" ? "on" : ""} onClick={() => setProvenance("synthetic")}>Synthetic / test</button>
                </div>
                {progress !== null ? (
                  <div style={{ width: "100%" }}>
                    <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>
                    <p className="tiny muted" style={{ marginTop: 8 }}>{progress < 1 ? `Uploading… ${(progress * 100).toFixed(0)}%` : "Uploaded. Opening your project…"}</p>
                  </div>
                ) : (
                  <div className="row">
                    <button className="btn ghost" onClick={() => { setFile(null); setPreview(null); setMeta(null); }}>Choose another</button>
                    <button className="btn primary" onClick={upload} data-testid="upload">Upload and continue →</button>
                  </div>
                )}
              </>
            )}
            {err && <div className="callout bad" role="alert" style={{ width: "100%", textAlign: "left" }}>{err}</div>}
          </div>
        </div>
      </section>

      {runs && runs.length > 0 && (
        <section className="block" id="projects">
          <div className="row" style={{ marginBottom: 18 }}>
            <div className="eyebrow">Your projects</div>
          </div>
          <div className="projects">
            {runs.map((r) => {
              const [label, tone] = STATE_LABEL[r.state] ?? [r.state, ""];
              const tab = r.has_program ? "result" : "process";
              return (
                <a className="project" key={r.id} href={`#/runs/${r.id}/${tab}`}>
                  <div className="thumb" style={{ backgroundImage: r.state === "rejected" || r.state === "ingesting" ? undefined : `url(${previewUrl(r.id, 0)})` }} />
                  <div className="meta">
                    <div className="row">
                      <span className="grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: 550 }}>{r.original_filename}</span>
                      <Badge tone={tone}>{label}</Badge>
                    </div>
                    <div className="tiny muted" style={{ marginTop: 4 }}>{r.frames ?? "?"} frames · {fmt.time(r.created_utc)}</div>
                  </div>
                </a>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
