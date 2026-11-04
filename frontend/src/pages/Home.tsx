import { useEffect, useRef, useState } from "react";
import { api, uploadVideo } from "../api";
import { navigate } from "../App";
import type { Health, RunListItem } from "../types";
import { Badge, Panel, fmt } from "../ui";

const STEPS = [
  ["Video", "real-world footage"],
  ["Perception", "Meta SAM 3.1 tracks the object"],
  ["Geometry", "outline centre, diameter, σ per frame"],
  ["Events", "contacts in the measured trajectory"],
  ["Physics", "projective scene program"],
  ["Simulation", "deterministic, executable"],
  ["Counterfactual", "change restitution, gravity"],
];

export const STATE_LABEL: Record<string, [string, "good" | "warn" | "bad" | "info" | ""]> = {
  ingesting: ["ingesting", "info"],
  rejected: ["rejected", "bad"],
  uploaded: ["uploaded", ""],
  planned: ["SAM plan ready", "info"],
  tracking: ["SAM running", "info"],
  tracking_failed: ["SAM failed", "bad"],
  tracked: ["tracked", "good"],
  reconstructing: ["reconstructing", "info"],
  reconstructed: ["reconstructed", "good"],
};

export function validateLocalFile(file: File, maxBytes: number | undefined): string | null {
  if (!/\.mp4$/i.test(file.name)) return `“${file.name}” is not an MP4 file. Decompile currently accepts MP4 video.`;
  if (maxBytes && file.size > maxBytes) return `The file is ${fmt.bytes(file.size)}; this server accepts up to ${fmt.bytes(maxBytes)}.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

export default function Home({ health }: { health: Health | null }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ w: number; h: number; d: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [provenance, setProvenance] = useState("real");
  const [progress, setProgress] = useState<number | null>(null);
  const [runs, setRuns] = useState<RunListItem[] | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.runs().then(setRuns).catch(() => setRuns([]));
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

  return (
    <div className="fade-in">
      <section className="hero">
        <div>
          <div className="eyebrow">Meta SAM 3.1 · geometric measurement · projective physics</div>
          <h1 style={{ marginTop: 14 }}>
            Turn reality into
            <br />
            executable physics. <em>Then change it.</em>
          </h1>
          <p className="lede">
            Decompile does not merely describe a video. It reconstructs an executable physical system from visual evidence — every
            parameter traceable to measured pixels and exact frame timestamps — and lets you run what-if experiments on it.
          </p>
          <div className="pipeline" aria-label="Decompile pipeline">
            {STEPS.map(([t, d], i) => (
              <div className="step" key={t}>
                <div className="n">{String(i + 1).padStart(2, "0")}</div>
                <div className="t">{t}</div>
                <div className="d">{d}</div>
              </div>
            ))}
          </div>
        </div>

        <div
          className={`dropzone ${over ? "over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            choose(e.dataTransfer.files?.[0]);
          }}
          data-testid="dropzone"
        >
          <input
            ref={input}
            type="file"
            accept="video/mp4,.mp4"
            hidden
            data-testid="file-input"
            onChange={(e) => choose(e.target.files?.[0] ?? undefined)}
          />
          {!file ? (
            <>
              <svg width="42" height="42" viewBox="0 0 42 42" aria-hidden>
                <rect x="1" y="7" width="40" height="28" rx="4" fill="none" stroke="#48515a" />
                <path d="M17 15l9 6-9 6z" fill="#7d7f80" />
              </svg>
              <div className="big">Drop a video of a bouncing object</div>
              <p className="muted small">
                MP4 (H.264 / H.265 / MPEG-4) · up to {health ? fmt.bytes(health.max_upload_bytes) : "…"} · static camera, one object, one bounce
              </p>
              <button className="btn primary" onClick={() => input.current?.click()}>
                Choose video
              </button>
            </>
          ) : (
            <>
              <video
                src={preview ?? undefined}
                controls
                muted
                playsInline
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  setMeta({ w: v.videoWidth, h: v.videoHeight, d: v.duration });
                }}
              />
              <div className="row wrap" style={{ justifyContent: "center" }}>
                <span className="mono small">{file.name}</span>
                <Badge>{fmt.bytes(file.size)}</Badge>
                {meta && <Badge>{`${meta.w}×${meta.h}`}</Badge>}
                {meta && <Badge>{`${meta.d.toFixed(2)} s`}</Badge>}
              </div>
              <p className="tiny muted">
                Frame count, exact timestamps and suitability are measured on the server after upload (FFmpeg decode). Uploading makes no
                SAM request.
              </p>
              <div className="row" style={{ justifyContent: "center" }}>
                <label className="row small ink2">
                  <input type="radio" name="prov" checked={provenance === "real"} onChange={() => setProvenance("real")} /> real footage
                </label>
                <label className="row small ink2">
                  <input type="radio" name="prov" checked={provenance === "synthetic"} onChange={() => setProvenance("synthetic")} />{" "}
                  synthetic / test video
                </label>
              </div>
              {progress !== null ? (
                <div style={{ width: "100%" }}>
                  <div className="progress">
                    <div style={{ width: `${progress * 100}%` }} />
                  </div>
                  <p className="tiny muted" style={{ marginTop: 6 }}>
                    {progress < 1 ? `Uploading… ${(progress * 100).toFixed(0)}%` : "Uploaded — validating on the server…"}
                  </p>
                </div>
              ) : (
                <div className="row">
                  <button className="btn ghost" onClick={() => { setFile(null); setPreview(null); setMeta(null); }}>
                    Clear
                  </button>
                  <button className="btn primary" onClick={upload} data-testid="upload">
                    Upload and analyse
                  </button>
                </div>
              )}
            </>
          )}
          {err && (
            <div className="callout bad" role="alert" style={{ width: "100%", textAlign: "left" }}>
              {err}
            </div>
          )}
        </div>
      </section>

      <Panel title="Projects" sub="stored on this server; reopen any run" flush>
        {runs === null ? (
          <div className="panel-body">
            <div className="skeleton" style={{ height: 40 }} />
          </div>
        ) : runs.length === 0 ? (
          <div className="empty">No runs yet. Upload a video to start.</div>
        ) : (
          <div className="runlist">
            {runs.map((r) => {
              const [label, tone] = STATE_LABEL[r.state] ?? [r.state, ""];
              return (
                <a className="item" key={r.id} href={`#/runs/${r.id}/process`}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.original_filename}</div>
                    <div className="tiny muted mono">
                      {r.id} · {r.frames ?? "?"} frames · {r.provenance}
                    </div>
                  </div>
                  <span className="tiny muted">{fmt.time(r.created_utc)}</span>
                  <div className="row">
                    {r.has_program && <Badge tone="good">scene program</Badge>}
                    <Badge tone={tone}>{label}</Badge>
                  </div>
                </a>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
