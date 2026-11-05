import type { ReactNode } from "react";
import type { Stage } from "./types";

export const fmt = {
  num: (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d)),
  bytes: (n: number | null | undefined) => {
    if (n === null || n === undefined) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(2)} MB`;
  },
  usd: (v: number | null | undefined) => (v === null || v === undefined ? "—" : `$${v < 0.01 ? v.toFixed(4) : v.toFixed(3)}`),
  time: (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : "—"),
  fps: (s: string | null | undefined) => {
    if (!s) return "—";
    const [a, b] = s.split("/").map(Number);
    return b ? (a / b).toFixed(a % b === 0 ? 0 : 3) : s;
  },
  vec: (v: number[] | null | undefined, d = 2) => (v ? `[${v.map((x) => x.toFixed(d)).join(", ")}]` : "—"),
};

export function Panel(props: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; bodyClass?: string; flush?: boolean }) {
  return (
    <section className={`panel ${props.className ?? ""}`}>
      {props.title !== undefined && (
        <header className="panel-head">
          <h3>{props.title}</h3>
          {props.sub && <span className="sub">{props.sub}</span>}
          <span className="grow" />
          {props.right}
        </header>
      )}
      {props.flush ? props.children : <div className={`panel-body ${props.bodyClass ?? ""}`}>{props.children}</div>}
    </section>
  );
}

export type Tone = "good" | "warn" | "bad" | "info" | "hypo" | "";

export function Badge({ tone = "", children, dot = true }: { tone?: Tone; children: ReactNode; dot?: boolean }) {
  return (
    <span className={`badge ${tone}`}>
      {dot && <span className="dot" />}
      {children}
    </span>
  );
}

/** Status glyphs: shape + colour, never colour alone. */
export function StatusIcon({ status, size = 18 }: { status: string; size?: number }) {
  const s = size;
  switch (status) {
    case "done":
    case "measured":
    case "completed":
      return (
        <svg className="icon" width={s} height={s} viewBox="0 0 18 18" aria-label={status}>
          <circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--status-good)" strokeWidth="1.5" />
          <path d="M5.5 9.2l2.3 2.3 4.6-4.8" fill="none" stroke="var(--status-good)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "running":
      return (
        <svg className="icon spin" width={s} height={s} viewBox="0 0 18 18" aria-label="running">
          <circle cx="9" cy="9" r="7" fill="none" stroke="var(--border-strong)" strokeWidth="1.6" />
          <path d="M9 2a7 7 0 0 1 7 7" fill="none" stroke="var(--ink-1)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "failed":
    case "missing":
    case "interrupted":
      return (
        <svg className="icon" width={s} height={s} viewBox="0 0 18 18" aria-label={status}>
          <circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--status-critical)" strokeWidth="1.5" />
          <path d="M6.3 6.3l5.4 5.4M11.7 6.3l-5.4 5.4" stroke="var(--status-critical)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "uncertain":
    case "warn":
      return (
        <svg className="icon" width={s} height={s} viewBox="0 0 18 18" aria-label={status}>
          <path d="M9 2l7.5 13h-15z" fill="none" stroke="var(--status-warn)" strokeWidth="1.5" strokeLinejoin="round" />
          <path d="M9 7v3.6M9 12.6v.1" stroke="var(--status-warn)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "skipped":
      return (
        <svg className="icon" width={s} height={s} viewBox="0 0 18 18" aria-label="skipped">
          <circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--ink-4)" strokeWidth="1.5" strokeDasharray="2 2" />
          <path d="M5.5 9h7" stroke="var(--ink-3)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg className="icon" width={s} height={s} viewBox="0 0 18 18" aria-label="pending">
          <circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--ink-4)" strokeWidth="1.5" />
        </svg>
      );
  }
}

function describeDetail(d: Record<string, unknown> | null): string {
  if (!d) return "";
  if (typeof d.reason === "string") return d.reason;
  if (Array.isArray(d.reasons)) return (d.reasons as string[]).join("; ");
  const parts: string[] = [];
  for (const [k, v] of Object.entries(d)) {
    if (v === null || v === undefined || typeof v === "object") continue;
    parts.push(`${k.replace(/_/g, " ")} ${v}`);
  }
  if (d.status_counts && typeof d.status_counts === "object") {
    const c = d.status_counts as Record<string, number>;
    parts.push(`measured ${c.measured} · uncertain ${c.uncertain} · missing ${c.missing}`);
  }
  if (d.usage && typeof d.usage === "object" && "video_frames_processed" in (d.usage as object))
    parts.push(`billed ${(d.usage as Record<string, number>).video_frames_processed} frames`);
  return parts.join(" · ");
}

export const TRACKING_ORDER = ["ingest", "timestamps", "sam_upload", "sam_segmentation", "sam_validation"];
export const RECON_ORDER = ["frames", "measurement", "trajectory", "physics", "scene_program", "simulation", "rendering", "views"];

export function StageList({ stages, order }: { stages: Record<string, Stage>; order?: string[] }) {
  const keys = order ?? Object.keys(stages);
  return (
    <ul className="stages">
      {keys
        .filter((k) => stages[k])
        .map((k) => {
          const s = stages[k];
          const d = s.detail as Record<string, any> | null;
          const prog =
            s.status === "running" && d && typeof d.done === "number" && typeof d.total === "number" ? d.done / d.total : null;
          const dur =
            s.started_utc && s.finished_utc ? `${((Date.parse(s.finished_utc) - Date.parse(s.started_utc)) / 1000).toFixed(1)} s` : "";
          return (
            <li key={k}>
              <StatusIcon status={s.status} />
              <div style={{ minWidth: 0 }}>
                <div className="title" style={{ color: s.status === "pending" ? "var(--ink-3)" : undefined }}>{s.title}</div>
                {d && <div className="detail">{describeDetail(d)}</div>}
                {prog !== null && (
                  <div className="progress" style={{ marginTop: 6 }} aria-label={`${Math.round(prog * 100)}%`}>
                    <div style={{ width: `${prog * 100}%` }} />
                  </div>
                )}
              </div>
              <span className="time">{s.status === "running" ? "running" : dur}</span>
            </li>
          );
        })}
    </ul>
  );
}

export function Readout({ label, value, unit, note }: { label: string; value: ReactNode; unit?: string; note?: ReactNode }) {
  return (
    <div className="readout">
      <span className="label">{label}</span>
      <span className="value">
        {value}
        {unit && <span className="unit">{unit}</span>}
      </span>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export const STATUS_COLOR: Record<string, string> = {
  measured: "var(--status-good)",
  uncertain: "var(--status-warn)",
  missing: "var(--status-critical)",
};

export function Logo() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="7" fill="#16191c" stroke="#2b3137" />
      <path d="M6 9 Q11 24 16 24 T26 12" fill="none" stroke="#199e70" strokeWidth="1.6" strokeDasharray="2 2.2" />
      <circle cx="16" cy="20.5" r="3.5" fill="none" stroke="#3987e5" strokeWidth="1.8" />
      <path d="M6 25h20" stroke="#e9e7e1" strokeWidth="1.6" />
    </svg>
  );
}
