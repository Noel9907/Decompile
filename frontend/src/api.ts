import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AskResult, FitReview, Analysis, Artifact, Counterfactual, Health, Perception, Plan, ReconParams, Run, RunEvent, RunListItem,
} from "./types";

export class ApiError extends Error {
  status: number;
  details: string[];
  constructor(status: number, message: string, details: string[] = []) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  const text = await r.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!r.ok) {
    const d = body?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => x.msg ?? String(x)).join("; ") : `HTTP ${r.status}`;
    throw new ApiError(r.status, msg, body?.details ?? []);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  runs: () => req<RunListItem[]>("/api/runs"),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  frameTable: (id: string) => req<{ frame_count: number; width: number; height: number; t_s: number[] }>(`/api/runs/${id}/frame-table`),
  plan: (id: string, prompt: Record<string, unknown>) => req<Plan>(`/api/runs/${id}/plan`, json({ prompt })),
  confirmSam: (id: string, plan: Plan) =>
    req<Run>(`/api/runs/${id}/sam/confirm`, json({ confirm: true, plan_sha256: plan.sha256, acknowledged_cost_usd: plan.cost.expected_cost_usd })),
  importSam: (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<Run>(`/api/runs/${id}/sam/import`, { method: "POST", body: fd });
  },
  tracks: (id: string) =>
    req<{ objects: Record<string, { f: number; box: number[]; conf: number | null; mask: boolean }[]>; summary: Record<string, any>; frame_count: number; width: number; height: number }>(
      `/api/runs/${id}/sam/tracks`),
  reconstruct: (id: string, p: Partial<ReconParams>) => req<{ recon_id: string; run: Run }>(`/api/runs/${id}/reconstructions`, json(p)),
  recon: (id: string, rid: string) => req<{ reconstruction: any; analysis: Analysis | null }>(`/api/runs/${id}/reconstructions/${rid}`),
  perception: (id: string, rid: string) => req<Perception>(`/api/runs/${id}/reconstructions/${rid}/perception`),
  counterfactual: (id: string, rid: string, p: { restitution: number | null; gravity_scale: number | null; extend_s: number; velocity_scale?: number | null; remove_floor?: boolean }) =>
    req<Counterfactual>(`/api/runs/${id}/reconstructions/${rid}/counterfactuals`, json(p)),
  source: (id: string, rid: string) => req<{ display: string; exact: string; filename: string }>(`/api/runs/${id}/reconstructions/${rid}/source`),
  predict: (id: string, rid: string) => req<unknown>(`/api/runs/${id}/reconstructions/${rid}/predict`, { method: "POST" }),
  explain: (id: string, rid: string) => req<FitReview>(`/api/runs/${id}/reconstructions/${rid}/explain`, { method: "POST" }),
  ask: (id: string, rid: string, question: string) => req<AskResult>(`/api/runs/${id}/reconstructions/${rid}/ask`, json({ question })),
  renderCounterfactual: (id: string, rid: string, cf: string) =>
    req<{ path: string }>(`/api/runs/${id}/reconstructions/${rid}/counterfactuals/${cf}/video`, { method: "POST" }),
  artifacts: (id: string) => req<Artifact[]>(`/api/runs/${id}/artifacts`),
  report: (id: string, rid?: string) => req<Record<string, any>>(`/api/runs/${id}/report${rid ? `?recon_id=${rid}` : ""}`),
  text: async (url: string) => {
    const r = await fetch(url);
    if (!r.ok) throw new ApiError(r.status, `HTTP ${r.status}`);
    return r.text();
  },
};

export const fileUrl = (runId: string, path: string, download = false) =>
  `/api/runs/${runId}/files/${path.split("/").map(encodeURIComponent).join("/")}${download ? "?download=1" : ""}`;
export const frameUrl = (runId: string, rid: string, index: number) => `/api/runs/${runId}/reconstructions/${rid}/frames/${index}`;
export const previewUrl = (runId: string, index: number) => `/api/runs/${runId}/preview/${index}`;
export const videoUrl = (runId: string) => `/api/runs/${runId}/video`;

/** Upload with real byte-level progress (XMLHttpRequest exposes upload progress; fetch does not). */
export function uploadVideo(file: File, provenance: string, onProgress: (sent: number, total: number) => void): Promise<Run> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/runs");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded, e.total);
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as Run);
      else reject(new ApiError(xhr.status, body?.detail ?? `Upload failed (HTTP ${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    const fd = new FormData();
    fd.append("file", file);
    fd.append("provenance", provenance);
    xhr.send(fd);
  });
}

/** Run state kept in sync with the server: initial fetch, then refetch on every server-sent event.
 *  The server's run.json is the source of truth, so a page refresh restores everything. */
export function useRun(runId: string) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setRun(await api.run(runId));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [runId]);

  useEffect(() => {
    setRun(null);
    setEvents([]);
    refresh();
    // Fallback: plain polling (no SSE) when EventSource is unavailable or ?poll is in the URL
    // (e.g. behind proxies that buffer event streams).
    if (typeof EventSource === "undefined" || new URLSearchParams(window.location.search).has("poll")) {
      const h = window.setInterval(refresh, 1500);
      return () => window.clearInterval(h);
    }
    const es = new EventSource(`/api/runs/${runId}/events`);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (m) => {
      try {
        const ev = JSON.parse(m.data) as RunEvent;
        setEvents((prev) => (prev.length && prev[prev.length - 1].seq >= ev.seq ? prev : [...prev.slice(-400), ev]));
      } catch {
        return;
      }
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(refresh, 120);
    };
    return () => {
      es.close();
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [runId, refresh]);

  return { run, error, events, connected, refresh, setRun };
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    fn()
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e) => alive && setError((e as Error).message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading, setData };
}
