import { useEffect, useMemo, useRef, useState } from "react";
import { api, useRun } from "../api";
import { navigate } from "../App";
import type { Analysis, Health } from "../types";
import { Badge, Empty, fmt } from "../ui";
import { STATE_LABEL } from "./Home";
import ProcessTab from "./ProcessTab";
import ResultTab from "./ResultTab";
import ReconstructionTab from "./ReconstructionTab";
import PhysicsTab from "./PhysicsTab";
import SimulationTab from "./SimulationTab";
import ArtifactsTab from "./ArtifactsTab";

export type TabKey = "process" | "result" | "simulation" | "reconstruction" | "physics" | "artifacts";

const STEPS: { key: TabKey; label: string; needsRecon: boolean }[] = [
  { key: "process", label: "Track & decompile", needsRecon: false },
  { key: "result", label: "Decompiled", needsRecon: true },
  { key: "simulation", label: "What if…", needsRecon: true },
  { key: "reconstruction", label: "Evidence", needsRecon: true },
  { key: "physics", label: "Physics", needsRecon: true },
  { key: "artifacts", label: "Export", needsRecon: false },
];

export default function RunPage({ runId, tab, health }: { runId: string; tab: TabKey; health: Health | null }) {
  const { run, error, events, refresh } = useRun(runId);
  const completed = useMemo(() => (run?.reconstructions ?? []).filter((r) => r.status === "completed"), [run]);
  const storeKey = `decompile.recon.${runId}`;
  const [rid, setRid] = useState<string | null>(() => localStorage.getItem(storeKey));
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisErr, setAnalysisErr] = useState<string | null>(null);
  const prevJob = useRef<string | null>(null);

  useEffect(() => {
    if (!run) return;
    if (rid && completed.some((r) => r.id === rid)) return;
    const latest = completed[completed.length - 1];
    setRid(latest ? latest.id : null);
  }, [run, completed, rid]);
  useEffect(() => {
    if (rid) localStorage.setItem(storeKey, rid);
  }, [rid, storeKey]);

  // When a reconstruction finishes while the user watches, take them straight to the result.
  useEffect(() => {
    const j = run?.job ? `${run.job.kind}:${run.job.status}` : null;
    if (prevJob.current === "reconstruction:running" && j === "reconstruction:completed") {
      const latest = completed[completed.length - 1];
      if (latest) {
        setRid(latest.id);
        navigate(`/runs/${runId}/result`);
      }
    }
    prevJob.current = j;
  }, [run?.job, completed, runId]);

  const finishedAt = completed.find((r) => r.id === rid)?.finished_utc;
  useEffect(() => {
    setAnalysis(null);
    setAnalysisErr(null);
    if (!rid) return;
    api.recon(runId, rid).then((r) => setAnalysis(r.analysis)).catch((e) => setAnalysisErr((e as Error).message));
  }, [runId, rid, finishedAt]);

  if (error && !run)
    return (
      <div className="callout bad" role="alert" style={{ marginTop: 40 }}>
        {error === "run not found" ? "This project does not exist on the server." : error} <a href="#/">Back to projects</a>
      </div>
    );
  if (!run) return <div className="skeleton" style={{ height: 420, marginTop: 40 }} />;

  const [label, tone] = STATE_LABEL[run.state] ?? [run.state, ""];
  const go = (t: TabKey) => navigate(`/runs/${runId}/${t}`);
  const view = (() => {
    if (tab === "process")
      return <ProcessTab run={run} health={health} events={events} refresh={refresh} activeRecon={rid} onOpenRecon={(id) => { setRid(id); go("result"); }} />;
    if (tab === "artifacts") return <ArtifactsTab run={run} rid={rid} />;
    if (!rid)
      return <Empty>Nothing decompiled yet. Finish the steps on <a href={`#/runs/${runId}/process`} style={{ textDecoration: "underline" }}>Track &amp; decompile</a>.</Empty>;
    if (analysisErr) return <div className="callout bad">{analysisErr}</div>;
    if (!analysis) return <div className="skeleton" style={{ height: 480 }} />;
    if (tab === "result") return <ResultTab run={run} rid={rid} analysis={analysis} />;
    if (tab === "reconstruction") return <ReconstructionTab run={run} rid={rid} analysis={analysis} />;
    if (tab === "physics") return <PhysicsTab run={run} analysis={analysis} />;
    return <SimulationTab run={run} rid={rid} analysis={analysis} health={health} />;
  })();

  return (
    <div className="fade-in">
      <div className="runhead">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs">
            <a href="#/">Projects</a>
            <span>/</span>
            <span className="mono tiny">{run.id}</span>
          </div>
          <h2 className="h1" style={{ marginTop: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{run.original_filename}</h2>
          <div className="row wrap small muted" style={{ marginTop: 10 }}>
            <Badge tone={tone}>{label}</Badge>
            <span>{run.provenance === "real" ? "Real footage" : "Synthetic video"}</span>
            {run.video?.frame_count !== undefined && <span>· {run.video.frame_count} frames</span>}
            {run.video?.duration_s && <span>· {fmt.num(run.video.duration_s, 2)} s</span>}
          </div>
        </div>
        <span className="grow" />
        {completed.length > 1 && (
          <label className="field" style={{ minWidth: 240 }}>
            <span>Showing</span>
            <select className="input" value={rid ?? ""} onChange={(e) => setRid(e.target.value)}>
              {completed.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} · frames {r.params.start_frame}–{r.params.end_frame ?? "end"} · {r.outcome === "scene_program" ? "decompiled" : "no program"}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <nav className="stepper" role="tablist">
        {STEPS.map((s, i) => (
          <span key={s.key} style={{ display: "contents" }}>
            {i > 0 && <span className="sep" />}
            <button
              role="tab"
              aria-selected={tab === s.key}
              className={`${tab === s.key ? "active" : ""} ${s.key === "process" && rid ? "done" : ""}`}
              disabled={s.needsRecon && !rid}
              onClick={() => go(s.key)}
            >
              <span className="n">{s.key === "process" && rid ? "✓" : i + 1}</span>
              {s.label}
              {s.key === "simulation" && analysis && !analysis.program && <Badge tone="warn" dot={false}>no program</Badge>}
            </button>
          </span>
        ))}
      </nav>
      {view}
    </div>
  );
}
