import { useEffect, useMemo, useState } from "react";
import { api, useRun } from "../api";
import { navigate } from "../App";
import type { Analysis, Health } from "../types";
import { Badge, Empty, fmt } from "../ui";
import { STATE_LABEL } from "./Home";
import ProcessTab from "./ProcessTab";
import ReconstructionTab from "./ReconstructionTab";
import PhysicsTab from "./PhysicsTab";
import SimulationTab from "./SimulationTab";
import ArtifactsTab from "./ArtifactsTab";

export type TabKey = "process" | "reconstruction" | "physics" | "simulation" | "artifacts";

const TABS: { key: TabKey; label: string }[] = [
  { key: "process", label: "Processing" },
  { key: "reconstruction", label: "Reconstruction" },
  { key: "physics", label: "Physics" },
  { key: "simulation", label: "Simulation & counterfactuals" },
  { key: "artifacts", label: "Artifacts" },
];

export default function RunPage({ runId, tab, health }: { runId: string; tab: TabKey; health: Health | null }) {
  const { run, error, events, refresh } = useRun(runId);
  const completed = useMemo(() => (run?.reconstructions ?? []).filter((r) => r.status === "completed"), [run]);
  const storeKey = `decompile.recon.${runId}`;
  const [rid, setRid] = useState<string | null>(() => localStorage.getItem(storeKey));
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisErr, setAnalysisErr] = useState<string | null>(null);

  // choose the active reconstruction: the remembered one if still valid, else the newest completed one
  useEffect(() => {
    if (!run) return;
    if (rid && completed.some((r) => r.id === rid)) return;
    const latest = completed[completed.length - 1];
    setRid(latest ? latest.id : null);
  }, [run, completed, rid]);
  useEffect(() => {
    if (rid) localStorage.setItem(storeKey, rid);
  }, [rid, storeKey]);

  const finishedAt = completed.find((r) => r.id === rid)?.finished_utc;
  useEffect(() => {
    setAnalysis(null);
    setAnalysisErr(null);
    if (!rid) return;
    api
      .recon(runId, rid)
      .then((r) => setAnalysis(r.analysis))
      .catch((e) => setAnalysisErr((e as Error).message));
  }, [runId, rid, finishedAt]);

  if (error && !run)
    return (
      <div className="callout bad" role="alert">
        {error === "run not found" ? "This run does not exist on the server." : error} <a href="#/">Back to projects</a>
      </div>
    );
  if (!run) return <div className="skeleton" style={{ height: 320 }} />;

  const [label, tone] = STATE_LABEL[run.state] ?? [run.state, ""];
  const hasRecon = !!analysis;
  const go = (t: TabKey) => navigate(`/runs/${runId}/${t}`);
  const view = (() => {
    if (tab === "process")
      return (
        <ProcessTab
          run={run}
          health={health}
          events={events}
          refresh={refresh}
          activeRecon={rid}
          onOpenRecon={(id) => {
            setRid(id);
            go("reconstruction");
          }}
        />
      );
    if (tab === "artifacts") return <ArtifactsTab run={run} rid={rid} />;
    if (!rid)
      return (
        <Empty>
          No completed reconstruction yet. Run SAM tracking and a reconstruction from the <a href={`#/runs/${runId}/process`}>Processing</a> tab.
        </Empty>
      );
    if (analysisErr) return <div className="callout bad">{analysisErr}</div>;
    if (!analysis) return <div className="skeleton" style={{ height: 420 }} />;
    if (tab === "reconstruction") return <ReconstructionTab run={run} rid={rid} analysis={analysis} />;
    if (tab === "physics") return <PhysicsTab run={run} analysis={analysis} />;
    return <SimulationTab run={run} rid={rid} analysis={analysis} />;
  })();

  return (
    <div className="fade-in">
      <div className="runhead">
        <div style={{ minWidth: 0 }}>
          <div className="eyebrow">
            Run <span className="mono">{run.id}</span>
          </div>
          <h2 style={{ marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{run.original_filename}</h2>
          <div className="row wrap small muted" style={{ marginTop: 6 }}>
            <Badge tone={tone}>{label}</Badge>
            <Badge tone={run.provenance === "real" ? "info" : "warn"}>{run.provenance === "real" ? "real footage" : "synthetic video"}</Badge>
            {run.video?.frame_count !== undefined && <span>{run.video.frame_count} frames</span>}
            {run.video?.display_width && <span>{`${run.video.display_width}×${run.video.display_height}`}</span>}
            {run.video?.duration_s && <span>{fmt.num(run.video.duration_s, 3)} s</span>}
          </div>
        </div>
        <span className="grow" />
        {completed.length > 0 && (
          <label className="field" style={{ minWidth: 220 }}>
            <span>Active reconstruction</span>
            <select className="input" value={rid ?? ""} onChange={(e) => setRid(e.target.value)}>
              {completed.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} · {r.params.physics_model} · frames {r.params.start_frame}–{r.params.end_frame ?? "end"} ·{" "}
                  {r.outcome === "scene_program" ? "program" : "no program"}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <nav className="tabs" role="tablist">
        {TABS.map((t, i) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            className={`tab ${tab === t.key ? "active" : ""}`}
            disabled={(t.key === "reconstruction" || t.key === "physics" || t.key === "simulation") && !rid}
            onClick={() => go(t.key)}
          >
            <span className="num">{String(i + 1).padStart(2, "0")}</span>
            {t.label}
            {t.key === "simulation" && hasRecon && !analysis?.program && <Badge tone="warn" dot={false}>no program</Badge>}
          </button>
        ))}
      </nav>
      {view}
    </div>
  );
}
