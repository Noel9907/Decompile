import { useEffect, useState } from "react";
import { api } from "./api";
import Home from "./pages/Home";
import RunPage, { type TabKey } from "./pages/RunPage";
import type { Health } from "./types";
import { Badge, Logo } from "./ui";

/** Hash routes keep the current run in the URL, so a refresh restores the workspace:
 *  #/                      home
 *  #/runs/<id>/<tab>       run workspace */
export function parseHash(hash: string): { page: "home" } | { page: "run"; id: string; tab: TabKey } {
  const m = hash.replace(/^#/, "").match(/^\/runs\/([A-Za-z0-9_]+)(?:\/([a-z]+))?/);
  if (m) return { page: "run", id: m[1], tab: (m[2] as TabKey) || "process" };
  return { page: "home" };
}

export function navigate(path: string) {
  window.location.hash = path;
}

export default function App() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setOffline(true));
  }, []);

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/">
          <Logo />
          <span className="brand-name">DECOMPILE</span>
          <span className="brand-tag">Turn reality into executable physics</span>
        </a>
        <span className="spacer" />
        <span className="server-pill">
          {offline ? (
            <Badge tone="bad">backend unreachable</Badge>
          ) : health ? (
            <>
              <Badge tone={health.live_sam_enabled && health.api_key_configured ? "good" : "warn"}>
                {health.live_sam_enabled ? (health.api_key_configured ? "live SAM 3.1 enabled" : "live SAM: no API key") : "live SAM disabled"}
              </Badge>
              <span className="mono tiny">{health.version}</span>
            </>
          ) : null}
        </span>
      </header>
      <main className="page">
        {route.page === "home" ? (
          <Home health={health} />
        ) : (
          <RunPage key={route.id} runId={route.id} tab={route.tab} health={health} />
        )}
      </main>
    </>
  );
}
