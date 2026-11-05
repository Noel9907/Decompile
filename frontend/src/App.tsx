import { useEffect, useState } from "react";
import { api } from "./api";
import Home from "./pages/Home";
import RunPage, { type TabKey } from "./pages/RunPage";
import type { Health } from "./types";
import { Logo } from "./ui";

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
  window.scrollTo({ top: 0 });
}

export default function App() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [theme, setTheme] = useState<string>(() => localStorage.getItem("decompile.theme") ?? "light");
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("decompile.theme", theme);
  }, [theme]);

  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    api.health().then(setHealth).catch(() => setOffline(true));
  }, []);

  const goHome = (anchor?: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    window.location.hash = "/";
    setTimeout(() => (anchor ? document.getElementById(anchor)?.scrollIntoView() : window.scrollTo({ top: 0 })), 30);
  };

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/" onClick={goHome()}>
          <Logo />
          <span className="brand-name">Decompile</span>
        </a>
        <nav className="nav">
          <a href="#how" onClick={goHome("how")}>How it works</a>
          <a href="#upload" onClick={goHome("upload")}>Decompile a video</a>
          <a href="#projects" onClick={goHome("projects")}>Projects</a>
        </nav>
        <span className="spacer" />
        <button className="theme-toggle" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label="toggle colour theme" title="Toggle light / dark">
          {theme === "light" ? "☾" : "☀"}
        </button>
        <span className="pill" title="Meta Model API status on this server">
          {offline ? (
            <><span className="led bad" />server offline</>
          ) : health ? (
            <><span className={`led ${health.api_key_configured ? "on" : "off"}`} />
              SAM 3.1 {health.live_sam_enabled && health.api_key_configured ? "live" : "off"} · Muse Spark {health.spark_available ? "live" : "off"}</>
          ) : (
            <>connecting…</>
          )}
        </span>
      </header>
      <main className="page">
        {route.page === "home" ? <Home health={health} /> : <RunPage key={route.id} runId={route.id} tab={route.tab} health={health} />}
      </main>
    </>
  );
}
