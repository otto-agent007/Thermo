import { useEffect, useState } from "react";
import {
  BookOpen,
  FileCheck,
  FlaskConical,
  House,
  Map,
  RefreshCw,
  ArrowUpRight,
} from "lucide-react";
import { useProject } from "./useProject";
import { Experiments } from "./features/StudyBrowser";
import { Overview } from "./features/Overview";
import { Roadmap } from "./features/Roadmap";
import { Proposals } from "./features/Proposals";
import { Research } from "./features/Research";
const nav = [
  ["overview", "Overview", House],
  ["experiments", "Experiments", FlaskConical],
  ["roadmap", "Roadmap", Map],
  ["research", "Research", BookOpen],
  ["proposals", "Proposals", FileCheck],
] as const;
const titles: Record<string, string> = {
  overview: "Project overview",
  experiments: "Experiments",
  roadmap: "Roadmap and activity",
  research: "Research",
  proposals: "Proposals",
};
export default function App() {
  const { data, error, refreshing, refresh } = useProject();
  const [view, setView] = useState(
    location.hash.slice(1).split("/")[0] || "overview",
  );
  useEffect(() => {
    const change = () =>
      setView(location.hash.slice(1).split("/")[0] || "overview");
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  return (
    <div className="app">
      <a
        className="skip"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#overview">
          Thermo
        </a>
        <span className="brand-caption">Research workspace</span>
        <nav aria-label="Main navigation">
          {nav.map(([id, label, Icon]) => (
            <a
              key={id}
              href={`#${id}`}
              aria-current={view === id ? "page" : undefined}
            >
              <Icon size={19} aria-hidden="true" />
              {label}
            </a>
          ))}
        </nav>
        <div className="sidebar-foot">
          <FlaskConical size={19} />
          <strong>Software evidence</strong>
          <p>Code, analysis and reproducibility.</p>
          <a
            href="https://github.com/otto-agent007/Thermo"
            target="_blank"
            rel="noreferrer"
          >
            <ArrowUpRight size={17} />
            View on GitHub
          </a>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div>
            <h1>{titles[view] ?? "Project overview"}</h1>
            <p>
              Research <span>/</span> Thermo <span>/</span>{" "}
              {titles[view] ?? "Overview"}
            </p>
          </div>
          <div className="refresh">
            <div className="fine">
              {data?.mode === "snapshot" ? "Snapshot exported" : "Last read"}
              <br />
              {data ? new Date(data.generatedAt).toLocaleString() : "Loading…"}
            </div>
            <button onClick={refresh} disabled={refreshing}>
              <RefreshCw size={16} />
              {refreshing ? "Reading…" : "Refresh"}
            </button>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          {error && (
            <p role="alert" className="notice">
              {data ? "Showing previous data. " : ""}
              {error}
            </p>
          )}
          {data?.issues.map((i) => (
            <p role="alert" className="notice" key={i}>
              {i}
            </p>
          ))}
          {!data ? (
            <p>Loading project evidence…</p>
          ) : (
            <>
              {view === "proposals" ? (
                <Proposals local={data.mode === "local"} />
              ) : view === "experiments" ? (
                <Experiments snapshot={data} />
              ) : view === "roadmap" ? (
                <Roadmap snapshot={data} />
              ) : view === "research" ? (
                <Research snapshot={data} />
              ) : (
                <Overview snapshot={data} />
              )}
              <footer>
                {data.mode === "snapshot"
                  ? "Published snapshot · Refresh reloads this publication. New studies require a new publication."
                  : "Local records · refresh every 15 seconds while visible."}{" "}
                <span>
                  M4G evidence commit{" "}
                  {data.sourceCommit?.slice(0, 12) ?? "unavailable"}
                </span>
                <a
                  href="https://github.com/otto-agent007/Thermo/pull/35"
                  target="_blank"
                  rel="noreferrer"
                >
                  M4G pull request / CI
                </a>
              </footer>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
