import { useCallback, useEffect, useRef, useState } from "react";
import type { ProposalList } from "../../shared/model";
const label = (value: string) => value.replaceAll("_", " ");
export function Proposals({ local }: { local: boolean }) {
  const [data, setData] = useState<ProposalList | null>(null);
  const [error, setError] = useState(false);
  const [reading, setReading] = useState(false);
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(() => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setReading(true);
    fetch("./data/proposals.json", {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) throw Error("unavailable");
        const value: ProposalList = await response.json();
        if (!Array.isArray(value.items) || !Array.isArray(value.issues))
          throw Error("invalid");
        if (!controller.signal.aborted) {
          setData(value);
          setError(false);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setReading(false);
      });
  }, []);
  useEffect(() => {
    refresh();
    const timer = local
      ? setInterval(() => {
          if (document.visibilityState === "visible") refresh();
        }, 15000)
      : undefined;
    return () => {
      clearInterval(timer);
      request.current?.abort();
    };
  }, [local, refresh]);
  return (
    <section aria-labelledby="proposals-title" className="proposals">
      <div className="row-between">
        <div>
          <h2 id="proposals-title">Improvement proposals</h2>
          <p className="muted">
            Local drafts and evidence for owner review. Read-only.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={reading}
          aria-label="Refresh proposals"
        >
          {reading ? "Reading…" : "Refresh proposals"}
        </button>
      </div>
      <p className="proposal-scope">
        A passing check does not accept a proposal. Dashboard changes need
        visual review; research trials are bounded three-site exact trials, not
        full studies or hardware evidence.
      </p>
      {error && (
        <p role="alert" className="notice">
          {data ? "Showing previous proposals. " : ""}Proposal refresh failed.
          Try again.
        </p>
      )}
      {data?.issues.map((issue) => (
        <p role="alert" className="notice" key={issue}>
          {issue}
        </p>
      ))}
      {!data && !error && <p role="status">Loading proposals…</p>}
      {data?.items.length === 0 && (
        <div className="panel">
          <h3>No proposals available</h3>
          <p>
            {local
              ? "No validated local proposal records are available yet."
              : "Local drafts are excluded from this static snapshot."}
          </p>
        </div>
      )}
      <div className="proposal-list">
        {data?.items.map((item) => (
          <article className="panel proposal-card" key={item.id}>
            <div className="row-between">
              <span className="tag proposal">
                {item.track === "research"
                  ? "Research · exact reference"
                  : "Dashboard · functional checks"}
              </span>
              <span className="muted">Draft proposal</span>
            </div>
            <h3>{item.objective}</h3>
            <p className="proposal-recommendation">{item.recommendation}</p>
            <dl className="proposal-status">
              <div>
                <dt>Execution</dt>
                <dd>{label(item.execution)}</dd>
              </div>
              <div>
                <dt>Verification</dt>
                <dd>{label(item.verification)}</dd>
              </div>
              <div>
                <dt>Research result</dt>
                <dd>{label(item.researchOutcome)}</dd>
              </div>
              <div>
                <dt>Owner review</dt>
                <dd>{label(item.review)}</dd>
              </div>
            </dl>
            <div className="proposal-comparison">
              <div>
                <strong>Baseline</strong>
                <p>{item.baselineSummary}</p>
              </div>
              <div>
                <strong>Candidate</strong>
                <p>{item.candidateSummary}</p>
              </div>
            </div>
            <p className="fine">
              {item.track === "research"
                ? "Bounded exact trial · three-site fixture only."
                : "Automated checks are separate from owner visual review."}
            </p>
            <dl className="proposal-source">
              <div>
                <dt>Baseline commit</dt>
                <dd className="mono">{item.baseline}</dd>
              </div>
              <div>
                <dt>Candidate ID</dt>
                <dd className="mono">{item.id}</dd>
              </div>
            </dl>
            <div className="proposal-links">
              {item.patchUrl ? (
                <a href={`.${item.patchUrl}`}>View draft patch</a>
              ) : (
                <span>Patch unavailable</span>
              )}
              {item.screenshotUrls.map((url) => (
                <a key={url} href={`.${url}`}>
                  View {url.split("/").at(-1)?.replace(".png", "")} screenshot
                </a>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
