import type { CyclePhase, ResearchCycle as Cycle } from "../../shared/model";

const phases: Record<CyclePhase, string> = {
  queued: "Queued",
  running: "Running (reported)",
  verifying: "Verifying evidence",
  awaiting_review: "Awaiting independent review",
  recorded: "Recorded (reported)",
  blocked: "Blocked",
  failed: "Execution failed",
};
const reviews = {
  pending: "Pending",
  changes_requested: "Changes requested",
  approved: "Approved (reported)",
};

export function ResearchCycle({
  cycle,
  mode,
}: {
  cycle: Cycle | null;
  mode: "local" | "snapshot";
}) {
  if (!cycle) return null;
  const status = cycle.status;
  const independentReview = cycle.independentReview;
  const reviewRecord = independentReview?.record;
  const ageSeconds = Math.floor(
    (Date.now() - Date.parse(cycle.observedAt)) / 1000,
  );
  return (
    <section className="panel research-cycle" aria-labelledby="cycle-heading">
      <span className="tag">
        {mode === "snapshot" ? "Staged snapshot" : "Local status observation"}
      </span>
      <h2 id="cycle-heading">Research cycle</h2>
      <p className="fine">
        Observed <time dateTime={cycle.observedAt}>{cycle.observedAt}</time>.
        This is a status-file snapshot, not live worker telemetry.
        {mode === "snapshot" &&
          " Refresh reloads this publication; a new export is needed for newer status."}
      </p>
      <p className="fine">
        Snapshot age at render:{" "}
        {ageSeconds >= 0
          ? `${ageSeconds.toLocaleString("en-US")} seconds`
          : "unavailable (observation clock is ahead)"}
        .
      </p>
      {cycle.notice && (
        <p className="notice" role="status">
          {cycle.notice}
        </p>
      )}
      {status && (
        <>
          <h3>{phases[status.phase]}</h3>
          <p>{status.message}</p>
          <dl className="detail-metrics">
            <div>
              <dt>Job</dt>
              <dd className="mono">{status.jobId}</dd>
            </div>
            <div>
              <dt>Worker-reported review status</dt>
              <dd>{reviews[status.reviewStatus]}</dd>
            </div>
            <div>
              <dt>Attempts used</dt>
              <dd>{status.attempts}</dd>
            </div>
            <div>
              <dt>Cumulative active time</dt>
              <dd>
                {status.elapsedSeconds.toLocaleString("en-US", {
                  maximumFractionDigits: 1,
                })}{" "}
                seconds
              </dd>
            </div>
            <div>
              <dt>Last reported heartbeat</dt>
              <dd>
                {status.heartbeatAt ? (
                  <time dateTime={status.heartbeatAt}>
                    {status.heartbeatAt}
                  </time>
                ) : (
                  "Not reported"
                )}
              </dd>
            </div>
            <div>
              <dt>Heartbeat age at observation</dt>
              <dd>
                {cycle.heartbeatAgeSeconds === null
                  ? "Unavailable"
                  : `${cycle.heartbeatAgeSeconds.toLocaleString("en-US")} seconds`}
              </dd>
            </div>
          </dl>
          <p className="fine">
            The configured job budget is not included in this status. Consult
            the approved request for its time and attempt limits.
          </p>
          <dl className="detail-metrics">
            <div>
              <dt>Pinned source SHA</dt>
              <dd className="mono">{status.sourceSha}</dd>
            </div>
            <div>
              <dt>Request digest</dt>
              <dd className="mono">{status.requestDigest}</dd>
            </div>
            <div>
              <dt>Evidence manifest digest (reported)</dt>
              <dd className="mono">
                {status.evidenceDigest ?? "Not reported"}
              </dd>
            </div>
            <div>
              <dt>Scientific outcome</dt>
              <dd>
                Consult the reviewed evidence. A negative result can be a
                successful execution.
              </dd>
            </div>
          </dl>
          <h3>Independent review record</h3>
          {independentReview?.notice && (
            <p className="notice" role="status">
              {independentReview.notice}
            </p>
          )}
          {!independentReview && (
            <p className="fine">
              No independent review record is available in this observation.
            </p>
          )}
          {reviewRecord && (
            <>
              <dl className="detail-metrics">
                <div>
                  <dt>Review decision</dt>
                  <dd>
                    {reviewRecord.decision === "approved"
                      ? "Approved by reviewer (reported)"
                      : "Changes requested"}
                  </dd>
                </div>
                <div>
                  <dt>Reviewer identity</dt>
                  <dd>{reviewRecord.reviewer}</dd>
                </div>
                <div>
                  <dt>Reviewed at</dt>
                  <dd>
                    <time dateTime={reviewRecord.reviewedAt}>
                      {reviewRecord.reviewedAt}
                    </time>
                  </dd>
                </div>
                <div>
                  <dt>Replay scope</dt>
                  <dd>
                    {reviewRecord.replay === "executed"
                      ? "Replay executed by reviewer (reported)"
                      : "Replay artifacts inspected; reviewer did not execute replay"}
                  </dd>
                </div>
                <div>
                  <dt>Evidence commit reviewed (reported)</dt>
                  <dd className="mono">{reviewRecord.evidenceCommit}</dd>
                </div>
              </dl>
              <p className="scope">
                This review record matches the displayed evidence identity.
                Reviewer approval does not establish owner acceptance or change
                the worker phase. Scope and findings remain in the authoritative
                review file.
              </p>
            </>
          )}
          <p className="scope">
            This panel reports worker state; it does not authenticate the
            evidence or establish scientific acceptance.
          </p>
        </>
      )}
    </section>
  );
}
