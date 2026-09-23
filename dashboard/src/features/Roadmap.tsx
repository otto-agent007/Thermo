import type { ProjectSnapshot, Activity } from "../../shared/model";
import { Sources } from "../components/SourceLink";
export function ActivityPanel({ activity }: { activity: Activity }) {
  return (
    <section className="panel activity">
      <h2>Recorded activity</h2>
      <p>Saved checkpoints are not a live heartbeat.</p>
      <dl>
        <dt>Persisted checkpoints</dt>
        <dd>
          {activity.checkpointCount === null
            ? "Unavailable"
            : `${activity.checkpointCount} / ${activity.expectedCheckpoints}`}
        </dd>
        <dt>Last file modification</dt>
        <dd>
          {activity.latestFileAt
            ? new Date(activity.latestFileAt).toLocaleString()
            : "Unavailable"}
        </dd>
        <dt>Observed state</dt>
        <dd>{activity.label}</dd>
      </dl>
      {activity.issues.map((i) => (
        <p className="notice" key={i}>
          {i}
        </p>
      ))}
      <p className="fine">
        Local files matched to archived fit slots; release completion comes from the
        archive.
      </p>
    </section>
  );
}
export function Roadmap({ snapshot }: { snapshot: ProjectSnapshot }) {
  return (
    <>
      <div className="page-intro">
        <h2>From local gates to complete programs</h2>
        <p>
          Each milestone asks a bounded question. Follow the evidence, including
          negative results.
        </p>
      </div>
      <div className="content-grid">
        <section className="panel">
          <div className="milestones">
            {snapshot.milestones.map((m) => (
              <article className="milestone" key={m.id}>
                <span className="milestone-id">{m.id}</span>
                <div>
                  <div className="row-between">
                    <h3>{m.question}</h3>
                    <span className="tag">{m.state}</span>
                  </div>
                  <p>{m.conclusion}</p>
                  <p className="fine">
                    {m.evidenceClasses.join(" · ") || "Proposed future work"}
                  </p>
                  <Sources sources={m.sources} />
                </div>
              </article>
            ))}
          </div>
        </section>
        <ActivityPanel activity={snapshot.activity} />
      </div>
    </>
  );
}
