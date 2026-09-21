import type { StudySummary } from "../../shared/model";
import { SourceLink } from "../components/SourceLink";
export function CostView({ study }: { study: StudySummary }) {
  return (
    <section className="panel">
      <h2>Complete cost accounting</h2>
      <p>
        Count the whole finite training grid. Oracle sweep cost remains
        unassigned.
      </p>
      <div className="cost-list">
        {study.costs.map((c) => (
          <article key={c.label}>
            <div className="row-between">
              <h3>{c.label}</h3>
              <strong>
                {c.value === null
                  ? "Unassigned"
                  : c.value.toLocaleString(undefined, {
                      maximumFractionDigits: 3,
                    })}{" "}
                <small>{c.unit}</small>
              </strong>
            </div>
            <p>{c.scope}</p>
            <span className="fine">{c.evidence} · </span>
            <SourceLink source={c.source} />
          </article>
        ))}
      </div>
    </section>
  );
}
