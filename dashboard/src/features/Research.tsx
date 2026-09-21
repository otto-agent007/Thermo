import type { ProjectSnapshot } from "../../shared/model";
import { Sources } from "../components/SourceLink";
export function Research({ snapshot }: { snapshot: ProjectSnapshot }) {
  return (
    <>
      <div className="page-intro">
        <h2>What we know. What to test next.</h2>
        <p>
          Findings, deductions and proposals have different evidential weight.
        </p>
      </div>
      <div className="research-list">
        {snapshot.research.map((r) => (
          <article className="panel research-item" key={r.id}>
            <span className={`tag ${r.kind === "proposal" ? "proposal" : ""}`}>
              {r.kind === "proposal"
                ? "Proposed"
                : r.kind === "deduction"
                  ? "Mathematical deduction"
                  : "Reported finding"}
            </span>
            <h2>{r.title}</h2>
            <p>{r.text}</p>
            <p className="scope">{r.scope}</p>
            <Sources sources={r.sources} />
          </article>
        ))}
      </div>
    </>
  );
}
