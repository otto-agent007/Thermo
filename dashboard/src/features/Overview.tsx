import type { ProjectSnapshot } from "../../shared/model";
import { Status } from "../components/Status";
import { Sources } from "../components/SourceLink";
import { ActivityPanel } from "./Roadmap";
import { ArrowRight } from "lucide-react";
const percent = (n: number) => (100 * n).toFixed(2) + "%";
export function Overview({ snapshot }: { snapshot: ProjectSnapshot }) {
  const s = snapshot.study;
  const cells = s?.cells ?? [];
  const leak = cells.map((c) => c.leakage);
  const survival = cells.map((c) => c.survival);
  return (
    <>
      <div className="page-intro">
        <h2>
          {s ? (
            <>
              A completed study.
              <br />A clear next question.
            </>
          ) : (
            "Project evidence unavailable"
          )}
        </h2>
        <p>
          M4G tests whether limited sampling budgets can preserve task quality.
        </p>
      </div>
      <div className="content-grid">
        <div className="main-column">
          <section className="status-strip">
            <Status title="Execution" status={snapshot.execution} />
            <Status title="Verification" status={snapshot.verification} />
            <Status title="Science" status={snapshot.science} />
          </section>
          {s ? (
            <>
              <dl className="counts">
                {Object.entries(s.counts).map(([k, v]) => (
                  <div key={k}>
                    <dd>{v}</dd>
                    <dt>
                      {
                        {
                          fits: "Fits",
                          updates: "Updates",
                          cells: "Evaluation cells",
                          pairs: "Paired diagnostics",
                        }[k]
                      }
                    </dt>
                  </div>
                ))}
              </dl>
              <section className="panel finding">
                <h2>Conservation is the limiting factor</h2>
                <p>
                  At K30, loss and local fidelity pass. Conservation does not.
                </p>
                <div className="quality-row">
                  <div className="row-between">
                    <strong>Terminal leakage</strong>
                    <strong>
                      {percent(Math.min(...leak))}–{percent(Math.max(...leak))}
                    </strong>
                  </div>
                  <div className="bar">
                    <span style={{ width: `${100 * Math.max(...leak)}%` }} />
                  </div>
                  <p className="fine">
                    Target ≤{percent(s.thresholds.leakage)} · sampled terminal
                    counts
                  </p>
                </div>
                <div className="quality-row">
                  <div className="row-between">
                    <strong>Uninterrupted survival</strong>
                    <strong>At most {percent(Math.max(...survival))}</strong>
                  </div>
                  <div className="bar">
                    <span
                      style={{ width: `${100 * Math.max(...survival)}%` }}
                    />
                  </div>
                  <p className="fine">
                    Target ≥{percent(s.thresholds.survival)} · exact
                    500-operation reference
                  </p>
                </div>
                <p className="fine">
                  Ranges span all 60 cells. All 60 fail the joint contract. No
                  inference-sweep savings established.
                </p>
                <a className="text-action" href="#experiments">
                  Explore the evidence <ArrowRight size={16} />
                </a>
              </section>
              <section className="panel">
                <h2>Recent evidence</h2>
                <div className="evidence-row">
                  <span className="milestone-id">M4G</span>
                  <div>
                    <strong>Task quality vs inference budget</strong>
                    <p>Completed release · quality unmet</p>
                  </div>
                </div>
                <Sources sources={s.sources} />
                <p className="fine">
                  Verification is recorded in the source archive. This dashboard
                  checks archive identity; it does not rerun numerical replay.
                </p>
              </section>
            </>
          ) : (
            <section className="panel">
              <h2>Evidence unavailable</h2>
              <p>
                The supported archive could not be read. Metrics are withheld.
              </p>
            </section>
          )}
        </div>
        <aside>
          <ActivityPanel activity={snapshot.activity} />
          <section className="panel next">
            <h2>Next on the roadmap</h2>
            <strong>M5 · Topology-aware meta-EBM</strong>
            <p>
              Reproduce the 12-spin target and evaluate connectivity and
              complete execution costs.
            </p>
            <a className="text-action" href="#roadmap">
              View roadmap <ArrowRight size={16} />
            </a>
            <hr />
            <h3>Research proposals</h3>
            <p>
              Exact survival-gradient audit and a count-preserving joint sampler
              pilot.
            </p>
            <a href="#research">Read the research</a>
          </section>
        </aside>
      </div>
    </>
  );
}
