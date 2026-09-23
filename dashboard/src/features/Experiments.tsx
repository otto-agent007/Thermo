import { useState } from "react";
import type {
  Filters,
  Horizon,
  Member,
  ProjectSnapshot,
} from "../../shared/model";
import { Sources } from "../components/SourceLink";
import { MetricChart, metricLabels, type Metric } from "./MetricChart";
import { CellTable } from "./CellTable";
import { CellDetail } from "./CellDetail";
import { CostView } from "./CostView";
import { downloadCsv, filterCells, toCsv } from "./filters";
import { Download } from "lucide-react";
const initial: Filters = { seeds: [], members: [], horizons: [], query: "" };
export function Experiments({ snapshot }: { snapshot: ProjectSnapshot }) {
  const [filters, setFilters] = useState<Filters>(initial);
  const [metric, setMetric] = useState<Metric>("leakage");
  const [id, setId] = useState<string | null>(null);
  const s = snapshot.study;
  if (!s)
    return (
      <section className="panel">
        <h2>Evidence unavailable</h2>
        <p>Check the archive before exploring results.</p>
      </section>
    );
  const rows = filterCells(s.cells, filters);
  return (
    <>
      <div className="page-intro">
        <h2>Task quality vs inference budget</h2>
        <p>
          M4G · 60 held-out cells · all three seeds retained · software
          simulation and exact reference
        </p>
      </div>
      <details className="panel catalog">
        <summary>
          Experiment catalog · {snapshot.milestones.length - 1} recorded
          milestones
        </summary>
        {snapshot.milestones
          .filter((m) => m.id !== "M5")
          .map((m) => (
            <article key={m.id}>
              <h3>
                {m.id} · {m.question}
              </h3>
              <span className="tag">{m.state}</span>
              <p>{m.conclusion}</p>
              <p className="fine">{m.evidenceClasses.join(" · ")}</p>
              <Sources sources={m.sources} />
            </article>
          ))}
      </details>
      <div className="filters">
        <label>
          Seed
          <select
            aria-label="Seed"
            value={filters.seeds[0] ?? "all"}
            onChange={(e) =>
              setFilters({
                ...filters,
                seeds: e.target.value === "all" ? [] : [Number(e.target.value)],
              })
            }
          >
            <option value="all">All seeds</option>
            {[0, 1, 2].map((n) => (
              <option key={n} value={n}>
                Seed {n}
              </option>
            ))}
          </select>
        </label>
        <label>
          Training member
          <select
            aria-label="Training member"
            value={filters.members[0] ?? "all"}
            onChange={(e) =>
              setFilters({
                ...filters,
                members:
                  e.target.value === "all" ? [] : [e.target.value as Member],
              })
            }
          >
            <option value="all">All members</option>
            {["finite", "equilibrium", "frozen"].map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Inference horizon
          <select
            aria-label="Inference horizon"
            value={filters.horizons[0] ?? "all"}
            onChange={(e) =>
              setFilters({
                ...filters,
                horizons:
                  e.target.value === "all"
                    ? []
                    : [
                        e.target.value === "equilibrium"
                          ? "equilibrium"
                          : (Number(e.target.value) as Horizon),
                      ],
              })
            }
          >
            <option value="all">All horizons</option>
            {[1, 2, 4, 8, 16, 30].map((h) => (
              <option value={h} key={h}>
                K{h}
              </option>
            ))}
            <option value="equilibrium">Oracle</option>
          </select>
        </label>
        <label>
          Search cells
          <input
            value={filters.query}
            placeholder="Cell ID or decision"
            onChange={(e) => setFilters({ ...filters, query: e.target.value })}
          />
        </label>
        <button onClick={() => setFilters(initial)}>Reset</button>
      </div>
      <div className="metric-tabs" role="group" aria-label="Metric">
        {Object.entries(metricLabels).map(([m, label]) => (
          <button
            key={m}
            aria-pressed={metric === m}
            onClick={() => setMetric(m as Metric)}
          >
            {label}
          </button>
        ))}
      </div>
      <MetricChart cells={rows} metric={metric} thresholds={s.thresholds} />
      <section className="panel results">
        <div className="row-between">
          <h2>
            Evaluation cells <span className="tag">{rows.length} / 60</span>
          </h2>
          <button onClick={() => downloadCsv(toCsv(rows))}>
            <Download size={16} />
            Download selected CSV
          </button>
        </div>
        <CellTable cells={rows} metric={metric} onSelect={setId} />
        <Sources
          sources={s.sources.filter((x) => x.label === "Full recorded CSV")}
        />
      </section>
      <details className="panel">
        <summary>18 paired diagnostics · descriptive only</summary>
        <p className="fine">
          Finite-trained minus equilibrium-trained population loss. Approximate
          pointwise paired jackknife normal 95% intervals, conditional on the
          frozen parameter pair. These do not determine quality acceptance.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Comparison</th>
                <th>Difference</th>
                <th>Interval</th>
                <th>Conclusion</th>
              </tr>
            </thead>
            <tbody>
              {s.pairedDiagnostics.map((p) => (
                <tr key={p.label}>
                  <td>{p.label}</td>
                  <td>{p.estimate.toPrecision(5)}</td>
                  <td>{p.interval.map((v) => v.toPrecision(5)).join(" – ")}</td>
                  <td>{p.conclusion}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <CostView study={s} />
      <Sources sources={s.sources} />
      {id && <CellDetail id={id} onClose={() => setId(null)} />}
    </>
  );
}
