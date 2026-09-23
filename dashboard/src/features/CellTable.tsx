import type { CellSummary } from "../../shared/model";
import { metricValue, type Metric } from "./MetricChart";
export function CellTable({
  cells,
  metric,
  onSelect,
}: {
  cells: CellSummary[];
  metric: Metric;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="table-scroll">
      <table>
        <caption>
          {cells.length} evaluation cells · select a cell for exact metrics and
          source identities
        </caption>
        <thead>
          <tr>
            <th>Cell</th>
            <th>Seed</th>
            <th>Member</th>
            <th>Horizon</th>
            <th>Value</th>
            <th>Recorded bounds</th>
            <th>Joint quality</th>
          </tr>
        </thead>
        <tbody>
          {cells.map((c) => (
            <tr key={c.id}>
              <td>
                <button
                  className="cell-link"
                  onClick={() => onSelect(c.id)}
                  aria-label={`Open ${c.id}`}
                >
                  Details
                </button>
              </td>
              <td>{c.seed}</td>
              <td>{c.member}</td>
              <td>
                {c.horizon === "equilibrium" ? "Oracle" : `K${c.horizon}`}
              </td>
              <td title={String(c[metric])}>
                {metricValue(c[metric], metric)}
              </td>
              <td>
                {metric === "loss"
                  ? c.lossBounds.map((v) => metricValue(v, metric)).join(" – ")
                  : metric === "leakage"
                    ? c.leakageInterval
                        .map((v) => metricValue(v, metric))
                        .join(" – ")
                    : "Exact reference"}
              </td>
              <td>
                <span className="quality-fail">
                  {c.decision === "fail" ? "Unmet" : c.decision}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!cells.length && (
        <p className="empty">
          No cells match these filters. Reset to show all 60 cells.
        </p>
      )}
    </div>
  );
}
