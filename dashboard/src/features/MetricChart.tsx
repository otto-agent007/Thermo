import type { CellSummary, StudySummary } from "../../shared/model";
export type Metric =
  "loss" | "leakage" | "survival" | "hopMae" | "asymmetryMae";
export const metricLabels: Record<Metric, string> = {
  loss: "Population loss",
  leakage: "Terminal leakage",
  survival: "Uninterrupted survival",
  hopMae: "Hop MAE",
  asymmetryMae: "Asymmetry MAE",
};
export const metricValue = (n: number, metric: Metric) =>
  ["leakage", "survival"].includes(metric)
    ? `${(100 * n).toFixed(2)}%`
    : n.toPrecision(4);
const colors = { finite: "#146b67", equilibrium: "#5b60a8", frozen: "#a45e36" };
export function MetricChart({
  cells,
  metric,
  thresholds,
}: {
  cells: CellSummary[];
  metric: Metric;
  thresholds: StudySummary["thresholds"];
}) {
  const threshold = thresholds[metric];
  const finite = cells.filter((c) => c.horizon !== "equilibrium");
  const oracle = cells.filter((c) => c.horizon === "equilibrium");
  const bounds = (c: CellSummary): [number, number] =>
    metric === "loss"
      ? c.lossBounds
      : metric === "leakage"
        ? c.leakageInterval
        : [c[metric], c[metric]];
  const upper =
    Math.max(
      threshold,
      ...cells.map((c) => Math.max(c[metric], bounds(c)[1])),
    ) * 1.12;
  const lower = Math.min(
    0,
    ...cells.map((c) => Math.min(c[metric], bounds(c)[0])),
  );
  const y = (value: number) => 260 - ((value - lower) / (upper - lower)) * 220;
  const x = (h: number) => 58 + [1, 2, 4, 8, 16, 30].indexOf(h) * 87;
  return (
    <section className="panel metric-panel">
      <div className="row-between">
        <h2>{metricLabels[metric]}</h2>
        <span className="tag">
          Target {metric === "survival" ? "≥" : "≤"}{" "}
          {metricValue(threshold, metric)}
        </span>
      </div>
      <p className="fine">
        {metric === "loss"
          ? "U-statistic estimates are descriptive. Vertical bounds are the simultaneous gate bounds."
          : metric === "leakage"
            ? "Sampled leakage with recorded simultaneous Clopper–Pearson intervals."
            : "Exact reference values; no sampling interval."}{" "}
        Each seed remains separate.
      </p>
      {!cells.length ? (
        <p>No cells match these filters.</p>
      ) : (
        <>
          <div className="chart-wrap">
            <svg
              viewBox="0 0 720 310"
              role="img"
              aria-label={`${metricLabels[metric]} by finite horizon, oracle separate. Exact values are in the cell table.`}
            >
              {[0, 0.25, 0.5, 0.75, 1].map((t) => {
                const v = lower + (upper - lower) * t;
                return (
                  <g key={t}>
                    <line
                      x1="52"
                      x2="700"
                      y1={y(v)}
                      y2={y(v)}
                      stroke="#e3eaed"
                    />
                    <text x="45" y={y(v) + 4} textAnchor="end">
                      {metricValue(v, metric)}
                    </text>
                  </g>
                );
              })}
              <line
                x1="52"
                x2="700"
                y1={y(threshold)}
                y2={y(threshold)}
                stroke="#aa731e"
                strokeDasharray="5 5"
              />
              {[1, 2, 4, 8, 16, 30].map((h) => (
                <text key={h} x={x(h)} y="286" textAnchor="middle">
                  K{h}
                </text>
              ))}
              <line
                x1="562"
                x2="562"
                y1="30"
                y2="265"
                stroke="#c4d1d6"
                strokeDasharray="3 5"
              />
              <text x="641" y="286" textAnchor="middle">
                Oracle
              </text>
              {(["finite", "equilibrium", "frozen"] as const).flatMap(
                (member) =>
                  [0, 1, 2].map((seed) => {
                    const series = finite
                      .filter((c) => c.member === member && c.seed === seed)
                      .sort((a, b) => Number(a.horizon) - Number(b.horizon));
                    return (
                      <polyline
                        key={`${member}${seed}`}
                        points={series
                          .map((c) => `${x(Number(c.horizon))},${y(c[metric])}`)
                          .join(" ")}
                        fill="none"
                        stroke={colors[member]}
                        strokeWidth="1.8"
                        strokeDasharray={
                          seed === 1 ? "6 3" : seed === 2 ? "2 3" : undefined
                        }
                        opacity=".8"
                      />
                    );
                  }),
              )}
              {[...finite, ...oracle].map((c) => {
                const cx =
                  c.horizon === "equilibrium"
                    ? 603 +
                      ["equilibrium", "frozen"].indexOf(c.member) * 65 +
                      c.seed * 7
                    : x(c.horizon);
                const [lo, hi] = bounds(c);
                return (
                  <g key={c.id}>
                    <title>
                      {c.id}: {c[metric]} · bounds {lo}, {hi}
                    </title>
                    <line
                      x1={cx}
                      x2={cx}
                      y1={y(lo)}
                      y2={y(hi)}
                      stroke={colors[c.member]}
                    />
                    <circle
                      cx={cx}
                      cy={y(c[metric])}
                      r="3.5"
                      fill={colors[c.member]}
                      stroke="white"
                      strokeWidth="1"
                    />
                  </g>
                );
              })}
              <text x="282" y="308" textAnchor="middle">
                Finite inference horizon · complete Gibbs sweeps
              </text>
            </svg>
          </div>
          <div className="legend">
            {Object.entries(colors).map(([m, color]) => (
              <span key={m}>
                <i style={{ background: color }} />
                {m}
              </span>
            ))}
            <span>Seed 0 — solid · 1 — dashed · 2 — dotted</span>
          </div>
        </>
      )}
    </section>
  );
}
