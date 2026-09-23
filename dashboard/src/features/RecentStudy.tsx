import type { RecentStudy, ReportTable } from "../../shared/model";
import { Sources } from "../components/SourceLink";

export function ReportTableView({ table }: { table: ReportTable }) {
  return (
    <div className="table-scroll" tabIndex={0} aria-label={table.title}>
      <table>
        <caption>{table.title}</caption>
        <thead>
          <tr>
            {table.columns.map((column, i) => (
              <th key={i} scope="col">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RecentStudyPanel({
  study,
  compact = false,
}: {
  study: RecentStudy;
  compact?: boolean;
}) {
  const available = study.availability === "available";
  const comparison: ReportTable | null =
    compact && available && study.id === "full-row"
      ? {
          title: "All four weights · survival and complete-row error",
          columns: [
            "Weight",
            "Survival",
            "Worst local error",
            "Meets both gates",
          ],
          rows: study.tables[0].rows.map((row) => [
            row[0],
            `${(Number(row[1]) * 100).toFixed(2)}%`,
            Number(row[3]).toFixed(4),
            row[9] === "True" ? "Yes" : "No",
          ]),
        }
      : null;
  return (
    <section className="panel recent-study">
      <div className="row-between study-meta">
        <span className="tag">
          {available ? "Archived result" : "Evidence unavailable"}
        </span>
        <time dateTime={study.date}>{study.date}</time>
      </div>
      <h2>{study.title}</h2>
      <p className="study-finding">{study.finding}</p>
      <p className="scope">{study.scope}</p>
      {available && (
        <>
          <p className="study-gate">
            <strong>Original criterion:</strong> {study.gate}
          </p>
          {comparison ? (
            <ReportTableView table={comparison} />
          ) : (
            study.tables.map((table, index) =>
              index === 0 ? (
                <ReportTableView key={table.title} table={table} />
              ) : (
                <details key={table.title}>
                  <summary>{table.title}</summary>
                  <ReportTableView table={table} />
                </details>
              ),
            )
          )}
          <p className="fine">
            Exact reference · recorded verification · report bytes
            authenticated. Numerical replay is recorded in the archive.
          </p>
        </>
      )}
      <Sources sources={study.sources} />
      {compact && (
        <a className="text-action" href="#experiments">
          Explore all recent studies →
        </a>
      )}
    </section>
  );
}
