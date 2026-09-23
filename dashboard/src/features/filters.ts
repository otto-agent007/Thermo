import type { CellSummary, Filters } from "../../shared/model";
export function filterCells(
  cells: CellSummary[],
  filters: Filters,
): CellSummary[] {
  const q = filters.query.trim().toLowerCase();
  return cells.filter(
    (c) =>
      (!filters.seeds.length || filters.seeds.includes(c.seed)) &&
      (!filters.members.length || filters.members.includes(c.member)) &&
      (!filters.horizons.length || filters.horizons.includes(c.horizon)) &&
      `${c.id} ${c.decision}`.toLowerCase().includes(q),
  );
}
function field(value: string | number) {
  let s = String(value);
  if (typeof value === "string" && /^[=+@\-\t\r]/.test(s)) s = "'" + s;
  return /[",\n\r]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
}
export function toCsv(cells: CellSummary[]): string {
  return (
    [
      "cell_id,seed,member,horizon,evaluation_seed,loss,loss_lower,loss_upper,leakage,leakage_lower,leakage_upper,survival,hop_mae,asymmetry_mae,decision",
      ...cells.map((c) =>
        [
          c.id,
          c.seed,
          c.member,
          c.horizon,
          c.evaluationSeed,
          c.loss,
          ...c.lossBounds,
          c.leakage,
          ...c.leakageInterval,
          c.survival,
          c.hopMae,
          c.asymmetryMae,
          c.decision,
        ]
          .map(field)
          .join(","),
      ),
    ].join("\r\n") + "\r\n"
  );
}
export function downloadCsv(text: string): void {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "thermo-selected-cells.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
