import { createHash } from "node:crypto";
import type { RecentStudy, ReportTable, Source } from "../shared/model.ts";
import { readBounded } from "./files.ts";
import pins from "./recent-pins.json" with { type: "json" };

export const recentEvidenceCommit = "918f67f62b9478276bc23d9e4a9a805937c58cf4";
const studies = [
  {
    id: "full-row",
    date: "2026-09-23",
    title: "Complete local-row fidelity",
    directory: "2026-09-23-return-fixture-full-row-pilot",
    status: "return_fixture_full_row_pilot_complete",
    finding:
      "No arm qualifies. Weight 100 lowers the worst local error from 0.4163 to 0.0851, while survival falls from 93.47% to 90.66%.",
    scope:
      "Three operations · K4 · four weights · 201 exact evaluations per arm · zero samples. This does not settle joint feasibility or kernel capacity.",
    gate: "Survival ≥ 95% AND maximum error across all 16 conditional entries ≤ 0.005. All four arms fail both thresholds.",
    tableTitles: [
      "All four final arms",
      "Complete row and optimizer diagnostics",
    ],
  },
  {
    id: "hop-fidelity",
    date: "2026-09-23",
    title: "Forward and reverse hop fidelity",
    directory: "2026-09-23-return-fixture-fidelity-pilot",
    status: "return_fixture_fidelity_pilot_complete",
    finding:
      "Weight 10 restores the reverse hop to 0.09072 versus 0.09037 target, but worsens the complete input-01 row. No arm qualifies.",
    scope:
      "Three operations · K4 · four weights · 201 exact evaluations per arm · zero samples. Fixing one hop does not restore the full row.",
    gate: "Survival ≥ 95% AND absolute error in each hop ≤ 0.01. The forward target is below 0.01, so this original gate permits a zero forward hop.",
    tableTitles: ["All four final arms", "Local row and optimizer diagnostics"],
  },
  {
    id: "return-fixture",
    date: "2026-09-23",
    title: "Three-operation return fixture",
    directory: "2026-09-23-three-operation-return-fixture",
    status: "return_fixture_study_complete",
    finding:
      "The return edge exposes reverse hops and merging histories. Path and terminal objectives now differ; their backtracking arms reach 93.47% and 93.59% survival, with poor reverse-hop fidelity.",
    scope:
      "Three sites · three operations · six arms · 201 exact evaluations per arm · zero samples. One development fixture, not a 500-operation quality result.",
    gate: "Objective comparison with survival and local fidelity reported separately. No full-program quality or inference-savings gate is passed.",
    tableTitles: [
      "Baseline and all six arms",
      "Target input-row visitation",
      "Complete local-row fidelity",
      "Optimizer diagnostics",
    ],
  },
  {
    id: "objective-steps",
    date: "2026-09-21",
    title: "Objective and step-rule comparison",
    directory: "2026-09-21-fixture-objective-step-comparison",
    status: "fixture_study_complete",
    finding:
      "Path-aware training with backtracking reaches 95.77% survival on two operations, but local hop fidelity remains poor. Objective and step rule both affect results.",
    scope:
      "Three sites · two operations · six named arms · 201 exact evaluations per arm · zero samples. Path and valid-terminal losses coincide structurally here.",
    gate: "Exact development comparison. The 95.77% fixture survival is not a pass of the full M4G quality contract.",
    tableTitles: ["Baseline and all six arms"],
  },
  {
    id: "survival-audit",
    date: "2026-09-21",
    title: "Saved survival-gradient audit",
    directory: "2026-09-21-survival-gradient-audit",
    status: "complete",
    finding:
      "103 of 105 saved updates increase survival and all 21 fits end above initialization. K30 still finishes around 3.38%, far below 95%.",
    scope:
      "21 saved M4G fits · 105 recorded updates · each fit's own training law · zero new fits or samples.",
    gate: "Diagnostic audit of saved updates. It does not prove that longer training would succeed or establish inference savings.",
    tableTitles: ["All 21 saved fits"],
  },
] as const;

function parseTables(
  markdown: string,
  titles: readonly string[],
): ReportTable[] {
  const tables: ReportTable[] = [];
  const groups = markdown
    .split(/\r?\n/)
    .reduce<string[][]>((blocks, line) => {
      if (line.startsWith("|")) {
        if (!blocks.length) blocks.push([]);
        blocks.at(-1)!.push(line);
      } else if (blocks.at(-1)?.length) blocks.push([]);
      return blocks;
    }, [])
    .filter((block) => block.length);
  for (const [index, group] of groups.entries()) {
    const rows = group.map((line) =>
      line
        .slice(1, -1)
        .split("|")
        .map((x) => x.trim()),
    );
    const [columns, separator, ...data] = rows;
    if (
      !separator ||
      !separator.every((x) => /^:?-+:?$/.test(x)) ||
      rows.some((r) => r.length !== columns.length) ||
      !data.length
    )
      throw new Error("Unsupported report table");
    tables.push({ title: titles[index], columns, rows: data });
  }
  if (tables.length !== titles.length) throw new Error("Missing report tables");
  return tables;
}

export async function loadRecentStudies(root: string): Promise<RecentStudy[]> {
  return Promise.all(
    studies.map(async (entry) => {
      const archive = `docs/experiment-reports/${entry.directory}`;
      const source = (label: string, name: string): Source => ({
        label,
        url: `https://github.com/otto-agent007/Thermo/blob/${recentEvidenceCommit}/${archive}/${name}`,
        recordedAt: entry.date,
      });
      const result: RecentStudy = {
        id: entry.id,
        title: entry.title,
        date: entry.date,
        availability: "unavailable",
        finding:
          "Evidence unavailable. Findings and metrics are withheld until the supported reports are authenticated.",
        scope: entry.scope,
        gate: "Unavailable",
        tables: [],
        sources: [
          source("Full study report", "summary.md"),
          source("Recorded completion", "completion.json"),
          source("Independent review", "review.md"),
        ],
      };
      try {
        const files: Record<string, string> = {};
        for (const [name, digest] of Object.entries(
          pins[archive as keyof typeof pins],
        )) {
          const { text } = await readBounded(
            root,
            `${archive}/${name}`,
            100_000,
          );
          if (createHash("sha256").update(text).digest("hex") !== digest)
            throw new Error("Changed report");
          files[name] = text;
        }
        const completion = JSON.parse(files["completion.json"]);
        if (completion.status !== entry.status || !completion.result_digest)
          throw new Error("Unsupported completion");
        result.tables = parseTables(files["summary.md"], entry.tableTitles);
        result.availability = "available";
        result.finding = entry.finding;
        result.gate = entry.gate;
      } catch {
        // Each report fails independently; missing evidence is never a zero or a pass.
      }
      return result;
    }),
  );
}

export const recentReportPaths = Object.entries(pins).flatMap(
  ([directory, files]) =>
    Object.keys(files).map((file) => `${directory}/${file}`),
);
