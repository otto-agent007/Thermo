import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  loadKnowledgeResearch,
  parseBacklog,
  parseLessons,
} from "../server/knowledge";

const lessons = `# Lessons

Intro prose with a | pipe that is not a table.

| Date | Study | Lesson | Evidence class | Source |
| --- | --- | --- | --- | --- |
| 2026-09-25 | M4H, M4I | Survival reached 89.61%, short of 95%. Closed as a negative result. | \`exact_reference\` | [M4I summary](../experiment-reports/2026-09-25-kernel-capacity-screen/summary.md), [CLAUDE.md](../../CLAUDE.md) |
| 2026-09-12 | M4B | No demonstrated finite-K4 advantage. | \`software_simulation\`, three seeds | [M4B summary](../experiment-reports/2026-09-12-matched-training-budget/summary.md) |

Trailing prose.
`;

const backlog = `# Experiment backlog

| # | Experiment | Source | Exact anchor | Cost | Fit |
| --- | --- | --- | --- | --- | --- |
| E0 | THRML finite-sweep contract | [THRML](sources/extropic-thrml.md) | Exact 32-state sweep matrix | Minutes, CPU | Checks a dependency already in use |
| E1 | Langevin trajectory-gradient contract | [Whitelam](sources/whitelam-gradient-descent.md) | Closed-form gradient | Seconds, CPU | M3's continuous analogue |

| Convention | Thermo side | THRML side | How a mismatch shows |
| --- | --- | --- | --- |
| Energy sign | E(s) | IsingEBM | control |
`;

test("lessons rows become findings with verbatim text and resolved links", () => {
  const items = parseLessons(lessons);
  assert.equal(items.length, 2);
  assert.equal(items[0].kind, "finding");
  assert.equal(items[0].title, "M4H, M4I (2026-09-25)");
  assert.equal(
    items[0].text,
    "Survival reached 89.61%, short of 95%. Closed as a negative result.",
  );
  assert.match(items[0].scope, /exact_reference/);
  assert.equal(
    items[1].scope.includes("software_simulation, three seeds"),
    true,
  );
  assert.deepEqual(
    items[0].sources.map((s) => s.url),
    [
      "https://github.com/otto-agent007/Thermo/blob/main/docs/experiment-reports/2026-09-25-kernel-capacity-screen/summary.md",
      "https://github.com/otto-agent007/Thermo/blob/main/CLAUDE.md",
    ],
  );
  assert.equal(items[0].sources[0].recordedAt, "2026-09-25");
});

test("backlog ranking rows become proposals and later tables are ignored", () => {
  const items = parseBacklog(backlog);
  assert.deepEqual(
    items.map((i) => i.id),
    ["backlog-E0", "backlog-E1"],
  );
  assert.equal(items[0].kind, "proposal");
  assert.equal(items[0].title, "E0: THRML finite-sweep contract");
  assert.match(items[0].text, /Exact anchor: Exact 32-state sweep matrix\./);
  assert.match(items[0].scope, /Not scheduled, not frozen, not evidence/);
  assert.equal(
    items[0].sources[0].url,
    "https://github.com/otto-agent007/Thermo/blob/main/docs/knowledge/sources/extropic-thrml.md",
  );
});

test("a missing knowledge file yields an issue, not an exception", async () => {
  const dir = await mkdtemp(join(tmpdir(), "thermo-knowledge-"));
  try {
    await mkdir(join(dir, "docs/knowledge"), { recursive: true });
    await writeFile(join(dir, "docs/knowledge/experiment-backlog.md"), backlog);
    const { items, issues } = await loadKnowledgeResearch(dir);
    assert.equal(items.length, 2);
    assert.equal(issues.length, 1);
    assert.match(issues[0], /^docs\/knowledge\/lessons\.md: /);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("the repository backlog parses to the ranked E-items", async () => {
  const { items } = await loadKnowledgeResearch("..");
  const ids = items.filter((i) => i.kind === "proposal").map((i) => i.id);
  assert.deepEqual(ids, [
    "backlog-E0",
    "backlog-E1",
    "backlog-E2",
    "backlog-E3",
    "backlog-E4",
    "backlog-E5",
  ]);
});
