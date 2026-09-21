import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, cp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { loadEvidence, parseEvidence } from "../server/evidence";
const root = resolve("..");
const archive =
  "docs/experiment-reports/2026-09-20-task-quality-inference-budget";
test("maps complete execution independently from scientific failure and preserves large identifiers", async () => {
  const { snapshot, details } = await loadEvidence(root);
  assert.deepEqual(snapshot.study?.counts, {
    fits: 21,
    updates: 105,
    cells: 60,
    pairs: 18,
  });
  assert.equal(snapshot.execution.state, "complete");
  assert.equal(snapshot.science.state, "quality_failure");
  assert.equal(
    snapshot.study?.cells.filter((c) => c.decision === "fail").length,
    60,
  );
  const cell = details.get("seed-0/finite/1")!;
  assert.equal(cell.evaluationSeed, "6717865023900054950");
  assert.equal(cell.leakage, 27280 / 32768);
  assert.equal(cell.survivalTrace.length, 500);
  assert.equal(
    cell.histogram.reduce((a, b) => a + b, 0),
    32768,
  );
  assert.equal(
    snapshot.study?.costs.find((c) => c.label === "Finite training grid")
      ?.value,
    4423680000,
  );
  assert.equal(
    snapshot.study?.costs.find((c) => c.label === "Oracle modeled sweeps")
      ?.value,
    null,
  );
  assert.ok(JSON.stringify(snapshot).length < 150000);
});
test("parser keeps integer identities exact while retaining metric numbers", () => {
  assert.deepEqual(
    parseEvidence('{"seed":6717865023900054950,"metric":0.25}'),
    { seed: "6717865023900054950", metric: 0.25 },
  );
});
for (const [name, mutate] of [
  ["missing", async (p: string) => rm(join(p, "study.json"))],
  ["truncated", async (p: string) => writeFile(join(p, "study.json"), "{")],
  [
    "unsupported",
    async (p: string) => {
      const f = join(p, "study.json");
      await writeFile(
        f,
        (await readFile(f, "utf8")).replace(
          "quality_budget_complete_study.v1",
          "unknown.v9",
        ),
      );
    },
  ],
  [
    "stale completion",
    async (p: string) => {
      const f = join(p, "completion.json");
      await writeFile(
        f,
        (await readFile(f, "utf8")).replace("a9e7bf3e", "00000000"),
      );
    },
  ],
  ["missing review", async (p: string) => rm(join(p, "evidence-review.json"))],
] as const)
  test(name + " evidence cannot claim release completion", async () => {
    const temp = await mkdtemp(join(tmpdir(), "thermo-evidence-"));
    try {
      await cp(join(root, archive), join(temp, archive), { recursive: true });
      await mutate(join(temp, archive));
      const { snapshot } = await loadEvidence(temp);
      assert.notEqual(snapshot.execution.state, "complete");
      assert.notEqual(snapshot.availability, "available");
      assert.ok(snapshot.issues.length);
    } finally {
      await rm(temp, { recursive: true, force: true });
    }
  });
