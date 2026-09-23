import { test } from "node:test";
import assert from "node:assert/strict";
import { cp, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { loadRecentStudies } from "../server/recent";

test("recent reports retain all arms, original gates and pinned sources", async () => {
  const studies = await loadRecentStudies(resolve(".."));
  assert.equal(studies.length, 5);
  assert.ok(studies.every((s) => s.availability === "available"));
  const full = studies[0];
  assert.equal(full.id, "full-row");
  assert.equal(full.tables[0].rows.length, 4);
  assert.equal(full.tables[0].rows[3][1], "0.90660262");
  assert.equal(full.tables[0].rows[3][3], "0.0850716883");
  assert.ok(full.tables[0].rows.every((r) => r.at(-1) === "False"));
  assert.match(full.gate, /0.005/);
  assert.match(studies[1].gate, /0.01/);
  assert.ok(
    full.sources.every((s) =>
      s.url.includes("918f67f62b9478276bc23d9e4a9a805937c58cf4"),
    ),
  );
  assert.equal(studies[4].tables[0].rows.length, 21);
});

test("missing or altered reports cannot supply metrics or completed findings", async () => {
  const directory = await mkdtemp(join(tmpdir(), "thermo-recent-"));
  const archive =
    "docs/experiment-reports/2026-09-23-return-fixture-full-row-pilot";
  try {
    await cp(join(resolve(".."), archive), join(directory, archive), {
      recursive: true,
    });
    let studies = await loadRecentStudies(directory);
    assert.equal(studies[0].availability, "available");
    assert.equal(studies[1].availability, "unavailable");
    await writeFile(
      join(directory, archive, "summary.md"),
      "Altered result: all pass",
    );
    studies = await loadRecentStudies(directory);
    assert.equal(studies[0].availability, "unavailable");
    assert.deepEqual(studies[0].tables, []);
    assert.match(studies[0].finding, /unavailable/i);
    assert.doesNotMatch(studies[0].finding, /No arm qualifies/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
