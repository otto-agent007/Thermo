import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { loadEvidence } from "../server/evidence";
import { filterCells, toCsv } from "../src/features/filters";
test("filters and CSV share exact rows with unpooled seeds and distinct oracle horizons", async () => {
  const cells = (await loadEvidence(resolve(".."))).snapshot.study!.cells;
  const rows = filterCells(cells, {
    seeds: [0],
    members: ["finite"],
    horizons: [30],
    query: "",
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].id, "seed-0/finite/30");
  assert.match(toCsv(rows), /seed,member,horizon/);
  assert.equal(toCsv(rows).trim().split("\n").length, 2);
  assert.equal(
    filterCells(cells, { seeds: [], members: [], horizons: [], query: "" })
      .length,
    60,
  );
  assert.equal(
    filterCells(cells, {
      seeds: [],
      members: [],
      horizons: ["equilibrium"],
      query: "",
    }).length,
    6,
  );
  assert.equal(
    filterCells(cells, {
      seeds: [],
      members: [],
      horizons: [],
      query: "absent",
    }).length,
    0,
  );
  assert.match(toCsv([{ ...rows[0], id: "=SUM(1,2)" }]), /"'=SUM\(1,2\)"/);
});
