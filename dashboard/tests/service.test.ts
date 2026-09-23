import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtemp,
  mkdir,
  rm,
  writeFile,
  readFile,
  symlink,
} from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { observeActivity } from "../server/activity";
import { getPayload } from "../server/service";
import { exportSnapshot } from "../server/export";
const root = resolve("..");
import { parseEvidence } from "../server/evidence";
import { archive } from "../server/catalog";
const checkpoint = "results/m4g-task-quality-study/checkpoints";
test("activity observes persisted checkpoints without claiming a live process", async () => {
  const dir = await mkdtemp(join(tmpdir(), "thermo-activity-"));
  try {
    assert.equal((await observeActivity(dir)).checkpointCount, null);
    await mkdir(join(dir, checkpoint), { recursive: true });
    await mkdir(join(dir, archive), { recursive: true });
    await writeFile(
      join(dir, archive, "study.json"),
      await readFile(join(root, archive, "study.json")),
    );
    const study = parseEvidence(
      await readFile(join(root, archive, "study.json"), "utf8"),
    ) as { fits: unknown[] };
    for (let i = 0; i < 3; i++)
      await writeFile(
        join(dir, checkpoint, `fit-0${i}.json`),
        JSON.stringify(study.fits[i]),
      );
    const partial = await observeActivity(dir);
    assert.equal(partial.checkpointCount, 3);
    assert.match(partial.label, /persisted/i);
    assert.doesNotMatch(partial.label, /running/i);
    await writeFile(join(dir, checkpoint, "fit-03.json"), "{");
    assert.equal((await observeActivity(dir)).availability, "stale");
    await rm(join(dir, checkpoint, "fit-03.json"));
    await symlink("/etc/passwd", join(dir, checkpoint, "fit-03.json"));
    assert.ok((await observeActivity(dir)).issues.length);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("read-only routes reject unknown cells, mutations and path traversal", async () => {
  for (const route of [
    "/data/cells/..%2F..%2Fsecret.json",
    "/data/cells/missing.json",
    "/data/other.json",
  ])
    assert.equal((await getPayload(root, route)).status, 404);
  assert.equal(
    (await getPayload(root, "/data/project.json", "POST")).status,
    405,
  );
  assert.equal(
    (await getPayload(root, "/data/project.json", "HEAD")).status,
    200,
  );
  const result = await getPayload(root, "/data/cells/seed-0%2Ffinite%2F1.json");
  assert.equal(result.status, 200);
});
test("static export is dated, portable and contains all individual cell details", async () => {
  const dir = await mkdtemp(join(tmpdir(), "thermo-export-"));
  try {
    await exportSnapshot(root, dir);
    const raw = await readFile(join(dir, "data/project.json"), "utf8");
    const s = JSON.parse(raw);
    assert.equal(s.mode, "snapshot");
    assert.ok(s.generatedAt);
    assert.ok(!raw.includes(root));
    assert.equal(s.study.cells.length, 60);
    assert.equal(
      JSON.parse(
        await readFile(join(dir, "data/cells/seed-0~finite~1.json"), "utf8"),
      ).evaluationSeed,
      "6717865023900054950",
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("unrelated and misassigned checkpoint records cannot count as study progress", async () => {
  const dir = await mkdtemp(join(tmpdir(), "thermo-slots-"));
  try {
    await mkdir(join(dir, checkpoint), { recursive: true });
    await mkdir(join(dir, archive), { recursive: true });
    const raw = await readFile(join(root, archive, "study.json"), "utf8");
    await writeFile(join(dir, archive, "study.json"), raw);
    const s = parseEvidence(raw) as { fits: Record<string, unknown>[] };
    await writeFile(
      join(dir, checkpoint, "fit-00.json"),
      JSON.stringify({
        ...s.fits[0],
        schema_version: "unsupported",
        steps: [null, null, null, null, null],
      }),
    );
    await writeFile(
      join(dir, checkpoint, "fit-01.json"),
      JSON.stringify(s.fits[0]),
    );
    const activity = await observeActivity(dir);
    assert.equal(activity.checkpointCount, 0);
    assert.equal(activity.availability, "stale");
    assert.equal(activity.issues.length, 2);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
