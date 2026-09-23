import { readdir, realpath } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { z } from "zod";
import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { archive } from "./catalog.ts";
import pins from "./archive-pins.json" with { type: "json" };
import { parseEvidence } from "./evidence.ts";
import { readBounded } from "./files.ts";
import type { Activity } from "../shared/model.ts";
const path = "results/m4g-task-quality-study/checkpoints";
const checkpoint = z.object({
  schema_version: z.literal("quality_budget_training_fit.v1"),
  request_hash: z.string().startsWith("sha256:"),
  result_digest: z.string().startsWith("sha256:"),
  steps: z.array(z.unknown()).length(5),
});
export async function observeActivity(repoRoot: string): Promise<Activity> {
  const result: Activity = {
    availability: "unavailable",
    observedAt: new Date().toISOString(),
    latestFileAt: null,
    checkpointCount: null,
    expectedCheckpoints: 21,
    label: "Local activity unavailable",
    issues: [],
  };
  try {
    const base = await realpath(repoRoot);
    const directory = await realpath(resolve(repoRoot, path));
    if (!directory.startsWith(base + sep)) throw Error("Directory escape");
    const files = (await readdir(directory)).filter((f) =>
      /^fit-(0\d|1\d|20)\.json$/.test(f),
    );
    const approved = await readBounded(repoRoot, `${archive}/study.json`);
    if (
      createHash("sha256").update(approved.text).digest("hex") !==
      pins["study.json"]
    )
      throw Error("Checkpoint reference unavailable");
    const expected = z
      .object({ fits: z.array(z.unknown()).length(21) })
      .parse(parseEvidence(approved.text)).fits;
    result.checkpointCount = 0;
    for (const file of files) {
      try {
        const record = await readBounded(
          repoRoot,
          `${path}/${file}`,
          5_000_000,
        );
        const value = parseEvidence(record.text);
        checkpoint.parse(value);
        const slot = Number(file.slice(4, 6));
        if (!isDeepStrictEqual(value, expected[slot]))
          throw Error("Checkpoint does not match expected fit slot");
        result.checkpointCount++;
        if (!result.latestFileAt || record.mtime > result.latestFileAt)
          result.latestFileAt = record.mtime;
      } catch {
        result.issues.push(`${file}: unreadable or incomplete checkpoint`);
      }
    }
    result.availability = result.issues.length ? "stale" : "available";
    result.label = `${result.checkpointCount} persisted checkpoints observed`;
  } catch {
    result.issues.push("Allowlisted local checkpoint directory unavailable");
  }
  return result;
}
