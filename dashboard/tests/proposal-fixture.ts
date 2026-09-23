import { createHash } from "node:crypto";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { join } from "node:path";
export const proposalId = "12345678-1234-4234-8234-123456789abc";
export const digest = (value: string | Buffer) =>
  "sha256:" + createHash("sha256").update(value).digest("hex");
export async function record(dir: string, name: string, value: unknown) {
  const raw = JSON.stringify(value) + "\n";
  await writeFile(join(dir, name + ".json"), raw);
  await writeFile(join(dir, name + ".sha256"), digest(raw).slice(7) + "\n");
}
export async function proposalFixture(root: string) {
  const dir = join(root, "results/harness", proposalId);
  await mkdir(join(dir, "candidate/artifacts"), { recursive: true });
  const patch =
    "diff --git a/dashboard/src/styles.css b/dashboard/src/styles.css\n--- a/dashboard/src/styles.css\n+++ b/dashboard/src/styles.css\n@@ -1 +1 @@\n-old\n+new\n";
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
    "base64",
  );
  await writeFile(join(dir, "patch.diff"), patch);
  await writeFile(join(dir, "candidate/artifacts/overview.png"), png);
  const request = {
    schema_version: 1,
    id: proposalId,
    track: "dashboard",
    objective: "Improve dashboard readability",
    baseline_commit: "a".repeat(40),
    allowed_paths: ["dashboard/src/"],
    plan_digest: "sha256:" + "b".repeat(64),
  };
  const log = "fixture check complete\n";
  const checks = ["dependencies", "unit", "typecheck", "build", "browser"].map(
    (name) => ({
      catalog_version: 1,
      name,
      argv: ["npm", "test"],
      exit_status: 0,
      duration_seconds: 0.1,
      log_path: `checks/${name}.log`,
      log_sha256: digest(log),
      log_tail: log,
      execution: "complete",
      verification: "passed",
    }),
  );
  const baselineRelative = `.plans/${"b".repeat(64)}/baseline`;
  const baselineDir = join(root, "results/harness", baselineRelative);
  await mkdir(join(baselineDir, "checks"), { recursive: true });
  await mkdir(join(dir, "candidate/checks"), { recursive: true });
  for (const check of checks) {
    await writeFile(join(baselineDir, check.log_path), log);
    await writeFile(join(dir, "candidate", check.log_path), log);
  }
  await record(baselineDir, "result", {
    checks,
    artifacts: {},
    plan_digest: request.plan_digest,
    baseline_commit: request.baseline_commit,
    observed_at: "2026-09-23T10:00:00+00:00",
    visual_evidence: "unavailable",
  });
  const result = {
    schema_version: 1,
    execution: "complete",
    verification: "passed",
    research_outcome: "not_applicable",
    recommendation:
      "Increase contrast and spacing. Owner visual review required.",
    patch_digest: digest(patch),
    artifacts: {
      "patch.diff": digest(patch),
      "candidate/artifacts/overview.png": digest(png),
    },
    checks,
    check_record_dir: "candidate",
    baseline_checks: checks.map(({ name, execution, verification }) => ({
      name,
      execution,
      verification,
    })),
    baseline_record: `${baselineRelative}/result.json`,
    baseline_record_digest: digest(
      await readFile(join(baselineDir, "result.json")),
    ),
    source_state: { candidate_worktree: "/private/workspace" },
    diagnostic: "secret diagnostic",
    log_tail: "secret token",
  };
  await record(dir, "request", request);
  await record(dir, "result", result);
  return { dir, request, result, patch, png };
}
