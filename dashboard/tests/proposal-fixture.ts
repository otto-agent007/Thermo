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
export async function proposalFixture(
  root: string,
  track: "dashboard" | "research" = "dashboard",
) {
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
    track,
    objective: "Improve dashboard readability",
    baseline_commit: "a".repeat(40),
    allowed_paths:
      track === "dashboard"
        ? ["dashboard/src/"]
        : ["src/thermo_lab/research_candidates/three_site.py"],
    plan_digest: "sha256:" + "b".repeat(64),
  };
  const log = "fixture check complete\n";
  const commands: [string, string[]][] =
    track === "dashboard"
      ? [
          ["dependencies", ["npm", "ci"]],
          ["unit", ["npm", "test"]],
          ["typecheck", ["npm", "run", "typecheck"]],
          ["build", ["npm", "run", "build"]],
          ["browser", ["npm", "run", "test:browser"]],
        ]
      : [
          ["dependencies", ["uv", "sync", "--frozen"]],
          [
            "exact_fixture",
            [
              "uv",
              "run",
              "pytest",
              "tests/unit/test_trajectory_reinforce_refinement.py::test_exact_objective_evaluation_ties_updated_parameters_across_both_occurrences",
              "-q",
            ],
          ],
          [
            "focused_tests",
            [
              "uv",
              "run",
              "pytest",
              "tests/unit/test_trajectory_reinforce_refinement.py",
              "-q",
            ],
          ],
          [
            "ruff",
            ["uv", "run", "ruff", "check", "src/thermo_lab", "tests/unit"],
          ],
        ];
  const checks = commands.map(([name, argv]) => ({
    catalog_version: 1,
    name,
    argv: [...argv],
    exit_status: 0,
    duration_seconds: 0.1,
    log_path: `checks/${name}.log`,
    log_sha256: digest(log),
    log_tail: log,
    execution: "complete",
    verification: "passed",
  }));
  const baselineRelative = `.plans/${"b".repeat(64)}/baseline`;
  const baselineDir = join(root, "results/harness", baselineRelative);
  await mkdir(join(baselineDir, "checks"), { recursive: true });
  await mkdir(join(dir, "candidate/checks"), { recursive: true });
  for (const check of checks) {
    await writeFile(join(baselineDir, check.log_path), log);
    await writeFile(join(dir, "candidate", check.log_path), log);
  }
  const baselineArtifacts: Record<string, string> = {};
  const candidateArtifacts: Record<string, string> = {
    "patch.diff": digest(patch),
  };
  await mkdir(join(baselineDir, "artifacts"));
  for (const name of ["overview", "mobile", "experiments"]) {
    await writeFile(join(baselineDir, "artifacts", `${name}.png`), png);
    await writeFile(join(dir, "candidate/artifacts", `${name}.png`), png);
    baselineArtifacts[`artifacts/${name}.png`] = digest(png);
    candidateArtifacts[`candidate/artifacts/${name}.png`] = digest(png);
  }
  for (const [name, content] of [
    ["plan.json", JSON.stringify(request)],
    [
      "recommendation.md",
      "Increase contrast and spacing. Owner visual review required.",
    ],
  ]) {
    await writeFile(join(dir, name), content);
    candidateArtifacts[name] = digest(content);
  }
  await record(baselineDir, "result", {
    checks,
    artifacts: baselineArtifacts,
    plan_digest: request.plan_digest,
    baseline_commit: request.baseline_commit,
    observed_at: "2026-09-23T10:00:00+00:00",
    visual_evidence: "available",
  });
  const result = {
    schema_version: 1,
    execution: "complete",
    verification: "passed",
    research_outcome: track === "research" ? "improved" : "not_applicable",
    science:
      track === "research"
        ? {
            before: 0.25,
            after: 0.2,
            delta: 0.2 - 0.25,
            evidence: "exact_reference",
            scope: "three-site fixture only",
            baseline_commit: request.baseline_commit,
            execution: "complete",
            verification: "passed",
            research_outcome: "improved",
            primary_metric: "exact_objective_delta",
            parameters: Array(9).fill(0),
          }
        : undefined,
    recommendation:
      "Increase contrast and spacing. Owner visual review required.",
    patch_digest: digest(patch),
    artifacts: candidateArtifacts,
    visual_evidence: "available",
    baseline_visual_evidence: "available",
    visual_review: "required",
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
