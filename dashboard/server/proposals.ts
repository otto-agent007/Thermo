import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir, realpath } from "node:fs/promises";
import { join, sep } from "node:path";
import { z } from "zod";
import { readBounded } from "./files.ts";
import type { ProposalSummary } from "../shared/model.ts";
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const hash = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const execution = z.enum(["complete", "failed", "timed_out", "unavailable"]);
const verification = z.enum(["passed", "failed", "inconclusive"]);
const check = z.object({
  name: z.string().min(1).max(100),
  execution,
  verification,
});
// Mirrors CheckResult v1 in improvement_harness/checks.py; logs never enter the response.
const fullCheck = check
  .extend({
    catalog_version: z.literal(1),
    argv: z.array(z.string()).min(1).max(20),
    exit_status: z.number().int().nullable(),
    duration_seconds: z.number().min(0),
    log_sha256: hash,
    log_tail: z.string().refine((value) => Array.from(value).length <= 4096),
    log_path: z.string().regex(/^checks\/[a-zA-Z0-9_.-]+\.log$/),
  })
  .strict()
  .refine((item) => {
    if (item.execution === "complete")
      return item.exit_status === 0 && item.verification === "passed";
    if (item.execution === "failed")
      return (
        item.exit_status !== null &&
        item.exit_status !== 0 &&
        item.verification === "failed"
      );
    return item.verification === "inconclusive";
  }, "Incoherent check status");
type Check = z.infer<typeof fullCheck>;
// Fixed catalog v1. Update alongside the Python catalog and contract tests.
const catalogs: Record<"dashboard" | "research", [string, string[]][]> = {
  dashboard: [
    ["dependencies", ["npm", "ci"]],
    ["unit", ["npm", "test"]],
    ["typecheck", ["npm", "run", "typecheck"]],
    ["build", ["npm", "run", "build"]],
    ["browser", ["npm", "run", "test:browser"]],
  ],
  research: [
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
    ["ruff", ["uv", "run", "ruff", "check", "src/thermo_lab", "tests/unit"]],
  ],
};
const screenshotNames = ["overview", "mobile", "experiments"];
const matchesCatalog = (track: "dashboard" | "research", checks: Check[]) =>
  JSON.stringify(checks.map((item) => [item.name, item.argv])) ===
  JSON.stringify(catalogs[track]);
const checkSummary = ({
  name,
  execution,
  verification,
}: z.infer<typeof check>) => ({ name, execution, verification });
const requestSchema = z
  .object({
    schema_version: z.literal(1),
    id: z.string().regex(uuid),
    parent_id: z.string().regex(uuid).nullable().optional(),
    track: z.enum(["dashboard", "research"]),
    objective: z.string().min(1).max(8000),
    baseline_commit: z.string().regex(/^[0-9a-f]{40}$/),
    plan_digest: hash,
    allowed_paths: z.array(z.string().min(1)).min(1).max(20),
  })
  .strict();
const resultSchema = z.object({
  schema_version: z.literal(1),
  execution,
  verification,
  research_outcome: z
    .enum(["improved", "regressed", "inconclusive", "not_applicable"])
    .optional(),
  recommendation: z.string().max(100_000).optional(),
  patch_digest: hash.optional(),
  artifacts: z.record(z.string(), hash).default({}),
  checks: z.array(fullCheck).max(10).default([]),
  baseline_checks: z.array(check).max(10).default([]),
  check_record_dir: z.enum(["candidate"]).optional(),
  baseline_record: z.string().optional(),
  baseline_record_digest: hash.optional(),
  visual_evidence: z.enum(["available", "unavailable"]).optional(),
  baseline_visual_evidence: z.enum(["available", "unavailable"]).optional(),
  science: z
    .object({
      execution: z.literal("complete"),
      verification: z.literal("passed"),
      research_outcome: z.enum(["improved", "regressed", "inconclusive"]),
      primary_metric: z.literal("exact_objective_delta"),
      parameters: z.array(z.number().min(-2).max(2)).length(9),
      delta: z.number(),
      before: z.number().min(0).max(3),
      after: z.number().min(0).max(3),
      evidence: z.literal("exact_reference"),
      baseline_commit: z.string().regex(/^[0-9a-f]{40}$/),
      scope: z.literal("three-site fixture only"),
    })
    .optional(),
});
const digest = (bytes: string | Buffer) =>
  "sha256:" + createHash("sha256").update(bytes).digest("hex");
const safeText = (value: string) =>
  value
    .replace(/(?:[A-Za-z]:[\\/]|\\\\)[^\s<>"']*/g, "[local path]")
    .replace(/(^|[^\w])\/[^\s<>"'\x60)\]}]*/g, "$1[local path]")
    .slice(0, 4000);
async function safePath(root: string, relative: string) {
  if (
    !relative ||
    relative.split("/").some((p) => !p || p === "." || p === "..") ||
    relative.includes("\\")
  )
    throw Error("Invalid path");
  const base = await realpath(root);
  let current = base;
  for (const part of relative.split("/")) {
    current = join(current, part);
    if ((await lstat(current)).isSymbolicLink()) throw Error("Symlink");
  }
  if (!(await realpath(current)).startsWith(base + sep))
    throw Error("Outside root");
  return current;
}
async function bytes(root: string, relative: string, max = 1_000_000) {
  const target = await safePath(root, relative);
  const file = await open(target, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const before = await file.stat();
    if (!before.isFile() || before.size > max) throw Error("Oversized file");
    const buffer = Buffer.alloc(before.size + 1);
    const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
    const after = await file.stat();
    if (
      bytesRead !== before.size ||
      before.size !== after.size ||
      before.mtimeMs !== after.mtimeMs
    )
      throw Error("Changed file");
    return buffer.subarray(0, bytesRead);
  } finally {
    await file.close();
  }
}
async function json(root: string, path: string) {
  await safePath(root, path);
  const { text } = await readBounded(root, path, 1_000_000);
  if (Buffer.byteLength(text) > 1_000_000) throw Error("Oversized JSON");
  return { value: JSON.parse(text), raw: text };
}
async function record(root: string, directory: string, name: string) {
  const payload = await json(root, `${directory}/${name}.json`);
  const sidecar = (await bytes(root, `${directory}/${name}.sha256`, 66))
    .toString()
    .trim();
  if (
    !/^[0-9a-f]{64}$/.test(sidecar) ||
    digest(payload.raw) !== `sha256:${sidecar}`
  )
    throw Error("Changed record");
  return payload;
}
async function names(root: string, path: string, max: number) {
  const directory = await opendir(await safePath(root, path));
  const found: string[] = [];
  for await (const entry of directory) {
    if (found.length >= max) throw Error("Too many entries");
    found.push(entry.name);
  }
  return found.sort();
}
async function authenticated(
  root: string,
  path: string,
  expected: string,
  max?: number,
) {
  const value = await bytes(root, path, max);
  if (digest(value) !== expected) throw Error("Changed artifact");
  return value;
}
async function load(root: string, id: string) {
  if (!uuid.test(id)) throw Error("Invalid ID");
  const dir = `results/harness/${id}`;
  const request = requestSchema.parse(
    (await record(root, dir, "request")).value,
  );
  if (
    request.id !== id ||
    request.allowed_paths.some(
      (p) =>
        p.startsWith("/") || p.split("/").includes("..") || p.includes("\\"),
    )
  )
    throw Error("Invalid request");
  const observed = await record(root, dir, "result");
  const result = resultSchema.parse(observed.value);
  const artifacts = new Map<string, Buffer>();
  const entries = Object.entries(result.artifacts);
  if (entries.length > 8) throw Error("Too many artifacts");
  for (const [name, expected] of entries) {
    if (
      !/^(plan\.json|recommendation\.md|patch\.diff|candidate\/artifacts\/(overview|mobile|experiments)\.png)$/.test(
        name,
      )
    )
      throw Error("Unsupported artifact");
    const value = await authenticated(
      root,
      `${dir}/${name}`,
      expected,
      name.endsWith(".png") ? 8_000_000 : 1_000_000,
    );
    if (
      name.endsWith(".png") &&
      !value
        .subarray(0, 8)
        .equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    )
      throw Error("Invalid PNG");
    artifacts.set(name, value);
  }
  let patch: Buffer | undefined;
  if (result.patch_digest)
    patch = await authenticated(root, `${dir}/patch.diff`, result.patch_digest);
  for (const item of result.checks) {
    if (!result.check_record_dir) throw Error("Missing check directory");
    await authenticated(
      root,
      `${dir}/${result.check_record_dir}/${item.log_path}`,
      item.log_sha256,
      65_536,
    );
  }
  let baselineChecks: Check[] = [];
  let baselineVisual = false;
  if (result.baseline_record) {
    const baselineDir = `results/harness/.plans/${request.plan_digest.slice(7)}/baseline`;
    if (
      result.baseline_record !==
      `.plans/${request.plan_digest.slice(7)}/baseline/result.json`
    )
      throw Error("Invalid baseline path");
    const baseline = await record(root, baselineDir, "result");
    if (
      digest(baseline.raw) !== result.baseline_record_digest ||
      baseline.value.plan_digest !== request.plan_digest ||
      baseline.value.baseline_commit !== request.baseline_commit
    )
      throw Error("Changed baseline");
    const checks = z.array(fullCheck).max(10).parse(baseline.value.checks);
    baselineChecks = checks;
    if (
      JSON.stringify(checks.map(checkSummary)) !==
      JSON.stringify(result.baseline_checks.map(checkSummary))
    )
      throw Error("Baseline check summaries disagree");
    for (const item of checks)
      await authenticated(
        root,
        `${baselineDir}/${item.log_path}`,
        item.log_sha256,
        65_536,
      );
    const baselineArtifacts = z
      .record(
        z.string().regex(/^artifacts\/(overview|mobile|experiments)\.png$/),
        hash,
      )
      .parse(baseline.value.artifacts ?? {});
    for (const [name, expected] of Object.entries(baselineArtifacts))
      await authenticated(root, `${baselineDir}/${name}`, expected, 8_000_000);
    baselineVisual =
      baseline.value.visual_evidence === "available" &&
      screenshotNames.every(
        (name) => baselineArtifacts[`artifacts/${name}.png`],
      );
  }
  let review: ProposalSummary["review"] = "proposed";
  for (const name of (await names(root, dir, 150)).filter((n) =>
    n.startsWith("review-"),
  )) {
    if (!/^review-\d{4}\.json$/.test(name)) throw Error("Invalid review file");
    const entry = z
      .object({
        schema_version: z.literal(1),
        candidate_id: z.literal(id),
        decision: z.enum(["proposed", "accepted", "rejected"]),
        note: z.string().min(1),
        observed_at: z.iso.datetime({ offset: true }),
      })
      .strict()
      .parse((await json(root, `${dir}/${name}`)).value);
    review = entry.decision;
  }
  const prefix = `/data/proposals/${id}`;
  const checksSummary = (items: z.infer<typeof check>[]) =>
    items.length
      ? `${items.filter((c) => c.verification === "passed").length} of ${items.length} checks passed`
      : "Unavailable";
  if (
    result.science &&
    (request.track !== "research" ||
      result.science.baseline_commit !== request.baseline_commit ||
      Math.abs(
        result.science.delta - (result.science.after - result.science.before),
      ) > 1e-12)
  )
    throw Error("Invalid science source");
  const exact = request.track === "research" ? result.science : undefined;
  const completeCatalogs =
    matchesCatalog(request.track, baselineChecks) &&
    matchesCatalog(request.track, result.checks);
  const completeArtifacts =
    Boolean(patch) &&
    ["plan.json", "recommendation.md", "patch.diff"].every((name) =>
      artifacts.has(name),
    );
  const trackEvidence =
    request.track === "dashboard"
      ? baselineVisual &&
        result.baseline_visual_evidence === "available" &&
        result.visual_evidence === "available" &&
        screenshotNames.every((name) =>
          artifacts.has(`candidate/artifacts/${name}.png`),
        )
      : Boolean(exact && exact.research_outcome === result.research_outcome);
  const checksPass = [...baselineChecks, ...result.checks].every(
    (item) => item.verification === "passed",
  );
  const canPass =
    result.execution === "complete" &&
    completeCatalogs &&
    completeArtifacts &&
    trackEvidence &&
    checksPass;
  // An unfinished observation remains reviewable; absent evidence never certifies success.
  const displayedVerification =
    result.verification === "passed" && !canPass
      ? "inconclusive"
      : result.verification;
  const summary: ProposalSummary = {
    id,
    track: request.track,
    objective: safeText(request.objective),
    baseline: request.baseline_commit,
    execution: result.execution,
    verification: displayedVerification,
    researchOutcome:
      request.track === "research" && !exact
        ? "inconclusive"
        : (result.research_outcome ?? "inconclusive"),
    review,
    recommendation: safeText(
      result.recommendation ?? "Recommendation unavailable.",
    ),
    baselineSummary: exact
      ? `Exact objective: ${exact.before}`
      : checksSummary(baselineChecks),
    candidateSummary: exact
      ? `Exact objective: ${exact.after}`
      : checksSummary(result.checks),
    patchUrl: patch ? prefix + "/patch.diff" : null,
    reportUrl: prefix + "/report.md",
    screenshotUrls: ["overview", "mobile", "experiments"]
      .filter((name) => artifacts.has(`candidate/artifacts/${name}.png`))
      .map((name) => `${prefix}/artifacts/${name}.png`),
  };
  // Derive the public report only from authenticated, validated fields. Never read report.md.
  const statusLines = (checks: Check[]) =>
    checks.length
      ? checks.map(
          (item) =>
            `- ${safeText(item.name).replace(/[\[\]<>*_`]/g, "")}: ${item.execution} / ${item.verification}`,
        )
      : ["Unavailable"];
  const report = [
    `# Proposal ${id}`,
    "",
    `Track: ${request.track}`,
    `Baseline commit: [${request.baseline_commit}](https://github.com/otto-agent007/Thermo/commit/${request.baseline_commit})`,
    `Plan digest: ${request.plan_digest}`,
    `Patch digest: ${result.patch_digest ?? "unavailable"}`,
    `Observation digest: ${digest(observed.raw)}`,
    `Execution: ${summary.execution}`,
    `Verification: ${summary.verification}`,
    `Research outcome: ${summary.researchOutcome}`,
    `Owner review: ${review}`,
    "",
    "## Baseline checks",
    ...statusLines(baselineChecks),
    "",
    "## Candidate checks",
    ...statusLines(result.checks),
    "",
    ...(exact
      ? [
          "## Exact source observation",
          `Before: ${exact.before}`,
          `After: ${exact.after}`,
          `Delta: ${exact.delta}`,
          `Evidence: ${exact.evidence}`,
          `Scope: ${exact.scope}`,
          "Bounded exact trial; no full-study or hardware claim.",
        ]
      : ["Research observation: unavailable or not applicable."]),
    "",
    "Derived read-only report. Logs, diagnostics, local paths and review notes are omitted.",
    "",
  ].join("\n");
  if (Buffer.byteLength(report) > 32_768) throw Error("Report too large");
  return { summary, patch, artifacts, report };
}
export async function readProposals(
  repoRoot: string,
): Promise<{ items: ProposalSummary[]; issues: string[] }> {
  const items: ProposalSummary[] = [],
    issues: string[] = [];
  let entries: string[];
  try {
    entries = await names(repoRoot, "results/harness", 250);
  } catch (error) {
    return {
      items,
      issues:
        (error as NodeJS.ErrnoException).code === "ENOENT"
          ? []
          : ["Local proposal records unavailable."],
    };
  }
  const ids = entries.filter((name) => uuid.test(name));
  if (ids.length > 100)
    return {
      items,
      issues: ["Too many local proposals; display limit is 100."],
    };
  for (const id of ids) {
    try {
      items.push((await load(repoRoot, id)).summary);
    } catch {
      issues.push(`Proposal ${id}: evidence unavailable or invalid.`);
    }
  }
  return { items, issues };
}
export async function readProposalArtifact(
  root: string,
  id: string,
  name: string,
) {
  try {
    const value = await load(root, id);
    const body =
      name === "report.md"
        ? value.report
        : name === "patch.diff"
          ? value.patch?.toString("utf8")
          : value.artifacts.get(`candidate/artifacts/${name}`);
    if (body !== undefined)
      return {
        status: 200,
        body,
        contentType:
          name === "report.md"
            ? "text/markdown; charset=utf-8"
            : name === "patch.diff"
              ? "text/x-diff; charset=utf-8"
              : "image/png",
      };
  } catch {
    /* Fail closed with no filesystem diagnostics. */
  }
  return { status: 404, body: { error: "Proposal artifact unavailable" } };
}
