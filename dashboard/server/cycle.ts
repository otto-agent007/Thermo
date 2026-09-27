import { z } from "zod";
import type { CyclePhase, ResearchCycle } from "../shared/model.ts";
import { readBounded } from "./files.ts";

export type CycleOptions = { cycleRoot?: string | null; observedAt?: string };
const digest = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const statusSchema = z
  .strictObject({
    schema_version: z.literal(1),
    job_id: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/),
    source_sha: z.string().regex(/^[0-9a-f]{40}$/),
    request_digest: digest,
    phase: z.enum([
      "queued",
      "running",
      "verifying",
      "awaiting_review",
      "recorded",
      "blocked",
      "failed",
    ]),
    attempts: z.number().int().min(0).max(2),
    elapsed_seconds: z.number().min(0).max(Number.MAX_SAFE_INTEGER),
    heartbeat_at: z.iso.datetime({ offset: true }).nullable(),
    evidence_digest: digest.nullable(),
    review_status: z.enum(["pending", "changes_requested", "approved"]),
    message: z.string().max(2048),
  })
  .refine(
    (s) =>
      (s.phase !== "recorded" || s.review_status === "approved") &&
      (!["awaiting_review", "recorded"].includes(s.phase) ||
        s.evidence_digest !== null),
  );
const reviewSchema = z.strictObject({
  schema_version: z.literal(1),
  job_id: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/),
  source_sha: z.string().regex(/^[0-9a-f]{40}$/),
  request_digest: digest,
  evidence_digest: digest,
  evidence_commit: z.string().regex(/^[0-9a-f]{40}$/),
  reviewed_at: z.iso.datetime({ offset: true }),
  reviewer: z.string().max(160).trim().min(1),
  replay: z.enum(["executed", "inspected"]),
  decision: z.enum(["changes_requested", "approved"]),
  scope: z.string().max(2048).trim().min(1),
  findings: z.array(z.string().max(1024).trim().min(1)).max(20),
});

async function observeReview(
  root: string,
  status: z.infer<typeof statusSchema>,
): Promise<ResearchCycle["independentReview"]> {
  if (status.evidence_digest === null) return null;
  try {
    const { text } = await readBounded(
      root,
      `reviews/${status.evidence_digest.slice(7)}.json`,
      32 * 1024,
    );
    const review = reviewSchema.parse(JSON.parse(text));
    if (
      review.job_id !== status.job_id ||
      review.source_sha !== status.source_sha ||
      review.request_digest !== status.request_digest ||
      review.evidence_digest !== status.evidence_digest
    )
      throw new Error("Review does not bind current evidence");
    // Identity is a public label, never arbitrary review prose or log content.
    const safeIdentity =
      /^[\p{L}\p{N} @._()+:,-]+$/u.test(review.reviewer) &&
      !/\b(?:bearer|password|secret|token|api[_-]?key)\b/i.test(
        review.reviewer,
      ) &&
      !/\b(?:gh[pousr]_|github_pat_|sk-|xox[baprs]-|AKIA)[A-Za-z0-9_-]+/.test(
        review.reviewer,
      ) &&
      !/[A-Za-z0-9_-]{32,}/.test(review.reviewer);
    return {
      availability: "available",
      notice: null,
      record: {
        reviewer: safeIdentity ? review.reviewer : "Identity withheld",
        reviewedAt: review.reviewed_at,
        evidenceCommit: review.evidence_commit,
        replay: review.replay,
        decision: review.decision,
      },
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    return {
      availability: "unavailable",
      notice:
        "Independent review record unavailable. Ask the operator to check its schema and binding to the current job, source, request and evidence.",
      record: null,
    };
  }
}
const messages: Record<CyclePhase, string> = {
  queued:
    "Approved request queued. No execution is inferred from this status file.",
  running:
    "The last status reports an execution attempt. Check the worker journal before taking action.",
  verifying:
    "Verification is in progress or a second evidence copy still needs verification. Inspect the worker journal for the next step.",
  awaiting_review:
    "The worker reports evidence awaiting review. Any independent review record is observed separately below.",
  recorded:
    "The worker status reports resolved review. Consult the accepted report for the scientific finding.",
  blocked:
    "Operator action required: inspect the worker journal, resolve the recorded blocker, then refresh this observation.",
  failed:
    "Operator action required: inspect the failed attempt and remaining approved budget before considering a retry.",
};

// This is a sanitized observation of an explicit status file, not evidence replay
// or a worker-liveness check. Free-form worker messages never leave the server.
export async function observeCycle(
  options: CycleOptions = {},
): Promise<ResearchCycle | null> {
  const root =
    options.cycleRoot === undefined
      ? process.env.THERMO_CYCLE_ROOT
      : options.cycleRoot;
  if (!root) return null;
  const observedAt = options.observedAt ?? new Date().toISOString();
  try {
    const record = await readBounded(root, "status.json", 8192);
    const raw = statusSchema.parse(JSON.parse(record.text));
    const age =
      raw.heartbeat_at === null
        ? null
        : (Date.parse(observedAt) - Date.parse(raw.heartbeat_at)) / 1000;
    const heartbeatAgeSeconds =
      age === null || age < 0 ? null : Math.floor(age);
    const active = raw.phase === "running" || raw.phase === "verifying";
    const stale = active && (age === null || age < 0 || age > 120);
    return {
      availability: stale ? "stale" : "available",
      observedAt,
      heartbeatAgeSeconds,
      independentReview: await observeReview(root, raw),
      notice: stale
        ? "Heartbeat is stale, missing or ahead of the observation clock. This does not establish whether the process is running; check lock ownership before retrying."
        : null,
      status: {
        jobId: raw.job_id,
        sourceSha: raw.source_sha,
        requestDigest: raw.request_digest,
        phase: raw.phase,
        attempts: raw.attempts,
        elapsedSeconds: raw.elapsed_seconds,
        heartbeatAt: raw.heartbeat_at,
        evidenceDigest: raw.evidence_digest,
        reviewStatus: raw.review_status,
        message: messages[raw.phase],
      },
    };
  } catch {
    return {
      availability: "unavailable",
      observedAt,
      heartbeatAgeSeconds: null,
      status: null,
      independentReview: null,
      notice:
        "Cycle status unavailable. Ask the operator to check the configured status file and schema, then refresh.",
    };
  }
}
