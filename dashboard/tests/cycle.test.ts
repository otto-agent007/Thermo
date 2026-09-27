import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { getPayload } from "../server/service";
import { exportSnapshot } from "../server/export";
import { Research } from "../src/features/Research";
import type { ProjectSnapshot } from "../shared/model";

const observedAt = "2026-09-26T12:00:00.000Z";
const status = {
  schema_version: 1,
  job_id: "m5a-first-cycle",
  source_sha: "a".repeat(40),
  request_digest: `sha256:${"b".repeat(64)}`,
  phase: "awaiting_review",
  attempts: 1,
  elapsed_seconds: 123.5,
  heartbeat_at: "2026-09-26T11:59:30+00:00",
  evidence_digest: `sha256:${"c".repeat(64)}`,
  review_status: "pending",
  message: "Negative scientific result; token=SECRET /private/worker/log.txt",
};

async function fixture(run: (dir: string) => Promise<void>) {
  const dir = await mkdtemp(join(tmpdir(), "thermo-cycle-"));
  try {
    await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
async function snapshot(cycleRoot: string | null) {
  return (
    await getPayload("/nonexistent", "/data/project.json", "GET", {
      cycleRoot,
      observedAt,
    })
  ).body as ProjectSnapshot;
}
function render(value: ProjectSnapshot) {
  return renderToStaticMarkup(createElement(Research, { snapshot: value }));
}

test("an unconfigured cycle is absent without inventing an active job", async () => {
  const value = await snapshot(null);
  assert.equal(value.cycle, null);
  assert.doesNotMatch(
    render(value),
    /Research cycle|Awaiting independent review/,
  );
});

test("invalid status is unavailable without publishing error text or local paths", async () => {
  await fixture(async (dir) => {
    for (const invalid of [
      "{SECRET",
      { ...status, schema_version: 2 },
      { ...status, attempts: true },
      { ...status, attempts: 3 },
      { ...status, elapsed_seconds: -1 },
      { ...status, source_sha: "/private/source" },
      { ...status, request_digest: "bad" },
      { ...status, heartbeat_at: "yesterday" },
      { ...status, job_id: "../private" },
      { ...status, phase: "recorded", review_status: "pending" },
      { ...status, phase: "awaiting_review", evidence_digest: null },
      { ...status, raw_log: "SECRET" },
      { ...status, message: "x".repeat(9000) },
    ]) {
      await writeFile(
        join(dir, "status.json"),
        typeof invalid === "string" ? invalid : JSON.stringify(invalid),
      );
      const value = await snapshot(dir);
      assert.equal(value.cycle?.availability, "unavailable");
      assert.equal(value.cycle?.status, null);
      assert.match(render(value), /Cycle status unavailable/);
      assert.doesNotMatch(
        JSON.stringify(value.cycle),
        /SECRET|private|thermo-cycle-/,
      );
    }
  });
});

test("missing and escaping status files do not masquerade as successful evidence", async () => {
  await fixture(async (dir) => {
    assert.equal((await snapshot(dir)).cycle?.availability, "unavailable");
    await symlink("/etc/passwd", join(dir, "status.json"));
    assert.equal((await snapshot(dir)).cycle?.availability, "unavailable");
  });
});

test("pending independent review remains separate from execution and scientific outcome", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    const value = await snapshot(dir);
    assert.equal(value.cycle?.availability, "available");
    assert.equal(value.cycle?.status?.phase, "awaiting_review");
    assert.equal(value.cycle?.status?.reviewStatus, "pending");
    assert.equal(value.cycle?.observedAt, observedAt);
    const html = render(value);
    assert.match(html, /Awaiting independent review/);
    assert.match(html, /Pending/);
    assert.match(html, /Scientific outcome/);
    assert.match(html, /A negative result can be a successful execution/);
    assert.doesNotMatch(
      html,
      /token=SECRET|private\/worker|Scientific success/,
    );
    assert.equal(value.science.state, "unknown");
  });
});

test("stale active heartbeat is an observation warning, never proof of process exit", async () => {
  await fixture(async (dir) => {
    await writeFile(
      join(dir, "status.json"),
      JSON.stringify({
        ...status,
        phase: "running",
        heartbeat_at: "2026-09-26T11:00:00Z",
        evidence_digest: null,
      }),
    );
    const value = await snapshot(dir);
    assert.equal(value.cycle?.availability, "stale");
    assert.equal(value.cycle?.heartbeatAgeSeconds, 3600);
    assert.equal(value.cycle?.status?.phase, "running");
    assert.match(render(value), /Heartbeat is stale/);
    assert.match(
      render(value),
      /does not establish whether the process is running/,
    );
    assert.doesNotMatch(render(value), /Process exited|Worker is live/);
  });
});

test("snapshot export carries only bounded sanitized cycle status and frozen observation time", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    await exportSnapshot("/nonexistent", join(dir, "export"), {
      cycleRoot: dir,
      observedAt,
    });
    const raw = await readFile(join(dir, "export/data/project.json"), "utf8");
    const value = JSON.parse(raw) as ProjectSnapshot;
    assert.equal(value.mode, "snapshot");
    assert.equal(value.generatedAt, observedAt);
    assert.equal(value.cycle?.observedAt, observedAt);
    assert.equal(
      value.cycle?.status?.evidenceDigest,
      `sha256:${"c".repeat(64)}`,
    );
    assert.doesNotMatch(raw, /SECRET|private\/worker|thermo-cycle-/);
    const html = render(value);
    assert.match(html, /Staged snapshot/);
    assert.match(html, /not live worker telemetry/);
  });
});

test("only an approved recorded status can display recorded review, without claiming favorable science", async () => {
  await fixture(async (dir) => {
    await writeFile(
      join(dir, "status.json"),
      JSON.stringify({
        ...status,
        phase: "recorded",
        review_status: "approved",
        message: "Negative outcome: target criterion unmet",
      }),
    );
    const value = await snapshot(dir);
    assert.equal(value.cycle?.availability, "available");
    assert.match(render(value), /Recorded \(reported\)/);
    assert.match(render(value), /Approved \(reported\)/);
    assert.match(
      render(value),
      /A negative result can be a successful execution/,
    );
    assert.doesNotMatch(render(value), /Scientific success|Target met/);
  });
});

test("a staged status shows its observation age without refreshing frozen heartbeat age", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    const value = await snapshot(dir);
    value.mode = "snapshot";
    assert.match(render(value), /Snapshot age at render/);
    assert.match(render(value), /Heartbeat age at observation/);
    assert.match(render(value), /30 seconds/);
  });
});

test("future and missing active heartbeats cannot appear current", async () => {
  await fixture(async (dir) => {
    for (const heartbeat_at of [null, "2026-09-26T12:01:00Z"]) {
      await writeFile(
        join(dir, "status.json"),
        JSON.stringify({ ...status, phase: "running", heartbeat_at }),
      );
      const value = await snapshot(dir);
      assert.equal(value.cycle?.availability, "stale");
      assert.equal(value.cycle?.heartbeatAgeSeconds, null);
    }
  });
});

const reviewRecord = {
  schema_version: 1,
  job_id: status.job_id,
  source_sha: status.source_sha,
  request_digest: status.request_digest,
  evidence_digest: status.evidence_digest,
  evidence_commit: "d".repeat(40),
  reviewed_at: "2026-09-26T11:59:55Z",
  reviewer: "Independent reviewer (separate session)",
  replay: "inspected",
  decision: "approved",
  scope: "Inspected replay artifacts at /private/evidence; token=SECRET",
  findings: [
    "Negative outcome is reproducible",
    "Private diagnostics /private/log",
  ],
};
async function writeReview(dir: string, value: unknown) {
  await mkdir(join(dir, "reviews"), { recursive: true });
  await writeFile(
    join(dir, "reviews", `${"c".repeat(64)}.json`),
    typeof value === "string" ? value : JSON.stringify(value),
  );
}

test("an evidence-bound independent approval remains separate from worker state and owner acceptance", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    for (const replay of ["inspected", "executed"] as const) {
      await writeReview(dir, { ...reviewRecord, replay });
      const value = await snapshot(dir);
      assert.equal(value.cycle?.independentReview?.availability, "available");
      assert.equal(
        value.cycle?.independentReview?.record?.decision,
        "approved",
      );
      assert.equal(value.cycle?.independentReview?.record?.replay, replay);
      assert.equal(
        value.cycle?.independentReview?.record?.evidenceCommit,
        "d".repeat(40),
      );
      assert.equal(value.cycle?.status?.phase, "awaiting_review");
      assert.equal(value.cycle?.status?.reviewStatus, "pending");
      const html = render(value);
      assert.match(html, /Independent review record/);
      assert.match(html, /Independent reviewer \(separate session\)/);
      assert.match(html, /2026-09-26T11:59:55Z/);
      assert.match(
        html,
        replay === "inspected"
          ? /Replay artifacts inspected/
          : /Replay executed by reviewer/,
      );
      assert.match(html, /does not establish owner acceptance/);
      assert.doesNotMatch(html, /Recorded \(reported\)/);
      await exportSnapshot("/nonexistent", join(dir, "export"), {
        cycleRoot: dir,
        observedAt,
      });
      const raw = await readFile(join(dir, "export/data/project.json"), "utf8");
      assert.doesNotMatch(
        raw,
        /SECRET|private\/|Private diagnostics|Negative outcome is reproducible/,
      );
      assert.equal(
        JSON.parse(raw).cycle.independentReview.record.replay,
        replay,
      );
    }
  });
});

test("missing independent review is unavailable here rather than inferred pending", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    const value = await snapshot(dir);
    assert.equal(value.cycle?.independentReview, null);
    assert.match(
      render(value),
      /No independent review record is available in this observation/,
    );
  });
});

test("malformed or mismatched independent review cannot approve current evidence", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    for (const invalid of [
      "{SECRET",
      { ...reviewRecord, job_id: "other-job" },
      { ...reviewRecord, source_sha: "e".repeat(40) },
      { ...reviewRecord, request_digest: `sha256:${"e".repeat(64)}` },
      { ...reviewRecord, evidence_digest: `sha256:${"e".repeat(64)}` },
      { ...reviewRecord, evidence_commit: "branch-name" },
      { ...reviewRecord, reviewed_at: "yesterday" },
      { ...reviewRecord, replay: "not-run" },
      { ...reviewRecord, decision: "recorded" },
      { ...reviewRecord, findings: Array(21).fill("finding") },
      { ...reviewRecord, reviewer: " ".repeat(160) + "a" },
      { ...reviewRecord, scope: " ".repeat(2048) + "a" },
      { ...reviewRecord, findings: [" ".repeat(1024) + "a"] },
      { ...reviewRecord, raw_log: "SECRET" },
    ]) {
      await writeReview(dir, invalid);
      const value = await snapshot(dir);
      assert.equal(value.cycle?.availability, "available");
      assert.equal(value.cycle?.independentReview?.availability, "unavailable");
      assert.equal(value.cycle?.independentReview?.record, null);
      assert.equal(value.cycle?.status?.phase, "awaiting_review");
      assert.match(render(value), /Independent review record unavailable/);
      assert.doesNotMatch(
        JSON.stringify(value.cycle),
        /SECRET|branch-name|yesterday|other-job/,
      );
    }
  });
});

test("reviewer identity never exposes local paths or credential-shaped text", async () => {
  await fixture(async (dir) => {
    await writeFile(join(dir, "status.json"), JSON.stringify(status));
    for (const reviewer of [
      "/private/reviewer.log",
      "token=SECRET",
      "Bearer SECRET",
      "C:\\private\\reviewer.log",
      "ghp_" + "a".repeat(36),
      "sk-proj-" + "a".repeat(40),
      "github_pat_" + "b".repeat(40),
      "xoxb-123456-abcdef",
    ]) {
      await writeReview(dir, {
        ...reviewRecord,
        reviewer,
        decision: "changes_requested",
      });
      const value = await snapshot(dir);
      assert.equal(
        value.cycle?.independentReview?.record?.decision,
        "changes_requested",
      );
      assert.equal(
        value.cycle?.independentReview?.record?.reviewer,
        "Identity withheld",
      );
      assert.ok(!JSON.stringify(value.cycle).includes(reviewer));
      assert.doesNotMatch(JSON.stringify(value.cycle), /SECRET|private/);
    }
  });
});
