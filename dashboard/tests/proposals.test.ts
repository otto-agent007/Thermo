import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtemp,
  rm,
  writeFile,
  readFile,
  symlink,
  rename,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getPayload } from "../server/service";
import { exportSnapshot } from "../server/export";
import {
  proposalFixture,
  proposalId,
  record,
  digest,
} from "./proposal-fixture";
const route = `/data/proposals/${proposalId}`;
async function fixture(
  fn: (
    root: string,
    value: Awaited<ReturnType<typeof proposalFixture>>,
  ) => Promise<void>,
) {
  const root = await mkdtemp(join(tmpdir(), "thermo-proposals-"));
  try {
    await fn(root, await proposalFixture(root));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}
test("proposal summary omits private provenance and serves authenticated patch and PNG bytes", () =>
  fixture(async (root, f) => {
    const response = await getPayload(root, "/data/proposals.json");
    assert.equal(response.status, 200);
    const body = response.body as {
      items: { objective: string; review: string; screenshotUrls: string[] }[];
      issues: string[];
    };
    assert.equal(body.items[0].objective, "Improve dashboard readability");
    assert.equal(body.items[0].review, "proposed");
    assert.deepEqual(body.items[0].screenshotUrls, [
      route + "/artifacts/overview.png",
      route + "/artifacts/mobile.png",
      route + "/artifacts/experiments.png",
    ]);
    assert.doesNotMatch(JSON.stringify(body), /private|secret/);
    assert.equal((await getPayload(root, route + "/patch.diff")).body, f.patch);
    assert.deepEqual(
      (await getPayload(root, route + "/artifacts/overview.png")).body,
      f.png,
    );
    assert.equal(
      (await getPayload(root, route + "/patch.diff", "HEAD")).status,
      200,
    );
  }));
for (const damage of [
  "missing",
  "digest",
  "oversized",
  "symlink",
  "patch",
  "image",
  "version",
  "review",
])
  test(`rejects ${damage} proposal evidence without disclosing paths`, () =>
    fixture(async (root, f) => {
      if (damage === "missing") await rm(join(f.dir, "result.json"));
      if (damage === "digest")
        await writeFile(join(f.dir, "request.json"), "{}");
      if (damage === "oversized")
        await writeFile(join(f.dir, "result.json"), " ".repeat(1_100_000));
      if (damage === "symlink") {
        await rename(f.dir, f.dir + "-saved");
        await symlink(f.dir + "-saved", f.dir);
      }
      if (damage === "patch")
        await writeFile(join(f.dir, "patch.diff"), "changed");
      if (damage === "image")
        await writeFile(
          join(f.dir, "candidate/artifacts/overview.png"),
          "changed",
        );
      if (damage === "version")
        await record(f.dir, "result", { ...f.result, schema_version: 2 });
      if (damage === "review")
        await writeFile(
          join(f.dir, "review-0001.json"),
          JSON.stringify({
            schema_version: 1,
            candidate_id: "wrong",
            decision: "accepted",
          }),
        );
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: unknown[];
        issues: string[];
      };
      assert.deepEqual(body.items, []);
      assert.ok(body.issues.length);
      assert.ok(!JSON.stringify(body).includes(root));
      assert.equal((await getPayload(root, route + "/patch.diff")).status, 404);
    }));
test("proposal routes deny traversal, extra artifact names and all mutations", () =>
  fixture(async (root) => {
    for (const path of [
      route + "/artifacts/secret.png",
      route + "/artifacts/../overview.png",
      "/data/proposals/../../secret/patch.diff",
      route + "/%2e%2e/patch.diff",
      route + "/raw-result.json",
    ])
      assert.equal((await getPayload(root, path)).status, 404);
    for (const method of ["POST", "PUT", "DELETE", "PATCH"])
      assert.equal(
        (await getPayload(root, "/data/proposals.json", method)).status,
        405,
      );
  }));
test("static export excludes an unreviewed draft and produces an empty portable list", () =>
  fixture(async (root) => {
    const out = join(root, "export");
    await exportSnapshot(root, out);
    assert.deepEqual(
      JSON.parse(await readFile(join(out, "data/proposals.json"), "utf8")),
      { items: [], issues: [] },
    );
  }));
test("research summaries read exact values from the CLI science observation", () =>
  fixture(async (root, f) => {
    await record(f.dir, "request", {
      ...f.request,
      track: "research",
      allowed_paths: ["src/thermo_lab/research_candidates/three_site.py"],
    });
    await record(f.dir, "result", {
      ...f.result,
      research_outcome: "improved",
      science: {
        before: 0.25,
        after: 0.2,
        delta: -0.05,
        evidence: "exact_reference",
        scope: "three-site fixture only",
        baseline_commit: f.request.baseline_commit,
        execution: "complete",
        verification: "passed",
        research_outcome: "improved",
        primary_metric: "exact_objective_delta",
        parameters: Array(9).fill(0),
      },
    });
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: { baselineSummary: string; candidateSummary: string }[];
    };
    assert.equal(body.items[0].baselineSummary, "Exact objective: 0.25");
    assert.equal(body.items[0].candidateSummary, "Exact objective: 0.2");
    const report = await getPayload(root, route + "/report.md");
    assert.equal(report.status, 200);
    assert.match(String(report.body), /Before: 0.25/);
    assert.match(String(report.body), /After: 0.2/);
    assert.match(String(report.body), /Evidence: exact_reference/);
    assert.match(String(report.body), /Scope: three-site fixture only/);
  }));
test("empty local proposal root is available with no records", async () => {
  const root = await mkdtemp(join(tmpdir(), "thermo-empty-"));
  try {
    assert.deepEqual((await getPayload(root, "/data/proposals.json")).body, {
      items: [],
      issues: [],
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("append-only owner review is distinct from verification", () =>
  fixture(async (root, f) => {
    await writeFile(
      join(f.dir, "review-0001.json"),
      JSON.stringify({
        schema_version: 1,
        candidate_id: proposalId,
        decision: "rejected",
        note: "private owner note",
        observed_at: "2026-09-23T10:00:00+00:00",
      }),
    );
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: { review: string; verification: string }[];
    };
    assert.equal(body.items[0].review, "rejected");
    assert.equal(body.items[0].verification, "passed");
    assert.doesNotMatch(JSON.stringify(body), /private owner note/);
  }));
for (const path of [
  "patch.diff",
  "candidate/artifacts/overview.png",
  "candidate/artifacts",
  "request.json",
])
  test(`symlinked ${path} cannot be served`, () =>
    fixture(async (root, f) => {
      const target = join(f.dir, path);
      await rename(target, target + "-saved");
      await symlink(target + "-saved", target);
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: unknown[];
        issues: string[];
      };
      assert.deepEqual(body.items, []);
      assert.ok(body.issues.length);
      assert.equal(
        (await getPayload(root, route + "/artifacts/overview.png")).status,
        404,
      );
    }));
for (const evidence of ["baseline", "candidate-log"])
  test(`changed ${evidence} evidence invalidates the candidate`, () =>
    fixture(async (root, f) => {
      const target =
        evidence === "baseline"
          ? join(root, "results/harness", f.result.baseline_record)
          : join(f.dir, "candidate/checks/unit.log");
      await writeFile(target, "changed");
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: unknown[];
        issues: string[];
      };
      assert.deepEqual(body.items, []);
      assert.ok(body.issues.length);
    }));
test("unknown baseline check catalog versions are unavailable", () =>
  fixture(async (root, f) => {
    const baselineDir = join(
      root,
      "results/harness",
      f.result.baseline_record,
      "..",
    );
    const baseline = JSON.parse(
      await readFile(join(baselineDir, "result.json"), "utf8"),
    );
    baseline.checks[0].catalog_version = 2;
    await record(baselineDir, "result", baseline);
    await record(f.dir, "result", {
      ...f.result,
      baseline_record_digest: digest(
        await readFile(join(baselineDir, "result.json")),
      ),
    });
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: unknown[];
      issues: string[];
    };
    assert.deepEqual(body.items, []);
    assert.ok(body.issues.length);
  }));
for (const invalid of [
  { execution: "timed_out", exit_status: 99, verification: "passed" },
  { execution: "complete", exit_status: 99, verification: "passed" },
  { execution: "failed", exit_status: 0, verification: "failed" },
  { execution: "unavailable", exit_status: null, verification: "passed" },
  { argv: [] },
  { exit_status: 1.5 },
  { duration_seconds: -1 },
  { duration_seconds: null },
  { log_tail: "x".repeat(4097) },
  { extra: true },
])
  test(`invalid CheckResult semantics are rejected: ${Object.keys(invalid).join(",")} ${JSON.stringify(invalid).slice(0, 70)}`, () =>
    fixture(async (root, f) => {
      await record(f.dir, "result", {
        ...f.result,
        checks: [
          { ...f.result.checks[0], ...invalid },
          ...f.result.checks.slice(1),
        ],
      });
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: unknown[];
        issues: string[];
      };
      assert.deepEqual(body.items, []);
      assert.ok(body.issues.length);
    }));
test("baseline summary copies cannot claim passed checks over a failed authenticated baseline", () =>
  fixture(async (root, f) => {
    const baselineDir = join(
      root,
      "results/harness",
      f.result.baseline_record,
      "..",
    );
    const baseline = JSON.parse(
      await readFile(join(baselineDir, "result.json"), "utf8"),
    );
    baseline.checks[0] = {
      ...baseline.checks[0],
      execution: "failed",
      verification: "failed",
      exit_status: 99,
    };
    await record(baselineDir, "result", baseline);
    await record(f.dir, "result", {
      ...f.result,
      baseline_record_digest: digest(
        await readFile(join(baselineDir, "result.json")),
      ),
    });
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: unknown[];
      issues: string[];
    };
    assert.deepEqual(body.items, []);
    assert.ok(body.issues.length);
  }));
for (const missing of [
  "checks",
  "baseline",
  "patch",
  "plan",
  "recommendation",
  "screenshots",
  "catalog",
  "duplicate",
])
  test(`complete/passed is not displayed without ${missing} evidence`, () =>
    fixture(async (root, f) => {
      const result = { ...f.result, artifacts: { ...f.result.artifacts } };
      if (missing === "checks") result.checks = [];
      if (missing === "baseline") {
        delete (result as Partial<typeof result>).baseline_record;
        delete (result as Partial<typeof result>).baseline_record_digest;
      }
      if (missing === "patch")
        delete (result as Partial<typeof result>).patch_digest;
      if (missing === "plan") delete result.artifacts["plan.json"];
      if (missing === "recommendation")
        delete result.artifacts["recommendation.md"];
      if (missing === "screenshots")
        delete result.artifacts["candidate/artifacts/mobile.png"];
      if (missing === "catalog")
        result.checks[0] = { ...result.checks[0], argv: ["npm", "test"] };
      if (missing === "duplicate")
        result.checks = result.checks.map(() => result.checks[0]);
      await record(f.dir, "result", result);
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: {
          execution: string;
          verification: string;
          baselineSummary: string;
        }[];
      };
      assert.equal(body.items[0].verification, "inconclusive");
      if (missing === "baseline")
        assert.equal(body.items[0].baselineSummary, "Unavailable");
    }));
test("research success without science stays inconclusive", () =>
  fixture(async (root, f) => {
    await record(f.dir, "request", { ...f.request, track: "research" });
    await record(f.dir, "result", {
      ...f.result,
      research_outcome: "improved",
    });
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: { verification: string; researchOutcome: string }[];
    };
    assert.equal(body.items[0].verification, "inconclusive");
    assert.equal(body.items[0].researchOutcome, "inconclusive");
  }));
for (const status of ["failed", "timed_out", "unavailable"])
  test(`legitimate ${status} draft without completed checks remains visible`, () =>
    fixture(async (root, f) => {
      await record(f.dir, "result", {
        schema_version: 1,
        execution: status,
        verification: status === "failed" ? "failed" : "inconclusive",
        research_outcome: "not_applicable",
        checks: [],
      });
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: { execution: string; verification: string }[];
      };
      assert.equal(body.items[0].execution, status);
      assert.notEqual(body.items[0].verification, "passed");
    }));
test("summary prose redacts Windows, UNC and single-component absolute paths", () =>
  fixture(async (root, f) => {
    const secret =
      "Inspect C:\\Users\\owner\\private and /workspace and \\\\server\\share\\private";
    await record(f.dir, "request", { ...f.request, objective: secret });
    await record(f.dir, "result", { ...f.result, recommendation: secret });
    const body = (await getPayload(root, "/data/proposals.json")).body;
    assert.doesNotMatch(
      JSON.stringify(body),
      /Users|owner|private|workspace|server|share/,
    );
  }));
test("derived report is bounded, authenticated and excludes raw report/log/provenance contents", () =>
  fixture(async (root, f) => {
    await writeFile(
      join(f.dir, "report.md"),
      "secret raw diagnostic /workspace C:\\Users\\owner",
    );
    const response = await getPayload(root, route + "/report.md");
    assert.equal(response.status, 200);
    assert.match(
      String(response.body),
      /Baseline commit:.*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/,
    );
    assert.ok(String(response.body).includes(f.request.plan_digest));
    assert.ok(String(response.body).includes(f.result.patch_digest));
    assert.match(String(response.body), /Baseline checks/);
    assert.match(String(response.body), /Candidate checks/);
    assert.doesNotMatch(
      String(response.body),
      /secret|workspace|Users|private|log_tail|source_state/,
    );
    assert.ok(Buffer.byteLength(String(response.body)) < 32_768);
    assert.equal(
      (await getPayload(root, route + "/report.md", "HEAD")).status,
      200,
    );
    assert.equal(
      (await getPayload(root, route + "/report.md", "POST")).status,
      405,
    );
    await writeFile(join(f.dir, "patch.diff"), "changed");
    assert.equal((await getPayload(root, route + "/report.md")).status, 404);
  }));
test("complete research catalog and authenticated exact evidence can verify passed", async () => {
  const root = await mkdtemp(join(tmpdir(), "thermo-research-proposal-"));
  try {
    await proposalFixture(root, "research");
    const body = (await getPayload(root, "/data/proposals.json")).body as {
      items: {
        verification: string;
        researchOutcome: string;
        candidateSummary: string;
      }[];
    };
    assert.equal(body.items[0].verification, "passed");
    assert.equal(body.items[0].researchOutcome, "improved");
    assert.equal(body.items[0].candidateSummary, "Exact objective: 0.2");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
for (const defect of ["catalog", "screenshot", "timeout"])
  test(`authenticated baseline ${defect} cannot certify a candidate`, () =>
    fixture(async (root, f) => {
      const baselineDir = join(
        root,
        "results/harness",
        f.result.baseline_record,
        "..",
      );
      const baseline = JSON.parse(
        await readFile(join(baselineDir, "result.json"), "utf8"),
      );
      if (defect === "catalog") baseline.checks[0].argv = ["npm", "test"];
      if (defect === "screenshot")
        delete baseline.artifacts["artifacts/mobile.png"];
      if (defect === "timeout")
        baseline.checks[0] = {
          ...baseline.checks[0],
          execution: "timed_out",
          verification: "inconclusive",
          exit_status: 99,
        };
      await record(baselineDir, "result", baseline);
      await record(f.dir, "result", {
        ...f.result,
        baseline_checks: baseline.checks.map(
          ({
            name,
            execution,
            verification,
          }: {
            name: string;
            execution: string;
            verification: string;
          }) => ({ name, execution, verification }),
        ),
        baseline_record_digest: digest(
          await readFile(join(baselineDir, "result.json")),
        ),
      });
      const body = (await getPayload(root, "/data/proposals.json")).body as {
        items: { verification: string }[];
      };
      assert.equal(body.items[0].verification, "inconclusive");
    }));
test("markdown-delimited POSIX paths are redacted in summary prose", () =>
  fixture(async (root, f) => {
    await record(f.dir, "result", {
      ...f.result,
      recommendation: "Fix `/workspace` before checking (/private/worktree).",
    });
    const body = (await getPayload(root, "/data/proposals.json")).body;
    assert.doesNotMatch(JSON.stringify(body), /workspace|private|worktree/);
  }));
