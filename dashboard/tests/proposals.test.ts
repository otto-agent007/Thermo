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
      route + "/report.md",
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
