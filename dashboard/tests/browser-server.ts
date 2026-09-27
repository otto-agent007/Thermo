// Isolated local records keep browser tests independent of developer drafts.
import { mkdtemp, cp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { archive } from "../server/catalog.ts";
import { recentReportPaths } from "../server/recent.ts";
import { evidencePlugin } from "../server/plugin.ts";
import { proposalFixture } from "./proposal-fixture.ts";
const root = await mkdtemp(join(tmpdir(), "thermo-browser-"));
await mkdir(join(root, archive), { recursive: true });
await cp(resolve("..", archive), join(root, archive), { recursive: true });
for (const path of recentReportPaths) {
  await mkdir(dirname(join(root, path)), { recursive: true });
  await cp(resolve("..", path), join(root, path));
}
await proposalFixture(root);
// Explicit synthetic status belongs only to this isolated browser-test server.
process.env.THERMO_CYCLE_ROOT = join(root, "cycle");
await mkdir(process.env.THERMO_CYCLE_ROOT);
await writeFile(
  join(process.env.THERMO_CYCLE_ROOT, "status.json"),
  JSON.stringify({
    schema_version: 1,
    job_id: "m5a-browser-fixture",
    source_sha: "a".repeat(40),
    request_digest: `sha256:${"b".repeat(64)}`,
    phase: "awaiting_review",
    attempts: 1,
    elapsed_seconds: 123.5,
    heartbeat_at: new Date().toISOString(),
    evidence_digest: `sha256:${"c".repeat(64)}`,
    review_status: "pending",
    message: "private token=SECRET must not leave the server",
  }),
);
const server = await createServer({
  configFile: false,
  root: resolve("."),
  base: "./",
  plugins: [react(), evidencePlugin(root)],
  server: { host: "127.0.0.1", port: 5174, strictPort: true },
});
await server.listen();
async function close() {
  await server.close();
  await rm(root, { recursive: true, force: true });
  process.exit(0);
}
process.on("SIGTERM", close);
process.on("SIGINT", close);
