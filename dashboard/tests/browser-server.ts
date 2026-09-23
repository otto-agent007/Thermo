// Isolated local records keep browser tests independent of developer drafts.
import { mkdtemp, cp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { archive } from "../server/catalog.ts";
import { evidencePlugin } from "../server/plugin.ts";
import { proposalFixture } from "./proposal-fixture.ts";
const root = await mkdtemp(join(tmpdir(), "thermo-browser-"));
await mkdir(join(root, archive), { recursive: true });
await cp(resolve("..", archive), join(root, archive), { recursive: true });
await proposalFixture(root);
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
