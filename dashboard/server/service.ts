import { readProposals, readProposalArtifact } from "./proposals.ts";
import { stat } from "node:fs/promises";
import { join } from "node:path";
import { loadEvidence } from "./evidence.ts";
import { observeActivity } from "./activity.ts";
import { archive } from "./catalog.ts";
import pins from "./archive-pins.json" with { type: "json" };
const cache = new Map<
  string,
  { key: string; pending: ReturnType<typeof loadEvidence> }
>();
export async function getEvidence(root: string) {
  const key = (
    await Promise.all(
      Object.keys(pins).map(async (name) => {
        try {
          const s = await stat(join(root, archive, name));
          return `${s.ino}:${s.size}:${s.mtimeMs}:${s.ctimeMs}`;
        } catch {
          return "missing";
        }
      }),
    )
  ).join("|");
  let entry = cache.get(root);
  if (!entry || entry.key !== key) {
    entry = { key, pending: loadEvidence(root) };
    cache.set(root, entry);
    if (cache.size > 8) cache.delete(cache.keys().next().value!);
  }
  return entry.pending;
}
export async function getPayload(
  root: string,
  route: string,
  method = "GET",
): Promise<{ status: number; body: unknown; contentType?: string }> {
  if (!["GET", "HEAD"].includes(method))
    return { status: 405, body: { error: "Read-only API" } };
  const path = route.split("?")[0];
  if (path === "/data/proposals.json")
    return { status: 200, body: await readProposals(root) };
  const proposal =
    /^\/data\/proposals\/([0-9a-f-]{36})\/(patch\.diff|report\.md|artifacts\/(?:overview|mobile|experiments)\.png)$/.exec(
      path,
    );
  if (proposal)
    return readProposalArtifact(
      root,
      proposal[1],
      proposal[2].replace("artifacts/", ""),
    );
  if (path === "/data/project.json") {
    const [{ snapshot }, activity] = await Promise.all([
      getEvidence(root),
      observeActivity(root),
    ]);
    return {
      status: 200,
      body: { ...snapshot, generatedAt: new Date().toISOString(), activity },
    };
  }
  if (path.startsWith("/data/cells/") && path.endsWith(".json")) {
    let id: string;
    try {
      id = decodeURIComponent(path.slice(12, -5)).replaceAll("~", "/");
    } catch {
      return { status: 404, body: { error: "Unknown cell" } };
    }
    if (
      !/^seed-[012]\/(finite|equilibrium|frozen)\/(1|2|4|8|16|30|equilibrium)$/.test(
        id,
      )
    )
      return { status: 404, body: { error: "Unknown cell" } };
    const cell = (await getEvidence(root)).details.get(id);
    if (cell) return { status: 200, body: cell };
  }
  return { status: 404, body: { error: "Unknown evidence route" } };
}
