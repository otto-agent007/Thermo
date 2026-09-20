import { mkdir, mkdtemp, writeFile, rename, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadEvidence } from "./evidence.ts";
import { observeActivity } from "./activity.ts";
export async function exportSnapshot(
  root: string,
  destination: string,
): Promise<void> {
  const { snapshot, details } = await loadEvidence(root);
  snapshot.mode = "snapshot";
  snapshot.activity = await observeActivity(root);
  await mkdir(destination, { recursive: true });
  const stage = await mkdtemp(join(destination, ".data-"));
  await mkdir(join(stage, "cells"));
  try {
    await writeFile(join(stage, "project.json"), JSON.stringify(snapshot));
    for (const [id, detail] of details)
      await writeFile(
        join(stage, "cells", id.replaceAll("/", "~") + ".json"),
        JSON.stringify(detail),
      );
    // Complete staging before moving the previous version aside. Build does not serve this directory concurrently.
    const target = join(destination, "data");
    const backup = join(destination, ".data-previous");
    await rm(backup, { recursive: true, force: true });
    let moved = false;
    try {
      await rename(target, backup);
      moved = true;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
    }
    try {
      await rename(stage, target);
    } catch (e) {
      if (moved) await rename(backup, target);
      throw e;
    }
    await rm(backup, { recursive: true, force: true });
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  await exportSnapshot(
    resolve(import.meta.dirname, "../.."),
    resolve(import.meta.dirname, "../public"),
  );
  console.log("Exported dated project snapshot and cell details.");
}
