import { readFile, realpath, stat } from "node:fs/promises";
import { resolve, sep } from "node:path";
export async function readBounded(
  root: string,
  path: string,
  maxBytes = 20_000_000,
): Promise<{ text: string; mtime: string }> {
  const base = await realpath(root);
  const target = await realpath(resolve(root, path));
  if (!target.startsWith(base + sep))
    throw new Error("Source escapes repository");
  for (let attempt = 0; attempt < 2; attempt++) {
    const before = await stat(target);
    if (!before.isFile() || before.size > maxBytes)
      throw new Error("Unsupported source size");
    const text = await readFile(target, "utf8");
    const after = await stat(target);
    if (before.size === after.size && before.mtimeMs === after.mtimeMs)
      return { text, mtime: after.mtime.toISOString() };
  }
  throw new Error("Source changed during read");
}
