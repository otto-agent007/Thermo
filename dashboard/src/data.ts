import type { CellDetail, ProjectSnapshot } from "../shared/model";
async function read<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${import.meta.env?.BASE_URL ?? "./"}data/${path}`, {
    signal,
    cache: "no-store",
  });
  if (!res.ok) throw Error("Evidence could not be loaded. Please refresh.");
  return res.json();
}
export const loadProject = (signal?: AbortSignal) =>
  read<ProjectSnapshot>("project.json", signal);
export const loadCell = (id: string, signal?: AbortSignal) =>
  read<CellDetail>(
    `cells/${encodeURIComponent(id.replaceAll("/", "~"))}.json`,
    signal,
  );
