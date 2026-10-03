import type { ResearchItem, Source } from "../shared/model.ts";
import { readBounded } from "./files.ts";

// Research items read from the knowledge base in docs/knowledge/ instead of a
// hand-written list. Lessons become "finding" items and the ranked backlog
// becomes "proposal" items. Text is copied verbatim; nothing here is evidence
// and nothing is reclassified. A missing or malformed file yields no items
// and one issue string, never an exception, so the dashboard still renders.
export const lessonsPath = "docs/knowledge/lessons.md";
export const backlogPath = "docs/knowledge/experiment-backlog.md";
const githubBlob = "https://github.com/otto-agent007/Thermo/blob/main/";
const LINK = /\[([^\]]+)\]\(([^)\s]+)\)/g;

type Row = string[];
function tableRows(markdown: string, firstHeader: string): Row[] {
  const rows: Row[] = [];
  let inTable = false;
  for (const raw of markdown.split("\n")) {
    const line = raw.trim();
    if (!line.startsWith("|")) {
      if (inTable) break;
      continue;
    }
    const cells = line
      .slice(1, line.endsWith("|") ? -1 : undefined)
      .split("|")
      .map((c) => c.trim());
    if (!inTable) {
      if (cells[0] === firstHeader) inTable = true;
      continue;
    }
    if (cells.every((c) => /^-+$/.test(c))) continue;
    rows.push(cells);
  }
  return rows;
}

function resolveDoc(fromDir: string, target: string): string {
  const parts = fromDir.split("/").filter(Boolean);
  for (const seg of target.split("/")) {
    if (seg === "..") parts.pop();
    else if (seg !== "." && seg !== "") parts.push(seg);
  }
  return parts.join("/");
}

function sources(
  cell: string,
  fromDir: string,
  recordedAt: string | null,
): Source[] {
  const out: Source[] = [];
  for (const match of cell.matchAll(LINK)) {
    const [, label, target] = match;
    const url = /^https?:/.test(target)
      ? target
      : githubBlob + resolveDoc(fromDir, target.split("#")[0]);
    out.push({ label, url, recordedAt });
  }
  return out;
}

function plain(cell: string): string {
  return cell.replace(LINK, "$1").replace(/`/g, "");
}

export function parseLessons(markdown: string): ResearchItem[] {
  // | Date | Study | Lesson | Evidence class | Source |
  return tableRows(markdown, "Date")
    .filter((r) => r.length >= 5)
    .map((r, i) => ({
      id: `lesson-${r[0]}-${i}`,
      kind: "finding" as const,
      title: `${r[1]} (${r[0]})`,
      text: plain(r[2]),
      scope: `Lesson recorded in the knowledge base; evidence class as the source states it: ${plain(r[3])}.`,
      sources: sources(r[4], "docs/knowledge", r[0]),
    }));
}

export function parseBacklog(markdown: string): ResearchItem[] {
  // | # | Experiment | Source | Exact anchor | Cost | Fit |
  return tableRows(markdown, "#")
    .filter((r) => r.length >= 6 && /^E\d+$/.test(r[0]))
    .map((r) => ({
      id: `backlog-${r[0]}`,
      kind: "proposal" as const,
      title: `${r[0]}: ${plain(r[1])}`,
      text: `Exact anchor: ${plain(r[3])}. Cost: ${plain(r[4])}. Fit: ${plain(r[5])}.`,
      scope:
        "Backlog candidate from docs/knowledge/experiment-backlog.md. Not scheduled, not frozen, not evidence; the owner chooses the next study.",
      sources: [
        ...sources(r[2], "docs/knowledge", null),
        {
          label: "Experiment backlog",
          url: githubBlob + backlogPath,
          recordedAt: null,
        },
      ],
    }));
}

export async function loadKnowledgeResearch(
  root: string,
): Promise<{ items: ResearchItem[]; issues: string[] }> {
  const items: ResearchItem[] = [];
  const issues: string[] = [];
  const load = async (path: string, parse: (s: string) => ResearchItem[]) => {
    try {
      items.push(...parse((await readBounded(root, path, 2_000_000)).text));
    } catch (error) {
      // Keep the snapshot portable: name the file, not the machine path.
      const code = (error as { code?: string } | null)?.code;
      issues.push(
        code === "ENOENT"
          ? `${path}: not present in this checkout`
          : `${path}: unreadable (${code ?? "error"})`,
      );
    }
  };
  await load(lessonsPath, parseLessons);
  await load(backlogPath, parseBacklog);
  return { items, issues };
}
