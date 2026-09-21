import { ArrowUpRight } from "lucide-react";
import type { Source } from "../../shared/model";
export function SourceLink({ source }: { source: Source }) {
  return (
    <a
      className="source-link"
      href={source.url}
      target="_blank"
      rel="noreferrer"
    >
      {source.label}
      <ArrowUpRight size={14} aria-hidden="true" />
    </a>
  );
}
export function Sources({ sources }: { sources: Source[] }) {
  return (
    <div className="sources">
      {sources.map((s) => (
        <SourceLink key={s.url} source={s} />
      ))}
    </div>
  );
}
