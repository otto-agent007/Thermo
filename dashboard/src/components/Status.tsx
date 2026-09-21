import { Check, FileText, CircleHelp, TriangleAlert } from "lucide-react";
import type { Status as State } from "../../shared/model";
export function Status({ title, status }: { title: string; status: State }) {
  const bad = status.state === "quality_failure";
  const Icon = bad
    ? TriangleAlert
    : status.state === "complete"
      ? Check
      : status.state === "recorded"
        ? FileText
        : CircleHelp;
  return (
    <div className={`status-item ${bad ? "amber" : ""}`}>
      <span className="status-icon">
        <Icon size={21} aria-hidden="true" />
      </span>
      <div>
        <strong>{title}</strong>
        <span>{status.label}</span>
      </div>
    </div>
  );
}
