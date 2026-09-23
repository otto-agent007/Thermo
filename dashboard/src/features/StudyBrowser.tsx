import { useState } from "react";
import type { ProjectSnapshot } from "../../shared/model";
import { Experiments as M4GExperiments } from "./Experiments";
import { RecentStudyPanel } from "./RecentStudy";

export function Experiments({ snapshot }: { snapshot: ProjectSnapshot }) {
  const [selected, setSelected] = useState(
    location.hash === "#experiments/m4g"
      ? "m4g"
      : (snapshot.recentStudies[0]?.id ?? "m4g"),
  );
  const study = snapshot.recentStudies.find((item) => item.id === selected);
  return (
    <>
      <div className="study-picker">
        <label htmlFor="study-picker">Study</label>
        <select
          id="study-picker"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
        >
          {snapshot.recentStudies.map((item) => (
            <option key={item.id} value={item.id}>
              {item.date} · {item.title}
            </option>
          ))}
          <option value="m4g">
            2026-09-20 · M4G task quality vs inference budget
          </option>
        </select>
      </div>
      {selected === "m4g" ? (
        <M4GExperiments snapshot={snapshot} />
      ) : (
        study && <RecentStudyPanel study={study} />
      )}
    </>
  );
}
