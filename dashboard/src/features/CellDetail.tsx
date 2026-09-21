import { useEffect, useRef, useState } from "react";
import type { CellDetail as Detail } from "../../shared/model";
import { loadCell } from "../data";
import { Sources } from "../components/SourceLink";
import { X } from "lucide-react";
export function CellDetail({
  id,
  onClose,
}: {
  id: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    return () => {
      dialog.current?.close();
      opener?.focus();
    };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setData(null);
    setError(null);
    loadCell(id, controller.signal)
      .then((d) => {
        if (!controller.signal.aborted) setData(d);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [id]);
  return (
    <dialog
      ref={dialog}
      className="detail"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      aria-labelledby="detail-title"
    >
      <div className="row-between">
        <h2 id="detail-title">{id}</h2>
        <button onClick={onClose} aria-label="Close cell detail">
          <X size={18} />
        </button>
      </div>
      {error ? (
        <p role="alert">{error}</p>
      ) : !data ? (
        <p>Loading cell evidence…</p>
      ) : (
        <>
          <p className="fine">
            {data.sampleCount.toLocaleString()} complete trajectories · sampled
            metrics: {data.sampledEvidence} · local metrics and survival:{" "}
            {data.exactEvidence}
          </p>
          <dl className="detail-metrics">
            {[
              ["Population loss", data.loss],
              ["Loss gate bounds", data.lossBounds.join(" – ")],
              ["Terminal leakage", data.leakage],
              ["Leakage bounds", data.leakageInterval.join(" – ")],
              ["Survival at operation 500", data.survival],
              ["Hop MAE", data.hopMae],
              ["Asymmetry MAE", data.asymmetryMae],
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
          <h3>Terminal particle histogram</h3>
          <div className="histogram">
            {data.histogram.map((count, i) => (
              <div key={i} title={`${i} particles: ${count}`}>
                <div className="hist-track">
                  <span
                    style={{ height: `${(100 * count) / data.sampleCount}%` }}
                  />
                </div>
                <small>{i}</small>
              </div>
            ))}
          </div>
          <p className="fine">
            Particle count on the x-axis · fraction of{" "}
            {data.sampleCount.toLocaleString()} trajectories
          </p>
          <h3>Uninterrupted survival</h3>
          <svg
            className="survival-chart"
            viewBox="0 0 600 170"
            role="img"
            aria-label="Survival probability across 500 operations"
          >
            <line x1="35" x2="580" y1="140" y2="140" stroke="#b8c8cd" />
            <text x="5" y="20">
              1
            </text>
            <text x="5" y="144">
              0
            </text>
            <text x="35" y="163">
              1
            </text>
            <text x="550" y="163">
              500
            </text>
            <polyline
              points={data.survivalTrace
                .map(
                  (p) =>
                    `${35 + ((p.operation - 1) * 545) / 499},${140 - p.probability * 125}`,
                )
                .join(" ")}
              fill="none"
              stroke="#146b67"
              strokeWidth="2"
            />
          </svg>
          <details>
            <summary>Exact histogram and survival values</summary>
            <p className="mono">
              {data.histogram.map((v, i) => `${i}: ${v}`).join(" · ")}
            </p>
            <div className="trace-table table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Operation</th>
                    <th>Survival probability</th>
                  </tr>
                </thead>
                <tbody>
                  {data.survivalTrace.map((p) => (
                    <tr key={p.operation}>
                      <td>{p.operation}</td>
                      <td>{p.probability}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <h3>Source identity</h3>
          <dl className="identities">
            {Object.entries(data.identities).map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          <Sources sources={data.sources} />
        </>
      )}
    </dialog>
  );
}
