import { useCallback, useEffect, useRef, useState } from "react";
import { loadProject } from "./data";
import type { ProjectSnapshot } from "../shared/model";
export function useProject() {
  const [data, setData] = useState<ProjectSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(() => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setRefreshing(true);
    loadProject(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) {
          setData(value);
          setError(null);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setRefreshing(false);
      });
  }, []);
  useEffect(() => {
    refresh();
    return () => request.current?.abort();
  }, [refresh]);
  useEffect(() => {
    if (data?.mode !== "local") return;
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, 15000);
    return () => clearInterval(timer);
  }, [data?.mode, refresh]);
  return { data, error, refreshing, refresh };
}
