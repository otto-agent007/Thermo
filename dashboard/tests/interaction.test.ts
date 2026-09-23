import { test } from "node:test";
import assert from "node:assert/strict";
import { Window } from "happy-dom";
import { createElement, act } from "react";
import { createRoot } from "react-dom/client";
import { loadEvidence } from "../server/evidence";
import { resolve } from "node:path";
test("skip shortcut preserves experiment filters and focuses main content", async () => {
  const window = new Window({ url: "http://localhost/#experiments" });
  Object.assign(globalThis, {
    window,
    document: window.document,
    location: window.location,
    HTMLElement: window.HTMLElement,
    IS_REACT_ACT_ENVIRONMENT: true,
  });
  const { snapshot } = await loadEvidence(resolve(".."));
  snapshot.mode = "snapshot";
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify(snapshot));
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const { default: App } = await import("../src/App");
  try {
    await act(async () => root.render(createElement(App)));
    const seed = document.querySelector("select")!;
    await act(async () => {
      seed.value = "0";
      seed.dispatchEvent(
        new window.Event("change", { bubbles: true }) as unknown as Event,
      );
    });
    assert.match(
      document.querySelector("caption")!.textContent!,
      /20 evaluation cells/,
    );
    const skip = document.querySelector(".skip")!;
    await act(async () => {
      skip.dispatchEvent(
        new window.MouseEvent("click", {
          bubbles: true,
          cancelable: true,
          button: 0,
        }) as unknown as Event,
      );
    });
    assert.equal(document.activeElement?.id, "main-content");
    assert.equal(location.hash, "#experiments");
    assert.match(
      document.querySelector("caption")!.textContent!,
      /20 evaluation cells/,
    );
  } finally {
    await act(async () => root.unmount());
    globalThis.fetch = originalFetch;
    await window.happyDOM.close();
  }
});
