import { test } from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { Overview } from "../src/features/Overview";
import { loadEvidence } from "../server/evidence";
test("missing archive cannot imply a completed study in the overview headline", async () => {
  const { snapshot } = await loadEvidence("/nonexistent");
  const html = renderToStaticMarkup(createElement(Overview, { snapshot }));
  assert.doesNotMatch(html, /A completed study/);
  assert.match(html, /Evidence unavailable/);
});
import { Roadmap } from "../src/features/Roadmap";
test("roadmap cannot present M4G as complete when archive is unavailable", async () => {
  const { snapshot } = await loadEvidence("/nonexistent");
  assert.equal(
    snapshot.milestones.find((m) => m.id === "M4G")?.state,
    "Unavailable",
  );
  const html = renderToStaticMarkup(createElement(Roadmap, { snapshot }));
  assert.match(html, /Evidence unavailable/);
});
