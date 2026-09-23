import { test, expect } from "@playwright/test";
const staticBuild = process.env.DASHBOARD_STATIC === "1";
test("project status and sourced destinations are accessible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1506, height: 1045 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await expect(
    page.getByText("Joint quality unmet", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: "test-results/overview.png", fullPage: true });
  await page.getByRole("link", { name: "Roadmap", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Roadmap and activity" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Research", exact: true }).click();
  await expect(
    page.getByText("Proposed", { exact: true }).first(),
  ).toBeVisible();
});
test("mobile has no page overflow and supports keyboard focus", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toBeVisible();
});
test("filters, metrics, detail and CSV stay consistent", async ({ page }) => {
  await page.goto("/#experiments");
  await page.getByLabel("Study", { exact: true }).selectOption("m4g");
  await expect(
    page.getByRole("heading", {
      name: "Task quality vs inference budget",
      exact: true,
    }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/experiments.png",
    fullPage: true,
  });
  await page.getByLabel("Seed", { exact: true }).selectOption("0");
  await page.getByLabel("Training member").selectOption("finite");
  await page.getByLabel("Inference horizon").selectOption("30");
  await expect(page.getByRole("caption")).toContainText("1 evaluation cells");
  for (const metric of [
    "Population loss",
    "Terminal leakage",
    "Uninterrupted survival",
    "Hop MAE",
    "Asymmetry MAE",
  ]) {
    await page.getByRole("button", { name: metric, exact: true }).click();
    await expect(
      page.getByRole("heading", { name: metric, exact: true }),
    ).toBeVisible();
  }
  await page
    .getByRole("button", { name: "Open seed-0/finite/30", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("6717865023900054950");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download selected CSV" }).click();
  expect((await download).suggestedFilename()).toBe(
    "thermo-selected-cells.csv",
  );
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.getByRole("caption")).toContainText("60 evaluation cells");
  await page.getByLabel("Search cells").fill("no-match");
  await expect(
    page.getByText("No cells match these filters. Reset to show all 60 cells."),
  ).toBeVisible();
});
test("recent evidence appears first and the study selector preserves all arms", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "Complete local-row fidelity",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByText("90.66%", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Experiments", exact: true }).click();
  await expect(page.getByLabel("Study", { exact: true })).toHaveValue(
    "full-row",
  );
  await expect(
    page.getByText("All four final arms", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("table").first().locator("tbody tr")).toHaveCount(
    4,
  );
  await page
    .getByLabel("Study", { exact: true })
    .selectOption("survival-audit");
  await expect(page.locator("table").first().locator("tbody tr")).toHaveCount(
    21,
  );
  await page.getByRole("link", { name: "Roadmap", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Saved survival-gradient audit",
      exact: true,
    }),
  ).toBeVisible();
});
test("failed refresh preserves previous view with an explicit notice", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByText("Joint quality unmet", { exact: true }),
  ).toBeVisible();
  await page.route("**/data/project.json", (route) => route.abort());
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Showing previous data");
  await expect(
    page.getByText("Joint quality unmet", { exact: true }),
  ).toBeVisible();
});
test("proposals expose read-only recommendations and validated links with keyboard navigation", async ({
  page,
}) => {
  test.skip(
    staticBuild,
    "local proposal fixtures are excluded from static builds",
  );
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  await page.setViewportSize({ width: 1506, height: 1045 });
  await page.goto("/");
  const nav = page.getByRole("link", { name: "Proposals", exact: true });
  await nav.focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("heading", { name: "Improvement proposals" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Improve dashboard readability" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "Increase contrast and spacing. Owner visual review required.",
    ),
  ).toBeVisible();
  await expect(page.getByText("Owner review", { exact: true })).toBeVisible();
  const patch = page.getByRole("link", { name: "View draft patch" });
  await patch.focus();
  await expect(patch).toBeFocused();
  const response = await page.request.get((await patch.getAttribute("href"))!);
  expect(response.headers()["content-type"]).toContain("text/x-diff");
  expect(await response.text()).toContain("diff --git");
  const report = page.getByRole("link", { name: "Download evidence report" });
  await page.keyboard.press("Tab");
  await expect(report).toBeFocused();
  const reportResponse = await page.request.get(
    (await report.getAttribute("href"))!,
  );
  expect(reportResponse.status()).toBe(200);
  expect(await reportResponse.text()).toContain("Baseline commit:");
  expect(await reportResponse.text()).toContain("Plan digest:");
  const reportDownload = page.waitForEvent("download");
  await page.keyboard.press("Enter");
  expect((await reportDownload).suggestedFilename()).toBe("report.md");
  const screenshot = page.getByRole("link", {
    name: "View overview screenshot",
  });
  expect(
    (
      await page.request.get((await screenshot.getAttribute("href"))!)
    ).headers()["content-type"],
  ).toBe("image/png");
  await page.screenshot({
    path: "test-results/proposals-desktop.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
test("proposal refresh failure retains prior records with a notice", async ({
  page,
}) => {
  test.skip(
    staticBuild,
    "local proposal fixtures are excluded from static builds",
  );
  await page.goto("/#proposals");
  await expect(
    page.getByRole("heading", { name: "Improve dashboard readability" }),
  ).toBeVisible();
  await page.route("**/data/proposals.json", (route) => route.abort());
  await page.getByRole("button", { name: "Refresh proposals" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Showing previous proposals",
  );
  await expect(
    page.getByRole("heading", { name: "Improve dashboard readability" }),
  ).toBeVisible();
});
test("mobile proposals fit the viewport", async ({ page }) => {
  test.skip(
    staticBuild,
    "local proposal fixtures are excluded from static builds",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#proposals");
  await expect(
    page.getByRole("heading", { name: "Improve dashboard readability" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/proposals-mobile.png",
    fullPage: true,
  });
});

test("published snapshot excludes local proposal drafts", async ({ page }) => {
  test.skip(!staticBuild, "static build only");
  await page.goto("/#proposals");
  await expect(
    page.getByRole("heading", { name: "No proposals available" }),
  ).toBeVisible();
  await expect(
    page.getByText("Local drafts are excluded from this static snapshot."),
  ).toBeVisible();
});
