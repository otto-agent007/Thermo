import { test, expect } from "@playwright/test";
test("project status and sourced destinations are accessible", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await expect(
    page.getByText("Joint quality unmet", { exact: true }),
  ).toBeVisible();
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
