import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";
const staticBuild = process.env.DASHBOARD_STATIC === "1";
const port = staticBuild ? 5176 : 5174;
export default defineConfig({
  testDir: ".",
  testMatch: "browser.spec.ts",
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command: staticBuild
      ? "npm run preview -- --host 127.0.0.1 --port 5176 --strictPort"
      : "node --import tsx tests/browser-server.ts",
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
  },
  reporter: "list",
});
