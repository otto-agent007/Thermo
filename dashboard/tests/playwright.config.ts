import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";
export default defineConfig({
  testDir: ".",
  testMatch: "browser.spec.ts",
  workers: 1,
  use: { baseURL: "http://127.0.0.1:5174", headless: true },
  webServer: {
    command: "node --import tsx tests/browser-server.ts",
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    url: "http://127.0.0.1:5174",
    reuseExistingServer: false,
  },
  reporter: "list",
});
