import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { evidencePlugin } from "./server/plugin.ts";
export default defineConfig({
  base: "./",
  plugins: [react(), evidencePlugin(resolve(import.meta.dirname, ".."))],
  server: { host: "0.0.0.0", allowedHosts: ["terminal.local"], port: 5173 },
  preview: { host: "127.0.0.1", port: 4173 },
});
