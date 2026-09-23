import type { Plugin } from "vite";
import { getPayload } from "./service.ts";
export function evidencePlugin(root: string): Plugin {
  return {
    name: "thermo-read-only-evidence",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url?.startsWith("/data/")) return next();
        try {
          const result = await getPayload(root, req.url, req.method);
          res.statusCode = result.status;
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Cache-Control", "no-store");
          res.end(
            req.method === "HEAD" ? undefined : JSON.stringify(result.body),
          );
        } catch {
          res.statusCode = 503;
          res.end(
            JSON.stringify({ error: "Evidence temporarily unavailable" }),
          );
        }
      });
    },
  };
}
