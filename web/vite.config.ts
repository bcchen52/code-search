import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs separately (`cqa serve`, port 8000 by default);
// the dev server forwards the API's paths to it, so the browser talks to one origin.
const api = process.env.CQA_API ?? "http://127.0.0.1:8000";
const apiPaths = ["/repos", "/jobs", "/queries", "/metrics", "/eval"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiPaths.map((path) => [path, { target: api, changeOrigin: true }])),
  },
});
