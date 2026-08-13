import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Inside the docker network the API is reachable at http://api:8000,
// in local development — at http://localhost:8000
const apiTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
});
