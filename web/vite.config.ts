import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Внутри docker-сети API доступен как http://api:8000,
// при локальной разработке — http://localhost:8000
const apiTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
});
