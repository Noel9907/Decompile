import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the Vite server proxies /api to the FastAPI backend (python -m decompile_web, port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
  build: { outDir: "dist", sourcemap: false },
});
