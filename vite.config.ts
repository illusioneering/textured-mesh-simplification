import { defineConfig } from "vite";

export default defineConfig({
  base: "/textured-mesh-simplification/",
  optimizeDeps: {
    exclude: ["watlas"],
  },
  server: {
    host: "0.0.0.0",
  },
});
