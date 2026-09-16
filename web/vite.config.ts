/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 开发态代理：前端跑在 5173，后端跑在 8000。全仓没有 CORSMiddleware（P8 硬约束
// 「后端一行不改」），所以跨域只能靠代理绕开，而不是加中间件。
//
// 一个 `/sessions` 前缀条目同时覆盖 REST 与 WS —— Vite 按**路径前缀**匹配，
// `POST /sessions`、`/sessions/{id}/messages`、`GET /sessions/{id}/events` 与
// `GET /sessions/{id}/ws` 的 upgrade 全都命中，不需要 RegExp，也不需要 ws:// 目标
// （target 用 http:// 才会让普通 HTTP 请求也走通）。
const BACKEND = "http://127.0.0.1:8000";
const proxy = {
  "/sessions": { target: BACKEND, changeOrigin: false, ws: true },
};

export default defineConfig({
  plugins: [react()],
  // 产物用绝对路径 /assets/*，正好落在 StaticFiles 挂载的 "/" 上。别改成 "./"。
  base: "/",
  build: {
    // 产物直接落到后端托管的静态目录，`api/app.py` 一行不改。
    outDir: "../api/static",
    // ⚠️ 必须显式写 true：outDir 在 root 之外时 Vite 会**静默跳过**清空并只打一行警告，
    // 于是每次构建都在旧产物上叠加，index.html 引用新 hash、旧 hash 文件永远留着。
    emptyOutDir: true,
  },
  server: { port: 5173, strictPort: true, proxy },
  // preview 让「构建产物 + 真后端」在写进 api/static 之前就能验一遍。
  preview: { port: 4173, strictPort: true, proxy },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
