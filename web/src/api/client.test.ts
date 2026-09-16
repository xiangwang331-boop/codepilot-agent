/**
 * REST 封装 —— 重点是**四种错误形状的唯一归一化点**（`normalizeErrorBody`）。
 *
 * 这几条形状是实测的（`tests/test_web_ui_contract.py` 在后端侧同样钉了一遍，
 * 见 `test_error_shapes_match_client_ts`）：形状一变两边一起红，而不是等到界面上
 * 渲染出 `[object Object]` 才发现。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  createSession,
  deleteSession,
  getSession,
  listSessions,
  normalizeErrorBody,
  sendApproval,
  sendMessage,
} from "./client";

describe("normalizeErrorBody", () => {
  it("404 / 409重名 / 503：detail 是字符串 → 原样用", () => {
    expect(normalizeErrorBody(404, { detail: "会话 nope 不存在" })).toBe("会话 nope 不存在");
    expect(normalizeErrorBody(503, { detail: "服务正在启动" })).toBe("服务正在启动");
  });

  it("409 忙：detail 是对象 `{status, message}` → 取 message（不是 [object Object]）", () => {
    expect(
      normalizeErrorBody(409, { detail: { status: "awaiting_approval", message: "会话正等着批准" } }),
    ).toBe("会话正等着批准");
  });

  it("对象 detail 但没有 message → 退化成 JSON，至少能看见内容", () => {
    expect(normalizeErrorBody(409, { detail: { status: "running" } })).toBe('{"status":"running"}');
  });

  it("422：数组 → 拼成人话，`body` 这一段 loc 前缀被去掉", () => {
    const body = {
      detail: [
        { loc: ["body", "task"], msg: "Field required", type: "missing" },
        { loc: ["body", "task"], msg: "String should have at least 1 character", type: "too_short" },
      ],
    };
    expect(normalizeErrorBody(422, body)).toBe(
      "请求参数有误 —— task: Field required；task: String should have at least 1 character",
    );
  });

  it("422 里 loc 只有 body（路径为空）→ 只留 msg，不留一个孤零零的冒号", () => {
    const body = { detail: [{ loc: ["body"], msg: "整体不对", type: "value_error" }] };
    expect(normalizeErrorBody(422, body)).toBe("请求参数有误 —— 整体不对");
  });

  it("422 数组里混着非对象项 → 兜底成字符串，不崩", () => {
    expect(normalizeErrorBody(422, { detail: ["裸字符串"] })).toBe("请求参数有误 —— 裸字符串");
  });

  it("空数组 / 全空项的数组 → 退回状态码（不产出「请求参数有误 —— 」这种半截话）", () => {
    expect(normalizeErrorBody(422, { detail: [] })).toBe("请求失败（HTTP 422）");
    expect(normalizeErrorBody(422, { detail: [{ loc: ["body"], msg: "" }] })).toBe(
      "请求失败（HTTP 422）",
    );
  });

  it("空字符串 detail → 当没有 detail（否则界面弹一个空白错误框）", () => {
    expect(normalizeErrorBody(500, { detail: "   " })).toBe("请求失败（HTTP 500）");
  });

  it("没有 detail / body 不是对象 / body 是 null → 退回状态码", () => {
    expect(normalizeErrorBody(500, {})).toBe("请求失败（HTTP 500）");
    expect(normalizeErrorBody(500, "纯文本响应")).toBe("请求失败（HTTP 500）");
    expect(normalizeErrorBody(502, null)).toBe("请求失败（HTTP 502）");
  });

  it("永远不会返回空串（界面上空白错误框比没有错误框更糟）", () => {
    const bodies = [null, undefined, {}, [], "", 0, { detail: null }, { detail: "" }];
    for (const b of bodies) {
      expect(normalizeErrorBody(418, b).trim()).not.toBe("");
    }
  });
});

describe("ApiError.busyStatus", () => {
  // ⚠️ 构造函数的第三个参数是 **`body.detail`**，不是整个 body（见 `client.ts` 的
  // `ApiError.detail` 注释）。写错一层这些用例会全绿、线上 `busyStatus` 全 null。
  it("对象 detail 带 status → 取出来（前端据它决定转圈还是弹批准按钮）", () => {
    const e = new ApiError(409, "忙", { status: "awaiting_approval", message: "x" });
    expect(e.busyStatus).toBe("awaiting_approval");
  });

  it("字符串 / 数组 / null detail → null（不是不认识的 409 别硬猜）", () => {
    expect(new ApiError(409, "重名", "会话已存在").busyStatus).toBeNull();
    expect(new ApiError(422, "参数", []).busyStatus).toBeNull();
    expect(new ApiError(500, "炸了", null).busyStatus).toBeNull();
  });

  it("status 不是字符串 → null（不把对象渲染进 className）", () => {
    expect(new ApiError(409, "x", { status: 42 }).busyStatus).toBeNull();
  });

  it("是 Error 的子类，message 与 status/detail 都在", () => {
    const e = new ApiError(404, "会话不存在", "会话不存在");
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("ApiError");
    expect(e.status).toBe(404);
    expect(e.detail).toBe("会话不存在");
  });

  it("`detail` 存的是 body.detail **这一层**，不是整个 body（错一层 busyStatus 就永远 null）", () => {
    // 这条由 request 层端到端钉住（见下面的「409 忙」），这里只钉构造语义
    const e = new ApiError(409, "x", { status: "running", message: "x" });
    expect(e.busyStatus).toBe("running");
  });
});

// ---------------------------------------------------------------- 请求层

interface Call {
  url: string;
  init: RequestInit;
}

function fakeFetch(
  status: number,
  body: unknown,
  { text }: { text?: string } = {},
): { calls: Call[]; fetch: typeof fetch } {
  const calls: Call[] = [];
  const impl = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => (text !== undefined ? text : body === undefined ? "" : JSON.stringify(body)),
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return { calls, fetch: impl };
}

let original: typeof globalThis.fetch;

beforeEach(() => {
  original = globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = original;
  vi.restoreAllMocks();
});

describe("request 层", () => {
  it("GET 成功 → 返回解析后的 JSON，且不带 body", async () => {
    const f = fakeFetch(200, { sessions: [] });
    globalThis.fetch = f.fetch;
    expect(await listSessions()).toEqual({ sessions: [] });
    expect(f.calls[0]!.init.method).toBe("GET");
    expect(f.calls[0]!.init.body).toBeUndefined();
  });

  it("204 → undefined（DELETE 的空响应不能去 JSON.parse）", async () => {
    globalThis.fetch = fakeFetch(204, undefined).fetch;
    await expect(deleteSession("t1")).resolves.toBeUndefined();
  });

  it("POST 带 body → JSON 序列化 + Content-Type", async () => {
    const f = fakeFetch(202, { thread_id: "t1", status: "running" });
    globalThis.fetch = f.fetch;
    await sendMessage("t1", "写快排");
    expect(f.calls[0]!.init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(f.calls[0]!.init.body))).toEqual({ task: "写快排" });
  });

  it("不传 threadId 的 POST /sessions → **完全没有 body**（让后端自己生成 UUID）", async () => {
    const f = fakeFetch(201, { thread_id: "gen" });
    globalThis.fetch = f.fetch;
    await createSession();
    expect(f.calls[0]!.init.body).toBeUndefined();
    expect(f.calls[0]!.init.headers).toBeUndefined();
  });

  it("thread_id 走 encodeURIComponent（后端生成的 UUID 是十六进制，但不押注这一点）", async () => {
    const f = fakeFetch(200, {});
    globalThis.fetch = f.fetch;
    await getSession("a/b c");
    expect(f.calls[0]!.url).toBe("/sessions/a%2Fb%20c");
  });

  it("**路径是相对的** —— 开发态靠 Vite 代理、生产态同源，全程没有 CORS 配置", async () => {
    const f = fakeFetch(200, {});
    globalThis.fetch = f.fetch;
    await listSessions();
    expect(f.calls[0]!.url.startsWith("/")).toBe(true);
    expect(f.calls[0]!.url).not.toContain("http");
  });

  it("HTTP 错误 → 抛 ApiError，status 与归一化后的 message 都对", async () => {
    globalThis.fetch = fakeFetch(404, { detail: "会话 nope 不存在" }).fetch;
    await expect(getSession("nope")).rejects.toThrowError(
      expect.objectContaining({ name: "ApiError", status: 404, message: "会话 nope 不存在" }),
    );
  });

  it("错误体不是 JSON → 退回状态码，而不是抛一个 JSON 解析错误把真相盖掉", async () => {
    globalThis.fetch = fakeFetch(502, undefined, { text: "<html>Bad Gateway</html>" }).fetch;
    await expect(listSessions()).rejects.toThrowError("请求失败（HTTP 502）");
  });

  it("错误体是空串 → 同样退回状态码", async () => {
    globalThis.fetch = fakeFetch(500, undefined, { text: "" }).fetch;
    await expect(listSessions()).rejects.toThrowError("请求失败（HTTP 500）");
  });

  it("409 忙 → ApiError.busyStatus 拿得到状态（批准按钮 vs 转圈的分叉点）", async () => {
    // 真实 body 形状：`{"detail": {"status": …, "message": …}}`（`routes.py:65-71`）。
    // 这一条是端到端的：走完 fetch → normalize → ApiError，盯住**嵌套层级**没错位。
    globalThis.fetch = fakeFetch(409, {
      detail: { status: "awaiting_approval", message: "会话正等着批准" },
    }).fetch;
    const err = await sendMessage("t1", "再来一个").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).busyStatus).toBe("awaiting_approval");
    expect((err as ApiError).detail).toEqual({
      status: "awaiting_approval",
      message: "会话正等着批准",
    });
    expect((err as ApiError).message).toBe("会话正等着批准");
  });

  it("409 重名（字符串 detail）→ busyStatus 是 null，不能被当成「忙」", async () => {
    globalThis.fetch = fakeFetch(409, { detail: "会话 t1 已存在" }).fetch;
    const err = (await createSession("t1").catch((e: unknown) => e)) as ApiError;
    expect(err.busyStatus).toBeNull();
    expect(err.message).toBe("会话 t1 已存在");
  });

  it("送审批也是一个普通 POST（body 是 {approved}）", async () => {
    const f = fakeFetch(200, { thread_id: "t1", status: "running" });
    globalThis.fetch = f.fetch;
    await sendApproval("t1", false);
    expect(f.calls[0]!.url).toBe("/sessions/t1/approval");
    expect(JSON.parse(String(f.calls[0]!.init.body))).toEqual({ approved: false });
  });
});
