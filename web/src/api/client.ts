/**
 * REST 封装 + **FastAPI 四种错误形状的唯一归一化点**。
 *
 * 后端直接用 `HTTPException(status_code, detail=...)`，而 `detail` 的**类型随状态码变化**
 * （`api/routes.py` 实测）：
 *
 * | 状态 | detail 形状 | 出处 |
 * |---|---|---|
 * | 404 | 字符串 `"会话 x 不存在"` | `routes.py:61,114` |
 * | 409（重名） | 字符串 | `routes.py:91` 的 `str(SessionExistsError)` |
 * | 409（忙） | **对象** `{status, message}` | `routes.py:65-71` |
 * | 422 | **数组** `[{loc,msg,type}, …]` | FastAPI 校验 |
 * | 500 | 字符串（含被移除但删库失败的原因） | `routes.py:135-138`（P9） |
 * | 503 | 字符串 | `routes.py:51` |
 *
 * 界面上任何一处直接用 `detail` 都会在某个状态码下渲染成 `[object Object]`。
 * 所以**只有这里**碰错误体，其余地方一律拿 `ApiError.message`。
 */
import type { SessionInfo, SessionList } from "./types";

export class ApiError extends Error {
  readonly status: number;
  /**
   * 错误体里的 **`detail` 字段本身**，不是整个 body。
   *
   * ⚠️ 这一层之差是必须的：409 忙的 body 是 `{"detail": {"status": …, "message": …}}`，
   * 存整个 body 的话 `busyStatus` 就去读 `body.status`（不存在）→ 永远是 null，
   * 那个分支会**静默失效**（不报错、不崩，只是「忙」被当成「重名」处理）。
   */
  readonly detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** 409 忙：`detail` 是 `{status, message}`，据此判断该显示什么控件。 */
  get busyStatus(): string | null {
    const d = this.detail;
    if (d !== null && typeof d === "object" && !Array.isArray(d) && "status" in d) {
      const s = (d as { status: unknown }).status;
      return typeof s === "string" ? s : null;
    }
    return null;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/**
 * 把错误体归一化成一句人能读的话。**纯函数，是 `client.test.ts` 的主要目标。**
 */
export function normalizeErrorBody(status: number, body: unknown): string {
  const detail = isRecord(body) ? body.detail : undefined;

  if (typeof detail === "string" && detail.trim() !== "") return detail;

  // 409 忙：{status, message}
  if (isRecord(detail)) {
    const msg = detail.message;
    if (typeof msg === "string" && msg.trim() !== "") return msg;
    return JSON.stringify(detail);
  }

  // 422：FastAPI 校验错误数组
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (!isRecord(item)) return String(item);
        const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
        const msg = typeof item.msg === "string" ? item.msg : JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter((s) => s !== "");
    if (parts.length > 0) return `请求参数有误 —— ${parts.join("；")}`;
  }

  // 空体 / 不透明体：退回状态码本身
  return `请求失败（HTTP ${status}）`;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }

  // 路径一律相对 —— 开发态由 Vite 代理转发（见 vite.config.ts），生产态同源。
  // 全程没有 CORS，也不需要 VITE_API_BASE 这类环境变量。
  const res = await fetch(path, init);

  if (res.status === 204) return undefined as T;

  let parsed: unknown = null;
  const text = await res.text();
  if (text !== "") {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }

  if (!res.ok) {
    // 存进去的是 `body.detail`（见 `ApiError.detail` 的注释），不是 `parsed`。
    const detail = isRecord(parsed) ? (parsed.detail ?? null) : null;
    throw new ApiError(res.status, normalizeErrorBody(res.status, parsed), detail);
  }
  return parsed as T;
}

// ---------------------------------------------------------------- 端点

export function listSessions(): Promise<SessionList> {
  return request<SessionList>("GET", "/sessions");
}

export function getSession(threadId: string): Promise<SessionInfo> {
  return request<SessionInfo>("GET", `/sessions/${encodeURIComponent(threadId)}`);
}

/** 新建会话。不传 `threadId` 就让后端生成 UUID。 */
export function createSession(threadId?: string): Promise<SessionInfo> {
  return request<SessionInfo>(
    "POST",
    "/sessions",
    threadId ? { thread_id: threadId } : undefined,
  );
}

/** 下发指令。后端返回 **202** 与最新的 SessionInfo（任务是后台线程跑的）。 */
export function sendMessage(threadId: string, task: string): Promise<SessionInfo> {
  return request<SessionInfo>(
    "POST",
    `/sessions/${encodeURIComponent(threadId)}/messages`,
    { task },
  );
}

/** 批准/拒绝挂起中的委派。 */
export function sendApproval(threadId: string, approved: boolean): Promise<SessionInfo> {
  return request<SessionInfo>(
    "POST",
    `/sessions/${encodeURIComponent(threadId)}/approval`,
    { approved },
  );
}

export function deleteSession(threadId: string): Promise<void> {
  return request<void>("DELETE", `/sessions/${encodeURIComponent(threadId)}`);
}

// 刻意**没有** `getEvents()`：WS 连上就推整段回填（还带 `?since=`），
// REST 的 `/sessions/{id}/events` 是给 CLI 与冒烟脚本用的，前端调它就是死代码。
