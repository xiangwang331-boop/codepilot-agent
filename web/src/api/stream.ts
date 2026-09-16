/**
 * WebSocket 订阅客户端。**不含 React**，也不 import 任何 React —— 纯副作用边界，
 * 逻辑部分（URL、退避、关闭码）都是可单测的纯函数。
 *
 * ## 契约要点（对着 `api/ws.py` 核过）
 *
 * - 连上**第一条**必是 `kind:"status"`，然后是整段 `kind:"event"` 回填，再之后实时。
 * - `?since=<n>` 的过滤是 **`seq >= since`（闭区间）**（`api/ws.py:82`）。所以断线重连
 *   要传 `lastSeq + 1`，**并且**客户端自己再丢一次 `seq <= lastSeq` 兜底 ——
 *   重复渲染一条事件比重连时漏一条便宜得多。
 * - 关闭码 4404 = 会话不存在（服务重启后内存目录就没了，重连一万次也没用）、
 *   4503 = 服务未就绪（可以重试）。**这两种不再重连**，其余（1006 网络断等）指数退避。
 *
 * ## 为什么 `lastSeq` 是模块内的可变状态而不是 React state
 *
 * 它一变就重连会是个灾难（每收到一条事件就重建一次 socket）。它只服务于
 * 「重连时从哪继续」，所以留在闭包里。
 */
import type { Envelope, EventEnvelope, SessionInfo, StatusEnvelope } from "./types";

/** 应用自定义关闭码，与 `api/ws.py:48-49` 对齐。 */
export const WS_CLOSE_NO_SESSION = 4404;
export const WS_CLOSE_NOT_READY = 4503;

/** WebSocket 的最小接口，方便测试注入假 socket（Node 里没有 DOM 的 WebSocket 类型可用）。 */
export interface WebSocketLike {
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason?: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  close(code?: number, reason?: string): void;
}

export type StreamState =
  | { kind: "connecting"; attempt: number }
  | { kind: "open" }
  | { kind: "reconnecting"; attempt: number; delayMs: number }
  | { kind: "fatal"; reason: string; code: number | null }
  | { kind: "closed" };

export interface StreamHandlers {
  onStatus(session: SessionInfo): void;
  onEvent(envelope: EventEnvelope): void;
  onState(state: StreamState): void;
}

export interface Stream {
  /** 主动关闭：不会再重连，状态变 `closed`。 */
  close(): void;
  /** 立刻重连（回到可见时用），退避计数重置。 */
  reopen(): void;
}

export interface StreamDeps {
  createSocket?: (url: string) => WebSocketLike;
  origin?: string;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

/**
 * 拼 WS URL。**纯函数。**
 *
 * `since <= 0` 时省略参数（后端默认就是 0，省掉更干净）。
 * 用 `location.host` 而不是写死后端地址 —— 开发态走 Vite 代理、生产态同源，
 * 同一份代码两边都对。
 */
export function wsUrl(sessionId: string, since: number, origin: string): string {
  const ws = origin.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  const base = `${ws}/sessions/${encodeURIComponent(sessionId)}/ws`;
  return since > 0 ? `${base}?since=${since}` : base;
}

/** 指数退避：`base * 2^attempt`，封顶 `cap`。**纯函数。** */
export function nextRetryDelay(attempt: number, base = 1000, cap = 10000): number {
  const exp = base * Math.pow(2, Math.max(0, attempt));
  return Math.min(exp, cap);
}

/** 这两种关闭码是「重连也没用」的终局。**纯函数。** */
export function isFatalCloseCode(code: number): boolean {
  return code === WS_CLOSE_NO_SESSION || code === WS_CLOSE_NOT_READY;
}

function defaultCreateSocket(url: string): WebSocketLike {
  return new WebSocket(url) as unknown as WebSocketLike;
}

/** 打开一条会话事件流。返回的句柄可以 `close()` / `reopen()`。 */
export function openSessionStream(
  sessionId: string,
  handlers: StreamHandlers,
  deps: StreamDeps = {},
): Stream {
  const createSocket = deps.createSocket ?? defaultCreateSocket;
  const origin =
    deps.origin ?? (typeof location !== "undefined" ? location.origin : "http://127.0.0.1:8000");
  const baseDelay = deps.baseDelayMs ?? 1000;
  const maxDelay = deps.maxDelayMs ?? 10000;

  let socket: WebSocketLike | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  /** 已收到的最大 seq；-1 表示一条都还没有。重连游标 = lastSeq + 1。 */
  let lastSeq = -1;
  let stopped = false;

  const teardown = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (socket !== null) {
      const s = socket;
      socket = null;
      // 摘掉回调再关，否则我们自己的 onclose 会把这段收尾当成「断线」再排一次重连。
      s.onopen = null;
      s.onmessage = null;
      s.onclose = null;
      s.onerror = null;
      try {
        s.close();
      } catch {
        // 已经关掉的 socket 再 close 会抛，忽略
      }
    }
  };

  const scheduleReconnect = (): void => {
    if (stopped) return;
    const delay = nextRetryDelay(attempt, baseDelay, maxDelay);
    attempt += 1;
    handlers.onState({ kind: "reconnecting", attempt, delayMs: delay });
    timer = setTimeout(connect, delay);
  };

  const fail = (reason: string, code: number | null): void => {
    stopped = true;
    teardown();
    handlers.onState({ kind: "fatal", reason, code });
  };

  const handle = (raw: unknown): void => {
    let env: Envelope;
    try {
      env = JSON.parse(typeof raw === "string" ? raw : String(raw)) as Envelope;
    } catch {
      return; // 非 JSON 帧：忽略，不值得为它断线
    }

    if (env.kind === "status") {
      const { kind: _kind, ...session } = env as StatusEnvelope;
      handlers.onStatus(session as SessionInfo);
      return;
    }
    if (env.kind === "event") {
      // 闭区间过滤 + 本端去重：`?since=lastSeq+1` 已经排掉了旧的，
      // 但重连窗口里可能重复投递，这里再兜一层。
      if (env.seq <= lastSeq) return;
      lastSeq = env.seq;
      handlers.onEvent(env);
      return;
    }
    if (env.kind === "error") {
      // 服务端发完 error 就会 close(4404/4503)，真正的处置在 onclose 里。
      handlers.onState({ kind: "fatal", reason: env.message, code: null });
    }
  };

  function connect(): void {
    if (stopped) return;
    handlers.onState({ kind: "connecting", attempt });
    const url = wsUrl(sessionId, lastSeq + 1, origin);
    let s: WebSocketLike;
    try {
      s = createSocket(url);
    } catch (e) {
      scheduleReconnect();
      return;
    }
    socket = s;

    s.onopen = () => {
      if (stopped || socket !== s) return;
      attempt = 0; // 连上了就重置退避
      handlers.onState({ kind: "open" });
    };

    s.onmessage = (ev) => {
      if (stopped || socket !== s) return;
      handle(ev.data);
    };

    s.onerror = () => {
      // 具体原因在随后的 onclose 里（code 更有信息量），这里不处理避免重复排程。
    };

    s.onclose = (ev) => {
      if (stopped || socket !== s) return;
      socket = null;
      const code = typeof ev?.code === "number" ? ev.code : 1006;
      if (isFatalCloseCode(code)) {
        fail(
          code === WS_CLOSE_NO_SESSION
            ? "会话不存在（服务重启后会话目录就清空了）"
            : "服务尚未就绪",
          code,
        );
        return;
      }
      scheduleReconnect();
    };
  }

  handlers.onState({ kind: "connecting", attempt: 0 });
  connect();

  return {
    close(): void {
      stopped = true;
      teardown();
      handlers.onState({ kind: "closed" });
    },
    reopen(): void {
      if (stopped) return; // 主动关掉的流不复用，调用方重建一个
      teardown();
      attempt = 0;
      connect();
    },
  };
}
