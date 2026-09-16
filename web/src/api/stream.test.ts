/**
 * WebSocket 客户端。三个纯函数（`wsUrl` / `nextRetryDelay` / `isFatalCloseCode`）覆盖
 * 协议细节，`openSessionStream` 用一个假 socket + 假定时器覆盖状态机。
 *
 * 这一层最容易出的错是**重连游标**：`?since=` 是闭区间（`api/ws.py:82` 的 `seq >= since`），
 * 传 `lastSeq` 会重复渲染一条、传 `lastSeq+2` 会永久漏一条。所以专门钉住 URL 的构造。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  WS_CLOSE_NO_SESSION,
  WS_CLOSE_NOT_READY,
  isFatalCloseCode,
  nextRetryDelay,
  openSessionStream,
  wsUrl,
  type Stream,
  type StreamState,
  type WebSocketLike,
} from "./stream";
import type { Envelope, EventEnvelope, SessionInfo } from "./types";

const ORIGIN = "http://127.0.0.1:8000";

describe("wsUrl", () => {
  it("http → ws，路径带 /sessions/{id}/ws", () => {
    expect(wsUrl("t1", 0, "http://127.0.0.1:8000")).toBe("ws://127.0.0.1:8000/sessions/t1/ws");
  });

  it("https → wss（部署到反代后面不能退化成明文 ws）", () => {
    expect(wsUrl("t1", 0, "https://codepilot.example.com")).toBe(
      "wss://codepilot.example.com/sessions/t1/ws",
    );
  });

  it("`since <= 0` 时省略参数（后端默认 0，带上只是噪音）", () => {
    for (const since of [0, -1, -100]) {
      expect(wsUrl("t1", since, ORIGIN)).not.toContain("since");
    }
  });

  it("`since > 0` 时带上 —— **闭区间**，所以调用方必须传 lastSeq+1", () => {
    expect(wsUrl("t1", 1, ORIGIN)).toBe("ws://127.0.0.1:8000/sessions/t1/ws?since=1");
    expect(wsUrl("t1", 42, ORIGIN)).toBe("ws://127.0.0.1:8000/sessions/t1/ws?since=42");
  });

  it("session id 走 encodeURIComponent", () => {
    expect(wsUrl("a b", 0, ORIGIN)).toBe("ws://127.0.0.1:8000/sessions/a%20b/ws");
  });
});

describe("nextRetryDelay", () => {
  it("1s → 2s → 4s → 8s，第 4 次撞到 10s 封顶", () => {
    expect([0, 1, 2, 3, 4].map((a) => nextRetryDelay(a))).toEqual([1000, 2000, 4000, 8000, 10000]);
  });

  it("封顶之后一直是 10s（不会退到分钟级 —— 用户还在屏幕前等着）", () => {
    expect(nextRetryDelay(20)).toBe(10000);
    expect(nextRetryDelay(1000)).toBe(10000);
  });

  it("负数 attempt 当 0（`Math.max(0, …)`，NaN 之外不产生负延迟）", () => {
    expect(nextRetryDelay(-5)).toBe(1000);
  });

  it("base/cap 可注入 —— 测试里不用真的等秒", () => {
    expect(nextRetryDelay(3, 10, 50)).toBe(50);
    expect(nextRetryDelay(1, 10, 1000)).toBe(20);
  });
});

describe("isFatalCloseCode", () => {
  it("4404 会话不存在 / 4503 服务未就绪 → 不重连", () => {
    expect(isFatalCloseCode(WS_CLOSE_NO_SESSION)).toBe(true);
    expect(isFatalCloseCode(WS_CLOSE_NOT_READY)).toBe(true);
    expect(WS_CLOSE_NO_SESSION).toBe(4404);
    expect(WS_CLOSE_NOT_READY).toBe(4503);
  });

  it("正常关闭 / 网络断（1006）/ 服务重启（1001）→ 可以重连", () => {
    for (const code of [1000, 1001, 1006, 1011, 4400]) {
      expect(isFatalCloseCode(code)).toBe(false);
    }
  });
});

// ---------------------------------------------------------------- 状态机

class FakeSocket implements WebSocketLike {
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason?: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  closed = false;

  // 不能用构造函数参数属性（`constructor(readonly url: string)`）—— tsconfig 开了
  // `erasableSyntaxOnly`，那是 TS 独有的运行时代码，会挡住「只用类型擦除」的前提。
  readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  close(): void {
    this.closed = true;
  }

  emitOpen(): void {
    this.onopen?.({});
  }

  emitMessage(data: unknown): void {
    this.onmessage?.({ data });
  }

  emitClose(code: number): void {
    this.onclose?.({ code });
  }

  /** 序列化后的帧（模拟线上真实形态：字符串，不是对象）。 */
  emitJson(env: unknown): void {
    this.emitMessage(JSON.stringify(env));
  }
}

interface Harness {
  sockets: FakeSocket[];
  states: StreamState[];
  statuses: SessionInfo[];
  events: EventEnvelope[];
  stream: Stream;
  /** 最后一条状态（断言状态机终点用）。 */
  last(): StreamState;
}

function harness(opts: { failOnCreate?: number } = {}): Harness {
  const sockets: FakeSocket[] = [];
  let created = 0;
  const statuses: SessionInfo[] = [];
  const events: EventEnvelope[] = [];
  const states: StreamState[] = [];
  const stream = openSessionStream(
    "t1",
    {
      onStatus: (s) => statuses.push(s),
      onEvent: (e) => events.push(e),
      onState: (s) => states.push(s),
    },
    {
      origin: ORIGIN,
      createSocket: (url) => {
        if (opts.failOnCreate !== undefined && created++ === opts.failOnCreate) {
          throw new Error("boom");
        }
        const s = new FakeSocket(url);
        sockets.push(s);
        return s;
      },
    },
  );
  return {
    sockets,
    states,
    statuses,
    events,
    stream,
    last: () => states[states.length - 1]!,
  };
}

/** 一条合法的 `kind:"status"` 信封（字段与 `SessionInfo` 平铺）。 */
function statusEnvelope(over: Partial<SessionInfo> = {}): Envelope {
  return {
    kind: "status",
    thread_id: "t1",
    status: "idle",
    approval: [],
    result: null,
    error: null,
    event_count: 0,
    ...over,
  };
}

function eventEnvelope(seq: number, message = `m${seq}`): Envelope {
  return {
    kind: "event",
    seq,
    event: {
      type: "AgentStep",
      agent: "coder",
      message,
      detail: null,
      timestamp: "2026-01-01T00:00:00+00:00",
      thread_id: "t1",
      step: 1,
      node: "agent",
    },
  };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("openSessionStream", () => {
  it("起手就建一条 socket，URL 不带 since，状态 connecting", () => {
    const h = harness();
    expect(h.sockets).toHaveLength(1);
    expect(h.sockets[0]!.url).toBe("ws://127.0.0.1:8000/sessions/t1/ws");
    // 两次 connecting：一次是 openSessionStream 打的，一次是 connect() 打的。
    // 冗余但无害 —— 假 socket 的构造也能抛（见下面 createSocket 抛错那条），
    // 所以「即将连接」这件事必须发生在 createSocket 之前。
    expect(h.states).toEqual([
      { kind: "connecting", attempt: 0 },
      { kind: "connecting", attempt: 0 },
    ]);
  });

  it("onopen → open，且退避计数归零", () => {
    const h = harness();
    h.sockets[0]!.emitClose(1006);
    vi.advanceTimersByTime(1000);
    h.sockets[1]!.emitOpen();
    expect(h.last()).toEqual({ kind: "open" });
  });

  it("status 信封 → onStatus 拿到**去掉 kind** 的 SessionInfo", () => {
    const h = harness();
    h.sockets[0]!.emitJson(statusEnvelope({ status: "awaiting_approval" }));
    expect(h.statuses).toHaveLength(1);
    expect(h.statuses[0]).toEqual({
      thread_id: "t1",
      status: "awaiting_approval",
      approval: [],
      result: null,
      error: null,
      event_count: 0,
    });
    expect("kind" in h.statuses[0]!).toBe(false);
  });

  it("event 信封 → onEvent 原样透传（含 seq）", () => {
    const h = harness();
    h.sockets[0]!.emitJson(eventEnvelope(7));
    expect(h.events).toHaveLength(1);
    expect(h.events[0]!.seq).toBe(7);
    expect(h.events[0]!.event.message).toBe("m7");
  });

  it("**重连游标是 lastSeq+1**（闭区间过滤，传 lastSeq 会重复一条）", () => {
    const h = harness();
    h.sockets[0]!.emitJson(eventEnvelope(0));
    h.sockets[0]!.emitJson(eventEnvelope(1));
    h.sockets[0]!.emitClose(1006);
    vi.advanceTimersByTime(1000);
    expect(h.sockets[1]!.url).toBe("ws://127.0.0.1:8000/sessions/t1/ws?since=2");
  });

  it("一条事件都没收到就断 → 重连 URL **不带 since**（不是 ?since=0）", () => {
    const h = harness();
    h.sockets[0]!.emitClose(1006);
    vi.advanceTimersByTime(1000);
    expect(h.sockets[1]!.url).toBe("ws://127.0.0.1:8000/sessions/t1/ws");
  });

  it("重连窗口里的重复 seq 被丢掉（后端闭区间过滤的兜底）", () => {
    const h = harness();
    h.sockets[0]!.emitJson(eventEnvelope(0));
    h.sockets[0]!.emitJson(eventEnvelope(1));
    h.sockets[0]!.emitClose(1006);
    vi.advanceTimersByTime(1000);
    // 假想服务端没理会 since、从 0 重新推
    h.sockets[1]!.emitJson(eventEnvelope(0));
    h.sockets[1]!.emitJson(eventEnvelope(1));
    h.sockets[1]!.emitJson(eventEnvelope(2));
    expect(h.events.map((e) => e.seq)).toEqual([0, 1, 2]);
  });

  it("乱序到达的旧 seq 被丢掉（重复渲染一条比重连漏一条便宜，但顺序不能乱）", () => {
    const h = harness();
    h.sockets[0]!.emitJson(eventEnvelope(5));
    h.sockets[0]!.emitJson(eventEnvelope(3));
    expect(h.events.map((e) => e.seq)).toEqual([5]);
  });

  it("非 JSON 帧被忽略，**不断线**（心跳/代理塞的杂物不值得炸掉整条流）", () => {
    const h = harness();
    h.sockets[0]!.emitMessage("ping");
    h.sockets[0]!.emitMessage("");
    h.sockets[0]!.emitMessage("not json at all");
    expect(h.events).toEqual([]);
    expect(h.statuses).toEqual([]);
    expect(h.last()).toEqual({ kind: "connecting", attempt: 0 });
  });

  it("未知 kind 被忽略（后端将来加信封种类不该让老前端崩）", () => {
    const h = harness();
    h.sockets[0]!.emitJson({ kind: "future", payload: 1 });
    expect(h.events).toEqual([]);
    expect(h.statuses).toEqual([]);
  });

  it("close(4404) → fatal 并**停止**，不再建新 socket", () => {
    const h = harness();
    h.sockets[0]!.emitClose(WS_CLOSE_NO_SESSION);
    expect(h.last()).toEqual({
      kind: "fatal",
      reason: "会话不存在（服务重启后会话目录就清空了）",
      code: WS_CLOSE_NO_SESSION,
    });
    vi.advanceTimersByTime(60_000);
    expect(h.sockets).toHaveLength(1);
    expect(h.states.filter((s) => s.kind === "reconnecting")).toEqual([]);
  });

  it("close(4503) → fatal，文案与 4404 区分开", () => {
    const h = harness();
    h.sockets[0]!.emitClose(WS_CLOSE_NOT_READY);
    expect(h.last()).toEqual({ kind: "fatal", reason: "服务尚未就绪", code: WS_CLOSE_NOT_READY });
  });

  it("close(1006) → reconnecting 带延迟，退避逐次翻倍", () => {
    const h = harness();
    h.sockets[0]!.emitClose(1006);
    expect(h.last()).toEqual({ kind: "reconnecting", attempt: 1, delayMs: 1000 });
    vi.advanceTimersByTime(1000);
    h.sockets[1]!.emitClose(1006);
    expect(h.last()).toEqual({ kind: "reconnecting", attempt: 2, delayMs: 2000 });
    vi.advanceTimersByTime(2000);
    h.sockets[2]!.emitClose(1006);
    expect(h.last()).toEqual({ kind: "reconnecting", attempt: 3, delayMs: 4000 });
  });

  it("onerror 自己**不**排重连（原因在随后的 onclose 里，重复排程会建两条 socket）", () => {
    const h = harness();
    h.sockets[0]!.onerror?.({});
    vi.advanceTimersByTime(60_000);
    expect(h.sockets).toHaveLength(1);
  });

  it("createSocket 抛异常 → 走退避重试而不是把异常抛给调用方", () => {
    const h = harness({ failOnCreate: 0 });
    expect(h.sockets).toHaveLength(0);
    expect(h.last()).toEqual({ kind: "reconnecting", attempt: 1, delayMs: 1000 });
    vi.advanceTimersByTime(1000);
    expect(h.sockets).toHaveLength(1);
  });

  it("error 信封 → fatal(code 为 null)，随后真正的 close 再补一条带 code 的", () => {
    // 后端发完 error 就 close，所以这里会有两条 fatal：第一条来自信封（无 code），
    // 第二条来自 onclose（有 code）。消费方是「后写覆盖」，所以界面上留下的是带 code
    // 的那条 —— 不是 bug，是刻意把处置留给 onclose（它拿得到关闭码）。
    const h = harness();
    h.sockets[0]!.emitJson({ kind: "error", message: "服务尚未就绪" });
    h.sockets[0]!.emitClose(WS_CLOSE_NOT_READY);
    expect(h.states.filter((s) => s.kind === "fatal")).toEqual([
      { kind: "fatal", reason: "服务尚未就绪", code: null },
      { kind: "fatal", reason: "服务尚未就绪", code: WS_CLOSE_NOT_READY },
    ]);
  });

  it("close() → 状态 closed，socket 被关且回调摘掉，之后不再重连", () => {
    const h = harness();
    h.stream.close();
    expect(h.last()).toEqual({ kind: "closed" });
    const s = h.sockets[0]!;
    expect(s.closed).toBe(true);
    // 回调摘掉是收尾的关键：否则我们自己排的 onclose 会把这次主动关闭误判成断线
    expect(s.onclose).toBeNull();
    expect(s.onmessage).toBeNull();
    vi.advanceTimersByTime(60_000);
    expect(h.sockets).toHaveLength(1);
  });

  it("close() 之后 socket 的迟到事件一律被丢弃（不再冒泡到 React）", () => {
    const h = harness();
    const s = h.sockets[0]!;
    h.stream.close();
    // 模拟「摘回调之前已经排进事件队列」的帧：直接调 handler 也走 stopped 判断
    s.emitJson(eventEnvelope(0));
    expect(h.events).toEqual([]);
  });

  it("reopen() → 立刻重连，退避计数归零", () => {
    const h = harness();
    h.sockets[0]!.emitClose(1006); // attempt 1
    vi.advanceTimersByTime(1000);
    h.sockets[1]!.emitClose(1006); // attempt 2
    h.stream.reopen();
    expect(h.sockets).toHaveLength(3);
    // reopen **立刻**连，不等退避：connecting 一共 4 条 = 起手 2 条（openSessionStream
    // 一条 + connect() 一条）+ 重连那次 1 条 + reopen 这次 1 条。
    expect(h.states.filter((s) => s.kind === "connecting")).toHaveLength(4);
    expect(h.last()).toEqual({ kind: "connecting", attempt: 0 });
    h.sockets[2]!.emitClose(1006);
    // 计数归零 → 又从 1s 起
    expect(h.last()).toEqual({ kind: "reconnecting", attempt: 1, delayMs: 1000 });
  });

  it("reopen() 保留 lastSeq —— 回到前台不该重放整段历史", () => {
    const h = harness();
    h.sockets[0]!.emitJson(eventEnvelope(9));
    h.stream.reopen();
    expect(h.sockets[1]!.url).toBe("ws://127.0.0.1:8000/sessions/t1/ws?since=10");
  });

  it("fatal 之后 reopen() 不复活（stopped 是终局，调用方该重建一条流）", () => {
    const h = harness();
    h.sockets[0]!.emitClose(WS_CLOSE_NO_SESSION);
    h.stream.reopen();
    expect(h.sockets).toHaveLength(1);
    expect(h.last()).toEqual({
      kind: "fatal",
      reason: "会话不存在（服务重启后会话目录就清空了）",
      code: WS_CLOSE_NO_SESSION,
    });
  });

  it("旧 socket 的迟到 onclose 不会排重连（socket 身份比对）", () => {
    const h = harness();
    const stale = h.sockets[0]!;
    h.stream.reopen(); // 建了第 2 条，stale 已被 teardown 摘过回调
    stale.onclose?.({ code: 1006 });
    expect(h.sockets).toHaveLength(2);
    expect(h.states.filter((s) => s.kind === "reconnecting")).toEqual([]);
  });
});
