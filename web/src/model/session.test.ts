/**
 * 会话状态判定 / 轮询节奏 / 首屏选中。
 *
 * 这一层看着琐碎，但 `isBusy` 是**唯一**决定「能不能下发指令 / 该不该显示批准按钮」的
 * 判据。LangGraph 会静默吞掉挂起中的 interrupt，所以后端**绝不能**
 * 靠「worker 线程是否活着」判忙，前端同理 —— 只能看 `status` 字段。
 */
import { describe, expect, it } from "vitest";

import type { SessionInfo, SessionStatus } from "../api/types";
import {
  canApprove,
  canSend,
  isBusy,
  isInterrupted,
  isTerminal,
  pickInitialSession,
  pollDelayMs,
  resultText,
  statusLabel,
  withLiveStatus,
} from "./session";

const ALL: SessionStatus[] = [
  "idle",
  "running",
  "awaiting_approval",
  "interrupted",
  "closed",
];

function info(thread_id: string, status: SessionStatus): SessionInfo {
  return { thread_id, status, approval: [], result: null, error: null, event_count: 0 };
}

describe("五态判定", () => {
  it("isBusy 只认 running 与 awaiting_approval", () => {
    expect(ALL.filter(isBusy)).toEqual(["running", "awaiting_approval"]);
  });

  it("canSend 只认 idle —— `awaiting_approval` 下发指令会吃 409", () => {
    expect(ALL.filter(canSend)).toEqual(["idle"]);
  });

  it("canApprove 只认 awaiting_approval", () => {
    expect(ALL.filter(canApprove)).toEqual(["awaiting_approval"]);
  });

  it("isTerminal 只认 closed", () => {
    expect(ALL.filter(isTerminal)).toEqual(["closed"]);
  });

  it("isInterrupted 只认 interrupted", () => {
    expect(ALL.filter(isInterrupted)).toEqual(["interrupted"]);
  });

  it("穷举五态的真值表（`awaiting_approval` 同时是「忙」与「可批准」）", () => {
    // 用对象而不是数组：真值表是**按状态查**的，不该依赖 ALL 的排列顺序
    const table = Object.fromEntries(
      ALL.map((s) => [
        s,
        {
          busy: isBusy(s),
          send: canSend(s),
          approve: canApprove(s),
          terminal: isTerminal(s),
          cut: isInterrupted(s),
        },
      ]),
    );
    expect(table).toEqual({
      idle: { busy: false, send: true, approve: false, terminal: false, cut: false },
      running: { busy: true, send: false, approve: false, terminal: false, cut: false },
      // ⚠️ 挂起态**既是忙也是可批准** —— 这两个谓词**不互斥**，别按互斥去写。
      // 写成互斥就会在等批准时把「发送」按钮放出来，点下去必然 409。
      awaiting_approval: { busy: true, send: false, approve: true, terminal: false, cut: false },
      // ⚠️ P9 第五态的关键一行：**既不忙也不能发**（后端 `begin()` 只接受 IDLE）。
      // 把它错算成 busy 会让侧栏永远 2 秒轮询一个状态再也不会变的会话（`pollDelayMs`）。
      interrupted: { busy: false, send: false, approve: false, terminal: false, cut: true },
      closed: { busy: false, send: false, approve: false, terminal: true, cut: false },
    });
  });

  it("statusLabel 五态都有中文，未注册值原样返回（不显示 undefined）", () => {
    expect(ALL.map(statusLabel)).toEqual(["空闲", "运行中", "待批准", "已中断", "已关闭"]);
    expect(statusLabel("weird" as SessionStatus)).toBe("weird");
  });
});

describe("pollDelayMs", () => {
  it("全闲 → 慢轮询（10s）", () => {
    expect(pollDelayMs([info("a", "idle"), info("b", "closed")])).toBe(10000);
  });

  it("`interrupted` 不算忙 → 慢轮询（状态再也不会变，快轮是纯浪费）", () => {
    expect(pollDelayMs([info("a", "interrupted"), info("b", "idle")])).toBe(10000);
  });

  it("有一个忙 → 快轮询（2s）", () => {
    expect(pollDelayMs([info("a", "idle"), info("b", "running")])).toBe(2000);
    expect(pollDelayMs([info("a", "awaiting_approval")])).toBe(2000);
  });

  it("空列表 → 慢轮询（不能因为空就狂打后端）", () => {
    expect(pollDelayMs([])).toBe(10000);
  });
});

describe("pickInitialSession", () => {
  const sessions = [info("s1", "idle"), info("s2", "running")];

  it("URL 有 ?session= 且存在 → 选它", () => {
    expect(pickInitialSession(sessions, "s2")?.thread_id).toBe("s2");
  });

  it("URL 有 ?session= 但不存在 → **null**，不偷偷跳去别的会话", () => {
    // 深链失效还有真实场景：会话被 DELETE 了、sqlite 换了 checkpoint 库、
    // 恢复失败降级。悄悄选第一个会让用户以为「我的会话还在，只是内容变空了」，
    // 那比明说「不存在」糟得多。
    expect(pickInitialSession(sessions, "nope")).toBeNull();
  });

  it("没有 ?session= → 选第一个", () => {
    expect(pickInitialSession(sessions, null)?.thread_id).toBe("s1");
  });

  it("列表为空 → null（无论有没有 ?session=）", () => {
    expect(pickInitialSession([], null)).toBeNull();
    expect(pickInitialSession([], "s1")).toBeNull();
  });

  it("空字符串的 ?session= 按「没带」处理", () => {
    expect(pickInitialSession(sessions, "")?.thread_id).toBe("s1");
  });
});

describe("withLiveStatus", () => {
  const sessions = [info("s1", "idle"), info("s2", "idle")];

  it("live 为 null → 原样返回（WS 还没连上）", () => {
    expect(withLiveStatus(sessions, null)).toBe(sessions);
  });

  it("命中已有行 → **替换**那一行（WS 比 10 秒轮询新）", () => {
    const next = withLiveStatus(sessions, info("s2", "running"));
    expect(next.map((s) => s.status)).toEqual(["idle", "running"]);
    expect(next).toHaveLength(2);
  });

  it("没命中 → 插到最前面（新建的会话立刻出现在侧栏，不等下一次轮询）", () => {
    const next = withLiveStatus(sessions, info("s3", "running"));
    expect(next.map((s) => s.thread_id)).toEqual(["s3", "s1", "s2"]);
  });

  it("不改原数组（React 靠引用变化重渲染）", () => {
    const before = sessions.slice();
    withLiveStatus(sessions, info("s2", "running"));
    expect(sessions).toEqual(before);
  });
});

describe("resultText", () => {
  it("null / 非字符串 / 空白 → null（界面据此隐藏结果面板）", () => {
    expect(resultText(null)).toBeNull();
    expect(resultText({})).toBeNull();
    expect(resultText({ result: 42 })).toBeNull();
    expect(resultText({ result: "   \n " })).toBeNull();
  });

  it("有内容就原样返回（首尾空白不清 —— 摘要里的缩进有意义）", () => {
    expect(resultText({ result: "  完成\n" })).toBe("  完成\n");
  });
});
