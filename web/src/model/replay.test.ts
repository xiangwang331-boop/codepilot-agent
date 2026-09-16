/**
 * 重跑判重 —— 前端识别「interrupt 恢复后 tools 节点整段重跑」的唯一依据。
 *
 * 这一层的 bug 特别隐蔽：判重键**过宽**会把两条正常事件误标成重跑（用户在界面上
 * 看到莫名其妙的「↻」），**过窄**则识别不出真重跑（委派卡重复出现）。所以两侧都要钉。
 */
import { describe, expect, it } from "vitest";

import type { AgentEvent, EventEnvelope, EventType } from "../api/types";
import { APPROVED, BATCH } from "./__fixtures__";
import { canonicalJson, markReruns, replayKey } from "./replay";

function ev(over: Partial<AgentEvent> = {}): AgentEvent {
  return {
    type: "AgentStep" as EventType,
    agent: "coder",
    message: "决定调用 1 个工具: write_file",
    detail: null,
    timestamp: "2026-01-01T00:00:00+00:00",
    thread_id: "t",
    step: 1,
    node: "agent",
    ...over,
  };
}

function env(...events: AgentEvent[]): EventEnvelope[] {
  return events.map((event, i) => ({ kind: "event" as const, seq: i, event }));
}

describe("canonicalJson", () => {
  it("键序无关：{a,b} 与 {b,a} 同一个串", () => {
    expect(canonicalJson({ a: 1, b: 2 })).toBe(canonicalJson({ b: 2, a: 1 }));
  });

  it("嵌套对象也排序（递归生效，不是只排顶层）", () => {
    expect(canonicalJson({ x: { b: 1, a: 2 } })).toBe('{"x":{"a":2,"b":1}}');
  });

  it("数组**保持顺序**（顺序是语义的一部分，不能排）", () => {
    expect(canonicalJson([2, 1])).toBe("[2,1]");
    expect(canonicalJson([2, 1])).not.toBe(canonicalJson([1, 2]));
  });

  it("值是 undefined 的键被丢掉（与 JSON.stringify 一致），不是写成 null", () => {
    expect(canonicalJson({ a: 1, b: undefined })).toBe('{"a":1}');
  });

  it("null / undefined 都渲染成 null（对齐后端 default=str 的不抛语义）", () => {
    expect(canonicalJson(null)).toBe("null");
    expect(canonicalJson(undefined)).toBe("null");
  });

  it("标量按 JSON 走，字符串加引号", () => {
    expect(canonicalJson(1)).toBe("1");
    expect(canonicalJson("a")).toBe('"a"');
    expect(canonicalJson(true)).toBe("true");
  });
});

describe("replayKey", () => {
  it("图外事件（step === null）没有判重键", () => {
    expect(replayKey(ev({ step: null, node: null }))).toBeNull();
  });

  it("**agent 参与键** —— 这是与后端算法唯一的刻意偏离", () => {
    // 后端的键不含 agent：两个 specialist 在同一 superstep 各发一条 AgentStarted，
    // message 都是空串、detail 都是 null → 键完全相同 → 第二条被误判成重跑。
    const a = ev({ type: "AgentStarted", agent: "coder", message: "" });
    const b = ev({ type: "AgentStarted", agent: "tester", message: "" });
    expect(replayKey(a)).not.toBe(replayKey(b));
  });

  it("step / node / type / message / detail 逐项参与", () => {
    const base = ev({ detail: { args: { path: "a.py" } } });
    const variants = [
      ev({ step: 2, detail: { args: { path: "a.py" } } }),
      ev({ node: "tools", detail: { args: { path: "a.py" } } }),
      ev({ type: "AgentCompleted", detail: { args: { path: "a.py" } } }),
      ev({ message: "别的", detail: { args: { path: "a.py" } } }),
      ev({ detail: { args: { path: "b.py" } } }),
    ];
    for (const v of variants) expect(replayKey(v)).not.toBe(replayKey(base));
  });

  it("detail 的键序不影响键（走 canonicalJson）", () => {
    const a = ev({ detail: { args: { path: "a.py", content: "x" } } });
    const b = ev({ detail: { args: { content: "x", path: "a.py" } } });
    expect(replayKey(a)).toBe(replayKey(b));
  });

  it("node 为 null 但不该被判重的事件：step 有值就仍然有键", () => {
    expect(replayKey(ev({ step: 3, node: null }))).not.toBeNull();
  });
});

describe("markReruns", () => {
  it("首次不算重跑，第二次逐字相同的才算", () => {
    const items = markReruns(env(ev(), ev()));
    expect(items.map((i) => i.isRerun)).toEqual([false, true]);
  });

  it("seq 原样透传（重连游标 `?since=` 靠它）", () => {
    const items = markReruns(env(ev(), ev(), ev()));
    expect(items.map((i) => i.seq)).toEqual([0, 1, 2]);
  });

  it("图外事件永不标重跑（根 AgentStarted / AgentCompleted 本来就长一样）", () => {
    const outer = (): AgentEvent => ev({ type: "AgentStarted", agent: "Supervisor", message: "", step: null, node: null });
    const items = markReruns(env(outer(), outer(), outer()));
    expect(items.map((i) => i.isRerun)).toEqual([false, false, false]);
  });

  it("顺序被打乱也照样认（它只看键集合，不看相邻）", () => {
    const items = markReruns(env(ev({ message: "A" }), ev({ message: "B" }), ev({ message: "A" })));
    expect(items.map((i) => i.isRerun)).toEqual([false, false, true]);
  });
});

// ---------------------------------------------------------------- 真实夹具

describe("对真实夹具判重", () => {
  it("批准夹具：只有挂起重跑的那一条被标记", () => {
    const items = markReruns(APPROVED);
    const reruns = items.filter((i) => i.isRerun);
    expect(reruns).toHaveLength(1);
    const e = reruns[0]!.event;
    expect(e.type).toBe("ToolCallStarted");
    expect(e.message).toBe("delegate");
    // 它是 seq 2 的逐字重发 → 是第二条，不是第一条
    expect(reruns[0]!.seq).toBe(3);
  });

  it("一批两个委派的夹具：重跑是**一整段连续的 seq**，不是零星几条", () => {
    const items = markReruns(BATCH);
    // 第二轮重跑把第一个委派连同它的 7 条子事件、以及第二个委派的开块事件
    // **整段原样再发一遍**：seq 11..19 连续 9 条全是重跑。这条断言把「重跑 = 一整段」
    // 这个形状钉死——如果哪天后端改成只补发没有的那条，这里会立刻红。
    expect(items.filter((i) => i.isRerun).map((i) => i.seq)).toEqual([
      11, 12, 13, 14, 15, 16, 17, 18, 19,
    ]);
  });

  it("两条 delegate 开块事件各自认得出自己的原件（不串台）", () => {
    const opens = markReruns(BATCH).filter(
      (i) => i.event.type === "ToolCallStarted" && i.event.message === "delegate",
    );
    expect(opens.map((i) => i.seq)).toEqual([2, 10, 11, 19]);
    expect(opens.map((i) => i.isRerun)).toEqual([false, false, true, true]);
    const key = (i: (typeof opens)[number]): string | null => replayKey(i.event);
    // 重发的与**自己**的首发同键……
    expect(key(opens[2]!)).toBe(key(opens[0]!));
    expect(key(opens[3]!)).toBe(key(opens[1]!));
    // ……与**另一个**委派不同键。少了这一条，两张卡会被并成一张。
    expect(key(opens[0]!)).not.toBe(key(opens[1]!));
  });

  it("子 agent 的 AgentStarted 也会被标重跑 —— 这是对的，且不会漏到界面上", () => {
    // 子 agent 的 `AgentStarted` 是**父图的 tools 节点**发的（`supervisor.py:113`），
    // step/node 都取父图位置，所以重跑时它与首发逐字相同、自然被判成重跑。
    // 它**不会**让界面多出一行，因为折叠算法在「重开块」时把 `childRuns` 归零了
    // ——「跑过几轮」由 `attempts` 表达，不靠堆叠重复内容表达。
    const childStarts = markReruns(BATCH).filter(
      (i) => i.event.type === "AgentStarted" && i.event.agent !== "Supervisor",
    );
    expect(childStarts.map((i) => [i.seq, i.isRerun])).toEqual([
      [3, false],
      [12, true],
      [20, false],
    ]);
  });
});
