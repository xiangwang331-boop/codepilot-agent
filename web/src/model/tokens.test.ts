/**
 * Token 聚合。数据源只有 `TokenUsage` 事件的 detail，而**只有真实 LLM 才发这个事件**
 * （FakeLLM 的 AIMessage 没有 `response_metadata.token_usage`），所以这里全部用造的事件。
 */
import { describe, expect, it } from "vitest";

import type { AgentEvent, EventType } from "../api/types";
import type { TimelineEvent } from "./replay";
import { aggregateTokens, formatTokens } from "./tokens";

function usage(agent: string, prompt: number, completion: number, detail?: unknown): TimelineEvent {
  const event: AgentEvent = {
    type: "TokenUsage" as EventType,
    agent,
    message: `消耗 ${prompt + completion} tokens`,
    detail: (detail ?? {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
    }) as Record<string, unknown>,
    timestamp: "2026-01-01T00:00:00+00:00",
    thread_id: "t",
    step: 1,
    node: "agent",
  };
  return { seq: 0, event, isRerun: false };
}

function other(type: EventType, agent = "coder"): TimelineEvent {
  return {
    seq: 0,
    event: {
      type,
      agent,
      message: "x",
      detail: null,
      timestamp: "2026-01-01T00:00:00+00:00",
      thread_id: "t",
      step: 1,
      node: "agent",
    },
    isRerun: false,
  };
}

describe("aggregateTokens", () => {
  it("空事件流 → 全零，不崩（新会话 / 假 LLM 会话的常态）", () => {
    const s = aggregateTokens([]);
    expect(s.total).toEqual({ agent: "__total__", calls: 0, prompt: 0, completion: 0, total: 0 });
    expect(s.byAgent).toEqual([]);
  });

  it("没有 TokenUsage 事件时同样全零（真实会发生的：任务还没跑）", () => {
    const s = aggregateTokens([other("AgentStarted"), other("ToolCallStarted")]);
    expect(s.total.calls).toBe(0);
    expect(s.byAgent).toEqual([]);
  });

  it("多 agent 分别累计，总计等于各 agent 之和", () => {
    const s = aggregateTokens([
      usage("coder", 100, 20),
      usage("coder", 50, 10),
      usage("tester", 30, 5),
    ]);
    expect(s.total).toMatchObject({ calls: 3, prompt: 180, completion: 35, total: 215 });
    expect(s.byAgent.map((b) => b.agent)).toEqual(["coder", "tester"]);
    expect(s.byAgent[0]).toMatchObject({ calls: 2, prompt: 150, completion: 30, total: 180 });
    expect(s.byAgent[1]).toMatchObject({ calls: 1, prompt: 30, completion: 5, total: 35 });
  });

  it("按 total 降序（面板读数从大到小扫）", () => {
    const s = aggregateTokens([usage("a", 1, 1), usage("b", 500, 500), usage("c", 10, 10)]);
    expect(s.byAgent.map((b) => b.agent)).toEqual(["b", "c", "a"]);
  });

  it("detail 缺 total_tokens → 用 prompt + completion 兜底", () => {
    const s = aggregateTokens([usage("coder", 0, 0, { prompt_tokens: 7, completion_tokens: 3 })]);
    expect(s.total.total).toBe(10);
  });

  it("detail 是 null → 记一次调用、金额为 0（不抛、不假装没调用过）", () => {
    const s = aggregateTokens([usage("coder", 0, 0, null)]);
    expect(s.total).toMatchObject({ calls: 1, total: 0 });
  });

  it("非数字 / NaN / 字符串数字都宽容处理", () => {
    const s = aggregateTokens([
      usage("coder", 0, 0, { prompt_tokens: "12", completion_tokens: null, total_tokens: NaN }),
      usage("coder", 0, 0, { prompt_tokens: undefined, completion_tokens: {}, total_tokens: "x" }),
    ]);
    // 第一条：prompt "12" → 12；completion null → 0；total_tokens 显式给了 NaN → 0
    // 第二条：全 0
    expect(s.total.total).toBe(12);
    expect(s.total.prompt).toBe(12);
    expect(s.total.calls).toBe(2);
  });

  it("同名 agent 归一个桶（大小写**不**归一 —— 真源里根就是 Supervisor 大写）", () => {
    const s = aggregateTokens([usage("Supervisor", 1, 1), usage("Supervisor", 2, 2)]);
    expect(s.byAgent).toHaveLength(1);
    expect(s.byAgent[0]).toMatchObject({ agent: "Supervisor", calls: 2 });
  });
});

describe("formatTokens", () => {
  it("< 1000 原样（不写 0.5k 这种）", () => {
    expect(formatTokens(0)).toBe("0");
    expect(formatTokens(999)).toBe("999");
  });

  it("千位保留一位小数", () => {
    expect(formatTokens(1000)).toBe("1.0k");
    expect(formatTokens(12_400)).toBe("12.4k");
  });

  it("百万位保留两位", () => {
    expect(formatTokens(1_000_000)).toBe("1.00M");
    expect(formatTokens(2_345_678)).toBe("2.35M");
  });
});
