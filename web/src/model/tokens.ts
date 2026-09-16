/**
 * Token 消耗聚合 —— `main.py` 收尾那段统计的 TS 镜像。
 *
 * 数据源是 `TokenUsage` 事件的 `detail`（`agent/core.py:69-77`）：
 * `{prompt_tokens, completion_tokens, total_tokens}`。
 *
 * ⚠️ **只有真实 LLM 才发这个事件**：`response_metadata["token_usage"]` 由
 * langchain-openai 从 OpenAI 兼容响应里映射出来，FakeLLM 没有该字段 → 不发
 * （`core.py:64`）。所以面板在假 LLM / 未跑过任务的会话上会是空的，这是正常的。
 */
import type { TokenUsageDetail } from "../api/types";
import type { TimelineEvent } from "./replay";

export interface TokenBucket {
  agent: string;
  /** LLM 调用次数（= TokenUsage 事件条数）。 */
  calls: number;
  prompt: number;
  completion: number;
  total: number;
}

export interface TokenSummary {
  total: TokenBucket;
  /** 按 total 降序。 */
  byAgent: TokenBucket[];
}

function emptyBucket(agent: string): TokenBucket {
  return { agent, calls: 0, prompt: 0, completion: 0, total: 0 };
}

/** 宽容地读一个数字：缺失 / 非数字 / NaN 一律算 0，绝不让聚合抛异常。 */
function num(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * 读一个**可能不存在**的数字：不存在或读不出数就返回 `null`（区别于 `num` 的 0）。
 *
 * 用来区分「后端给了 total_tokens=0」与「后端没给 / 给的是垃圾」——
 * 前者应当采信，后者应当退回 `prompt + completion`。
 */
function maybeNum(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

export function aggregateTokens(items: TimelineEvent[]): TokenSummary {
  const total = emptyBucket("__total__");
  const perAgent = new Map<string, TokenBucket>();

  for (const { event } of items) {
    if (event.type !== "TokenUsage") continue;
    const detail = (event.detail as TokenUsageDetail | null) ?? null;
    const prompt = num(detail?.prompt_tokens);
    const completion = num(detail?.completion_tokens);
    // 后端已经在 detail 里给了 total；缺了**或读不出数**就按 prompt+completion 兜底
    // ——「total 是 NaN」不该把已知的 prompt/completion 一起抹成 0。
    const t = maybeNum(detail?.total_tokens) ?? prompt + completion;

    let bucket = perAgent.get(event.agent);
    if (!bucket) {
      bucket = emptyBucket(event.agent);
      perAgent.set(event.agent, bucket);
    }
    bucket.calls += 1;
    bucket.prompt += prompt;
    bucket.completion += completion;
    bucket.total += t;

    total.calls += 1;
    total.prompt += prompt;
    total.completion += completion;
    total.total += t;
  }

  const byAgent = [...perAgent.values()].sort((a, b) => b.total - a.total);
  return { total, byAgent };
}

/** 人类可读的紧凑数字：12400 → "12.4k"。 */
export function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}
