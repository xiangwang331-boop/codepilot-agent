/**
 * 会话状态判定 / 轮询节奏 / 选中逻辑。
 *
 * `SessionStatus` 四态的真源是 `runtime/session.py` 的状态机。**忙闲判定必须只看
 * 这两个状态字段**，绝不能看「worker 线程还活着吗」—— LangGraph 会**静默吞掉**
 * 挂起中的 interrupt（CLAUDE.md 关键坑 #31）：挂起时再 invoke 一次不报错、`next`
 * 清空、`interrupts` 归零、委派被丢弃且全程无异常。后端为此专门有回归测试。
 */
import type { SessionInfo, SessionStatus } from "../api/types";

/** 正在占用会话的状态：这两种下发指令会吃 409。 */
export function isBusy(status: SessionStatus): boolean {
  return status === "running" || status === "awaiting_approval";
}

/** 能不能下发新指令。 */
export function canSend(status: SessionStatus): boolean {
  return status === "idle";
}

/** 能不能点批准/拒绝（只有挂起态可以）。 */
export function canApprove(status: SessionStatus): boolean {
  return status === "awaiting_approval";
}

/** 终态：关闭后不再变化。 */
export function isTerminal(status: SessionStatus): boolean {
  return status === "closed";
}

const STATUS_LABELS: Record<SessionStatus, string> = {
  idle: "空闲",
  running: "运行中",
  awaiting_approval: "待批准",
  closed: "已关闭",
};

export function statusLabel(status: SessionStatus): string {
  return STATUS_LABELS[status] ?? status;
}

/**
 * 侧栏轮询间隔：**有忙会话就快轮，全闲就慢轮**。
 * 忙的时候 WS 也在推，轮询只是兜底（比如 WS 断了）；闲的时候没必要每 2 秒打一次。
 */
export function pollDelayMs(sessions: SessionInfo[]): number {
  return sessions.some((s) => isBusy(s.status)) ? 2000 : 10000;
}

/**
 * 首屏选哪个会话。
 *
 * - URL 带了 `?session=<id>` → 只有它**存在**才选它；不存在返回 `null`
 *   （调用方据此显示「会话不存在」，而不是偷偷跳到别的会话）。
 * - 没带 → 选列表第一个；列表空 → `null`。
 */
export function pickInitialSession(
  sessions: SessionInfo[],
  urlParam: string | null,
): SessionInfo | null {
  if (urlParam) {
    return sessions.find((s) => s.thread_id === urlParam) ?? null;
  }
  return sessions[0] ?? null;
}

/**
 * 把 WS 推来的**实时**状态合并进轮询来的列表。
 *
 * WS 的 status 信封比轮询新（轮询最长滞后 10 秒），所以选中会话那一行必须以
 * WS 为准，否则会出现「时间线已经在跑、侧栏还写着空闲」的割裂。
 */
export function withLiveStatus(
  sessions: SessionInfo[],
  live: SessionInfo | null,
): SessionInfo[] {
  if (live === null) return sessions;
  const i = sessions.findIndex((s) => s.thread_id === live.thread_id);
  if (i === -1) return [live, ...sessions];
  const next = sessions.slice();
  next[i] = live;
  return next;
}

/** 从 `result` 里抠出给人看的最终文本（`AgentState.result` 是模型的最终回答）。 */
export function resultText(result: Record<string, unknown> | null): string | null {
  if (result === null) return null;
  const r = result.result;
  return typeof r === "string" && r.trim() !== "" ? r : null;
}
