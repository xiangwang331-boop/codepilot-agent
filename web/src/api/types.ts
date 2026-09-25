/**
 * 后端 JSON 契约的 TS 镜像。**纯类型，零逻辑。**
 *
 * 逐字对齐的真源（改任何一处都要两边一起改）：
 *   - `api/schemas.py`           SessionInfo / EventEnvelope 的字段名
 *   - `runtime/session.py`       snapshot() / event_payload()
 *   - `events/events.py`         EventType 十个值 + Event 数据类
 *   - `api/ws.py`                三种信封信封 kind
 *   - `tests/test_web_ui_contract.py`  钉死上面这些（后端侧）
 *   - `web/src/api/types.test.ts`      钉死这里（前端侧）
 *
 * 注意 `AgentEvent.step` / `node` 的 null 语义：**图外事件**（会话层发的
 * 根 AgentStarted/AgentCompleted/AgentFailed）没有图位置，`step` 为 null。
 * 重跑判重只看 `step !== null` 的事件（见 `model/replay.ts`）。
 */

/**
 * `runtime/session.py` 的 SessionStatus.value.
 *
 * `interrupted` 是 P9 新增的第五态：**服务重启时正在跑**的会话，从 checkpoint 恢复后
 * `snap.next` 还非空（或 state 写着 `running`），但那个 worker 线程已经不存在了 ——
 * 既不能续跑（`begin()` 只接受 `idle`）也不该假装它空闲。恢复成 `running` 更糟：
 * 那会同时被 `begin()` 与 `reapable()` 拒绝，变成一个只能删的死会话。所以单独立一态：
 * **只读历史**（`runtime/catalog.py:derive_status`）。
 */
export type SessionStatus = "idle" | "running" | "awaiting_approval" | "interrupted" | "closed";

/** `events/events.py` 的 EventType 十个值 —— 顺序与枚举一致。 */
export type EventType =
  | "AgentStarted"
  | "AgentStep"
  | "ToolCallStarted"
  | "ToolCallCompleted"
  | "ToolCallFailed"
  | "AgentCompleted"
  | "AgentFailed"
  | "Condense"
  | "TokenUsage"
  /**
   * 用户下发的指令本身（`agent` 恒为 `"User"`，`message` 是需求原文，可多行）。
   *
   * **它是图外事件**：由会话层的 worker 在起图之前发（`runtime/session.py` 的
   * `Session._run`），所以 `step`/`node` 都是 null —— 与根 AgentStarted 同类。
   * 位置语义上它排在本轮所有 agent 事件之前。
   */
  | "UserMessage";

/** `events/events.py` 的 Event（经 `session.event_payload()` 序列化后）。 */
export interface AgentEvent {
  type: EventType;
  agent: string;
  message: string;
  detail: Record<string, unknown> | null;
  /** ISO8601 **秒级**精度（`events.py:85` 的 `timespec="seconds"`）。 */
  timestamp: string;
  thread_id: string;
  /** 图内 superstep 序号；图外事件为 null。 */
  step: number | null;
  /** 图内节点名（"agent" / "tools" / "condense"）；图外事件为 null。 */
  node: string | null;
}

/** `agent/supervisor.py:84-89` 的 interrupt payload（四个键）。 */
export interface ApprovalPayload {
  type: "approval";
  specialist: string;
  task: string;
  question: string;
}

/** `api/schemas.py` 的 SessionInfo。 */
export interface SessionInfo {
  thread_id: string;
  status: SessionStatus;
  /** 待批准队列（可空数组）。每个元素是 ApprovalPayload。 */
  approval: ApprovalPayload[];
  /** 已完成的图结果；`awaiting_approval` 挂起路径上可能仍为 null。 */
  result: Record<string, unknown> | null;
  error: string | null;
  event_count: number;
}

/**
 * `api/schemas.py` 的 SessionList —— `GET /sessions` 的响应体。
 *
 * `history_available` 是**列表级**的能力位，不是每个会话的属性：它说的是「这个进程
 * 读不读得到历史事件流」。`PERSISTENCE_BACKEND=postgres` 时 true；sqlite 后端为 false
 * （事件从不落库），此时会话照样列得出来（从 checkpoint 反推）但点进去
 * 时间线是空的 —— 界面**必须**明说这件事，不能假装一样。
 */
export interface SessionList {
  sessions: SessionInfo[];
  history_available: boolean;
}

// ---------------------------------------------------------------- WS 信封

/** 连上后第一条：当前状态。字段是 SessionInfo 的平铺（`kind` 之外无嵌套）。 */
export type StatusEnvelope = { kind: "status" } & SessionInfo;

/** 事件帧。`seq` 是会话内单调递增序号，也是重连 `?since=` 的游标。 */
export interface EventEnvelope {
  kind: "event";
  seq: number;
  event: AgentEvent;
}

/** 出错帧，之后服务端会 close（4404 会话不存在 / 4503 未就绪）。 */
export interface ErrorEnvelope {
  kind: "error";
  message: string;
}

export type Envelope = StatusEnvelope | EventEnvelope | ErrorEnvelope;

// ---------------------------------------------------------------- detail 形状

/** `ToolCallStarted` 的 detail。**只有它带 args**；Completed/Failed 都没有 detail。 */
export interface ToolCallStartedDetail {
  args: Record<string, unknown>;
}

/** `TokenUsage` 的 detail（`agent/core.py:73-77`）。 */
export interface TokenUsageDetail {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** `Condense` 的 detail（`agent/condense.py:157-170`）。token 两项仅在 token 守卫触发时存在。 */
export interface CondenseDetail {
  before: number;
  after: number;
  removed: number;
  summary: string;
  before_tokens?: number;
  after_tokens?: number;
}

// ---------------------------------------------------------------- 类型守卫

/**
 * 把 `AgentEvent.detail` 收窄成具体形状。
 * `detail` 是 `Record<string, unknown> | null`，读之前必须收窄——后端只保证它是
 * 「可 JSON 序列化的 dict 或 null」，不保证键的存在。
 */
export function detailAs<T>(event: AgentEvent): T | null {
  return (event.detail as T | null) ?? null;
}
