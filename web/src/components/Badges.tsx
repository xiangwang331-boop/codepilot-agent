import { agentLabel, agentRole, agentSlug } from "../model/agent";
import { statusLabel } from "../model/session";
import type { StreamState } from "../api/stream";
import type { SessionStatus } from "../api/types";

/** 状态药丸。`running` 带转圈，`awaiting_approval` 用警示色。 */
export function StatusPill({ status }: { status: SessionStatus }): React.JSX.Element {
  return (
    <span className="pill" data-status={status}>
      {status === "running" && <span className="spinner" />}
      {statusLabel(status)}
    </span>
  );
}

/** Agent 徽章。色相由 `data-agent` 驱动（见 tokens.css 的映射表）。 */
export function AgentBadge({ agent }: { agent: string }): React.JSX.Element {
  return (
    <span className="agentbadge" data-agent={agentSlug(agent)} title={agentRole(agent)}>
      {agentLabel(agent)}
    </span>
  );
}

/**
 * 连接状态。
 *
 * 4404 是**终局**（服务重启后内存里的会话目录就清空了，重连一万次也没用），
 * 所以这里明说「会话不存在」并给出重建入口，而不是无限转圈。
 */
export function ConnBadge({ state }: { state: StreamState }): React.JSX.Element {
  const text = (() => {
    switch (state.kind) {
      case "open":
        return "已连接";
      case "connecting":
        return "连接中…";
      case "reconnecting":
        return `重连中（第 ${state.attempt} 次，${Math.round(state.delayMs / 1000)}s 后）`;
      case "fatal":
        return "已断开";
      case "closed":
        return "已关闭";
    }
  })();

  const title =
    state.kind === "fatal"
      ? `${state.reason}${state.code !== null ? `（关闭码 ${state.code}）` : ""}`
      : undefined;

  return (
    <span className="conn" data-state={state.kind} title={title}>
      <span className="conn__dot" />
      {text}
    </span>
  );
}

/** 重跑标记（判重键命中时挂上）。 */
export function RerunTag({ title }: { title?: string }): React.JSX.Element {
  return (
    <span className="rerun" title={title ?? "同一事件发了两次：interrupt 恢复后 tools 节点从头重跑"}>
      ↻ 重跑
    </span>
  );
}

/** 时间戳。**必须走 `new Date`**：ISO 串带 +00:00 时区，`slice(11,19)` 会把 UTC 当本地。 */
export function Time({ at }: { at: string }): React.JSX.Element | null {
  if (!at) return null;
  const d = new Date(at);
  if (Number.isNaN(d.getTime())) return null;
  return (
    <span className="mono" style={{ fontSize: "var(--fs-xs)", color: "var(--fg-mute)" }}>
      {d.toLocaleTimeString()}
    </span>
  );
}
