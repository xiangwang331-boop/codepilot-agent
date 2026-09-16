import { Panel } from "./Panel";
import { statusLabel } from "../model/session";
import type { SessionInfo } from "../api/types";

/** 会话信息：thread_id、状态、事件数、错误。 */
export function SessionMeta({ session }: { session: SessionInfo }): React.JSX.Element {
  return (
    <Panel title="会话信息" defaultOpen={false}>
      <dl className="meta" style={{ padding: 0 }}>
        <dt>会话 ID</dt>
        <dd>{session.thread_id}</dd>

        <dt>状态</dt>
        <dd>{statusLabel(session.status)}</dd>

        <dt>事件数</dt>
        <dd>{session.event_count}</dd>

        <dt>待批准</dt>
        <dd>{session.approval.length}</dd>
      </dl>

      {session.error !== null && (
        <div className="panel__empty" style={{ padding: 0, color: "var(--err)" }}>
          {session.error}
        </div>
      )}
    </Panel>
  );
}
