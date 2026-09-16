import { statusLabel } from "../model/session";
import type { SessionInfo } from "../api/types";

/**
 * 会话栏。
 *
 * ⚠️ 会话是**进程内存里的目录**（`runtime/registry.py`），服务一重启就清空 ——
 * 刷新页面发现列表空了不是 bug。4404 的提示语里也照实说了这一点。
 */
export function SessionRail({
  sessions,
  selectedId,
  onSelect,
  onCreate,
  onDelete,
  creating,
}: {
  sessions: SessionInfo[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  creating: boolean;
}): React.JSX.Element {
  return (
    <nav className="rail">
      <div className="rail__head">
        <button className="rail__new" onClick={onCreate} disabled={creating}>
          {creating ? "创建中…" : "+ 新建会话"}
        </button>
      </div>

      <div className="rail__list">
        {sessions.length === 0 ? (
          <div className="rail__empty">
            还没有会话。
            <br />
            新建一个开始吧。
          </div>
        ) : (
          sessions.map((s) => (
            <div
              key={s.thread_id}
              className="rail__item"
              aria-current={s.thread_id === selectedId}
              onClick={() => onSelect(s.thread_id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(s.thread_id);
                }
              }}
            >
              <span className="rail__dot" data-status={s.status} title={statusLabel(s.status)} />
              <span className="rail__id" title={s.thread_id}>
                {s.thread_id.slice(0, 8)}
              </span>
              <span className="rail__meta">{s.event_count}</span>
              <button
                className="rail__del"
                title="销毁会话（容器一并回收）"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.thread_id);
                }}
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </nav>
  );
}
