import { statusLabel } from "../model/session";
import type { SessionInfo } from "../api/types";

/**
 * 会话栏。
 *
 * P9 起会话目录的真源是**持久化**（`runtime/catalog.py`），进程内存只是缓存 ——
 * 服务重启后历史会话照旧列在这里，所以这一栏不再是「重启就空」的易失视图。
 * 唯一的例外是**事件流**：`PERSISTENCE_BACKEND=sqlite` 时事件从不落库，
 * 会话还在但点进去时间线是空的 —— 由下面那条横幅明确说清（P9 决定 ④）。
 */
export function SessionRail({
  sessions,
  selectedId,
  onSelect,
  onCreate,
  onDelete,
  creating,
  historyAvailable,
}: {
  sessions: SessionInfo[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  creating: boolean;
  /** 事件流能否跨重启回放（`SessionList.history_available`）—— false 时出横幅。 */
  historyAvailable: boolean;
}): React.JSX.Element {
  return (
    <nav className="rail">
      <div className="rail__head">
        <button className="rail__new" onClick={onCreate} disabled={creating}>
          {creating ? "创建中…" : "+ 新建会话"}
        </button>
      </div>

      {/*
        横幅**无条件显示**（不挂在「有会话」上）：它说的是当前后端的**能力**，
        而不是某个会话的状态。让用户在攒出一堆历史之前就知道「这台机器上历史留不住」，
        比等他重启后自己发现好得多 —— 那正是 P9 要修的那个惊讶。
      */}
      {!historyAvailable && (
        <div className="rail__notice" role="status">
          <strong>事件流不落库</strong>
          （当前 PERSISTENCE_BACKEND=sqlite）。会话本身能从 checkpoint 恢复，所以它们照旧
          列在下面；但<strong>事件只在进程内存里，服务重启即丢</strong> —— 重启前跑过的
          会话点进去时间线是空的，那是真的没有，不是界面出错。
        </div>
      )}

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
