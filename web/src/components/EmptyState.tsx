/**
 * 主区空状态。
 *
 * 刻意**不在页面加载时自动建会话**（旧测试页 `index.html:317` 是那样做的）：
 * 有了会话栏之后，每次刷新都凭空多一个会话就成了噪音。改成显式点「新建」。
 */
export function EmptyState({
  variant,
  reason,
  onCreate,
  onPick,
  hasSessions,
  creating,
}: {
  variant: "no-session" | "missing-session" | "loading";
  /** missing-session 时的具体原因（4404 的文案）。 */
  reason?: string;
  onCreate: () => void;
  onPick: () => void;
  hasSessions: boolean;
  creating: boolean;
}): React.JSX.Element {
  if (variant === "loading") {
    return (
      <div className="empty">
        <p>正在读取会话列表…</p>
      </div>
    );
  }

  if (variant === "missing-session") {
    return (
      <div className="empty">
        <h2>会话不存在</h2>
        <p>
          {reason ?? "这个会话 ID 已经不在服务端了。"}
          <br />
          {/* ⚠️ P9 起不能再归因到「服务重启」：持久化里的会话会被恢复出来（`runtime/catalog.py`），
              4404 现在只意味着「从没被持久化过 / 已被删除 / 恢复失败」。 */}
          服务端没有这个会话：它要么从没跑过、要么已被删除、要么恢复失败。这不是界面出错了。
        </p>
        <button onClick={hasSessions ? onPick : onCreate} disabled={creating}>
          {hasSessions ? "从列表里选一个" : creating ? "创建中…" : "+ 新建会话"}
        </button>
      </div>
    );
  }

  return (
    <div className="empty">
      <h2>开始一个新任务</h2>
      <p>
        每个会话有独立的 workspace 目录和独立的沙箱容器。下发需求后，Supervisor
        会拆解任务并委派给 specialist，过程实时显示在时间线上。
      </p>
      <button onClick={onCreate} disabled={creating}>
        {creating ? "创建中…" : "+ 新建会话"}
      </button>
    </div>
  );
}
