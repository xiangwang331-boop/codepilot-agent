import { AgentBadge } from "./Badges";
import type { ApprovalPayload } from "../api/types";

/**
 * 审批卡。挂在检查器顶部，**并且在时间线顶部镜像一条横幅** —— 这是一个
 * 「整个会话卡在这不动」的状态，错过一次就是干等。
 *
 * 除了 specialist 与完整任务文本，这里明确交代**拒绝之后会发生什么**：
 * 拒绝不是「整个任务失败」，而是把一条 ERROR 回流给 Supervisor，它会换条路继续
 * （`supervisor.py:90-91` → `core.py` 的 ToolMessage → Supervisor 下一轮决策）。
 * 旧测试页完全没交代这件事，用户不敢点拒绝。
 */
export function ApprovalGate({
  approvals,
  busy,
  onDecide,
  banner = false,
}: {
  approvals: ApprovalPayload[];
  busy: boolean;
  onDecide: (approved: boolean) => void;
  banner?: boolean;
}): React.JSX.Element | null {
  if (approvals.length === 0) return null;

  return (
    <>
      {approvals.map((a, i) => (
        <div className={banner ? "approval approval--banner" : "approval"} key={`${a.specialist}-${i}`}>
          <div className="approval__title">
            <span>⚠ 需要批准</span>
            <AgentBadge agent={a.specialist} />
          </div>

          <pre className="approval__question">{a.question}</pre>

          <div className="approval__consequence">
            批准后 {a.specialist} 会真的在 workspace 里执行。
            <br />
            <strong>拒绝不会终止任务</strong> —— 会把它当成一条错误回传给 Supervisor，
            由它换个路子继续（比如改派别的 specialist 或调整方案）。
          </div>

          <div className="approval__actions">
            <button className="approval__approve" disabled={busy} onClick={() => onDecide(true)}>
              批准执行
            </button>
            <button className="approval__reject" disabled={busy} onClick={() => onDecide(false)}>
              拒绝
            </button>
          </div>
        </div>
      ))}
    </>
  );
}
