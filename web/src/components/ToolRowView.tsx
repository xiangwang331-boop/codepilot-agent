import { RerunTag } from "./Badges";
import { summarizeArgs, type ToolRow } from "../model/events";

/**
 * 一次普通工具调用。
 *
 * ⚠️ **拿不到工具结果原文**：`agent/core.py:130-134` 发 `TOOL_CALL_COMPLETED` 时
 * 没有传 `detail`，所以这里只能显示「工具名 + 参数」。这不是 UI 偷懒，是后端
 * 事件契约的限制（属于已知的待办项）。
 */
export function ToolRowView({ row }: { row: ToolRow }): React.JSX.Element {
  // 三种状态各自的记号：✓ 完成 / ✗ 失败 / ▸ 进行中
  const mark = row.status === "done" ? "✓" : row.status === "failed" ? "✗" : "▸";
  const args = summarizeArgs(row.args);

  // 收口事件的 message 对 Completed 是 `name(args…)`、对 Failed 是 `name: 原因`。
  // 失败时把原因显示出来（这是唯一能看到的错误信息）。
  const failureNote = row.status === "failed" ? row.message : "";

  return (
    <div className="toolrow" data-status={row.status}>
      <span className="toolrow__mark mono">{mark}</span>
      <span className="toolrow__name">{row.name}</span>
      <span className="toolrow__args" title={args || failureNote || undefined}>
        {failureNote || args}
      </span>
      {row.isRerun && <RerunTag />}
    </div>
  );
}
