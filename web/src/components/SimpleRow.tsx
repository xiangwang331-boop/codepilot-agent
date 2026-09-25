import { useState } from "react";

import { AgentBadge, RerunTag, Time } from "./Badges";
import { agentSlug } from "../model/agent";
import { condenseDetail, type SimpleBlock } from "../model/events";

/**
 * 不属于任何委派块的零散事件行：根生命周期、AgentStep、**用户指令**、TokenUsage、
 * Condense、失败。
 *
 * `AgentStarted` / `AgentCompleted` 的 **message 是空串**（`session.py:427/460` 与
 * `supervisor.py:113` 都发空串），所以按事件类型给文案，**不能直接渲染 message**
 * ——否则时间线上会出现一排没有文字的空行。
 */
export function SimpleRow({ block }: { block: SimpleBlock }): React.JSX.Element {
  const { event } = block.item;

  const label = (() => {
    switch (event.type) {
      case "AgentStarted":
        return event.message || "任务开始";
      case "AgentCompleted":
        return event.message || "本轮结束";
      default:
        return event.message;
    }
  })();

  return (
    // `data-agent` 必须走 `agentSlug`：`event.agent` 是**原始大小写**（"Supervisor" /
    // "User"），而 tokens.css 的色相选择器全是小写 —— 直接传原始值会一个都匹配不上，
    // `--agent-hue` 静默退化成兜底灰（下面 `.simple[data-kind="user"]` 用的就是它）。
    // `AgentBadge` 内部本来就走 slug，所以这里错了也只有容器受影响、徽章是好的。
    <div className="simple" data-kind={block.kind} data-agent={agentSlug(event.agent)}>
      <AgentBadge agent={event.agent} />
      <div className="simple__msg">
        {label}
        {block.kind === "condense" && <CondenseDetailView block={block} />}
        {block.item.isRerun && (
          <>
            {" "}
            <RerunTag />
          </>
        )}{" "}
        <Time at={event.timestamp} />
      </div>
    </div>
  );
}

/**
 * Condense 的展开细节。
 *
 * detail 键来自 `agent/condense.py:157-170`：`{before, after, removed, summary, before_tokens?, after_tokens?}`
 * —— token 两项**只在 token 守卫触发时才有**（消息数触发的那条路径不带）。
 */
function CondenseDetailView({ block }: { block: SimpleBlock }): React.JSX.Element | null {
  const [open, setOpen] = useState(false);
  const detail = condenseDetail(block.item.event);
  if (!detail) return null;

  const tokenNote =
    detail.before_tokens !== undefined && detail.after_tokens !== undefined
      ? `　token ${detail.before_tokens} → ${detail.after_tokens}`
      : "";

  return (
    <>
      <div className="mono" style={{ fontSize: "var(--fs-xs)", color: "var(--warn)" }}>
        {detail.before} → {detail.after} 条（压缩掉 {detail.removed} 条）{tokenNote}
      </div>
      {detail.summary !== "" && (
        <>
          <button
            className="iconbtn"
            style={{ padding: "0 6px", fontSize: "var(--fs-xs)" }}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "收起摘要" : "查看摘要"}
          </button>
          {open && <pre className="condense__summary">{detail.summary}</pre>}
        </>
      )}
    </>
  );
}
