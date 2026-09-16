import { useState, type ReactNode } from "react";

import { AgentBadge, RerunTag, Time } from "./Badges";
import { agentRole } from "../model/agent";
import type { DelegationBlock } from "../model/events";

const STATE_NOTE: Record<DelegationBlock["state"], string | null> = {
  open: "执行中或等待批准 —— 子 agent 尚未收尾。",
  completed: null,
  failed: "子 agent 失败（父级的 delegate 调用照样会「完成」，所以只看这里会漏掉失败）。",
  // 拒绝批准（supervisor.py:91）与未知 specialist（:79）在事件流上**长得完全一样**
  // —— 都是 delegate 返回 ERROR 字符串、零个子事件。所以文案不写死「被拒绝」。
  not_executed: "未执行：批准被拒，或 specialist 名无法识别（事件流无法区分这两者）。",
};

/**
 * 一次委派 = 一个可折叠卡片，子 agent 的事件全在里面。
 *
 * 卡片头常显「谁 + 干了什么 + 几轮 + 多少 token」——这是整个界面信息量最高的一行。
 */
export function DelegationCard({
  block,
  children,
}: {
  block: DelegationBlock;
  children: ReactNode;
}): React.JSX.Element {
  // null = 跟随默认（进行中/失败/未执行自动展开，已完成的收起）；用户点过之后听用户的。
  const [manual, setManual] = useState<boolean | null>(null);
  const defaultOpen =
    block.state === "open" || block.state === "failed" || block.state === "not_executed";
  const open = manual ?? defaultOpen;

  const tokens = sumTokens(block);
  const note = STATE_NOTE[block.state];
  const childCount = block.blocks.length;

  return (
    <div className="delegation" data-state={block.state} data-agent={block.specialist}>
      <button className="delegation__head" onClick={() => setManual(!open)} aria-expanded={open}>
        <span className="delegation__caret" data-open={open}>
          ▶
        </span>
        <AgentBadge agent={block.specialist} />
        <span className="delegation__task" title={block.task}>
          {block.task || "(无任务描述)"}
        </span>
        <span className="delegation__stats">
          {block.childRuns > 0 && <span>{block.childRuns} 轮</span>}
          {tokens > 0 && <span>{tokens.toLocaleString()} tok</span>}
          {block.attempts > 1 && (
            <span title="审批挂起后 tools 节点从头重跑，所以执行了不止一次">尝试 {block.attempts} 次</span>
          )}
          <Time at={block.at} />
        </span>
        {block.isRerun && <RerunTag />}
      </button>

      {open && (
        <div className="delegation__body">
          {block.task && <pre className="delegation__taskfull">{block.task}</pre>}

          {block.summary && (
            <div
              className="delegation__summary"
              title="子 agent 最终回答的前 80 字 —— 事件流里只有截断版，全文只在 supervisor 的 ToolMessage 里"
            >
              {block.summary}
            </div>
          )}

          {note && <div className="delegation__note">{note}</div>}

          {childCount > 0 ? (
            <div className="delegation__children">{children}</div>
          ) : (
            block.state === "open" && (
              <div className="delegation__note">
                等待中…… {agentRole(block.specialist)}
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}

/**
 * 卡片头部的 token 小计，统计块内的 `TokenUsage` 事件。
 *
 * ⚠️ 只有真实 LLM 才发这个事件（`core.py:64` 读 `response_metadata["token_usage"]`，
 * FakeLLM 没有该字段），所以用假 LLM 跑的会话这里恒为 0，是正常的。
 */
function sumTokens(block: DelegationBlock): number {
  let total = 0;
  for (const b of block.blocks) {
    if (b.kind !== "token") continue;
    const t = b.item.event.detail?.total_tokens;
    if (typeof t === "number" && Number.isFinite(t)) total += t;
  }
  return total;
}
