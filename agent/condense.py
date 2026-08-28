"""P4-3: Condense 节点 —— 长会话 messages 压缩。

设计（见 DESIGN.md §6 决策 #15/#17）：
- messages 是唯一真源（add_messages）。超过预算时，把中间历史压缩为摘要
  （RemoveMessage，按 id 精确删），保留头部（系统提示 + 任务）与尾部 recent 窗口，
  追加一条 rule-based 摘要 SystemMessage。
- 不调 LLM（无新增失败路径），与 interrupt / delegate / Error Recovery 天然兼容。
- 位置：tools → condense → agent（每次 tools 完整返回后、agent 下次决策前检查一次）。

触发（P4-3-2 起双守卫 OR）：
  a) 消息数超限：len(messages) > trigger_count（防碎消息堆积）；
  b) token 超预算：estimated_tokens(messages) > context_limit - reserve_tokens
     （防「消息少但单个巨大」，如巨型 pytest 输出 / read_file 大文件）。
  context_limit=None 时 token 守卫关闭，行为与 P4-3-1 完全一致。

压缩后 messages 布局 = [系统提示, 任务, ...recent K 条, 摘要]。
被压缩的中间消息里无 id 的跳过（防御）；低于预算或无中间段可压缩时返回 {}（no-op，行为不变）。

token 估算（P4-3-2）：用 tiktoken cl100k_base（langchain 对未知模型的 fallback 编码，
探测实证 ChatOpenAI.get_num_tokens 对 deepseek-v4-flash 返回的值与 cl100k_base 一致）。
纯本地、确定性、不构造 LLM 实例；估算误差由 reserve_tokens 吸收。可注入 count_tokens_fn。
keep_recent 强制偶数：头部之后消息严格「AI(带 tool_calls) → ToolMessage」成对，偶数尾窗
保证从 pair 边界切——压缩永不劈开最新 tool_call ↔ ToolMessage 配对。
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import RemoveMessage, SystemMessage

from events.events import EventType, emit

# rule-based 摘要：只摘工具结果，单条截断，最多取 10 条（避免摘要自身过长）
_SUMMARY_TOOL_MAX = 10
_SUMMARY_ITEM_MAX = 150


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """tiktoken 估算一段文本的 token 数（本地、确定性）。

    cl100k_base 是 langchain 对未知模型的 fallback 编码（探测实证：get_num_tokens
    对 deepseek-v4-flash 返回的值与 cl100k_base 计数一致）。tiktoken 是
    langchain-openai 的硬依赖，正常环境必在；异常兜底按字符数近似——中文约
    0.8~1 token/字，字符数偏保守，对预算安全（宁多勿漏）。
    """
    try:
        import tiktoken

        return len(tiktoken.get_encoding(encoding_name).encode(text))
    except Exception:  # noqa: BLE001  (纯本地估算，异常兜底不算新增失败路径)
        return len(text)


def _content_text(content: Any) -> str:
    """把消息 content（str 或 OpenAI 内容块列表）归一成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


def _count_messages_tokens(messages: list, count: Callable[[str], int]) -> int:
    """估算整条 messages 的 token 和（逐条 content 求和，含头部与摘要）。"""
    return sum(count(_content_text(m.content)) for m in messages)


def _summarize_tool_results(removed: list, removed_count: int) -> str:
    """从被压缩的中间消息里取工具结果（写文件确认/pytest 输出/ERROR/委派报告）拼成摘要。"""
    lines: list[str] = []
    for m in removed:
        if getattr(m, "type", "") != "tool":
            continue
        content = (getattr(m, "content", None) or "").strip()
        if not content:
            continue
        if len(content) > _SUMMARY_ITEM_MAX:
            content = content[:_SUMMARY_ITEM_MAX] + "…"
        lines.append(content)
        if len(lines) >= _SUMMARY_TOOL_MAX:
            break
    head = f"历史压缩——已将 {removed_count} 条中间历史压缩为摘要。"
    if not lines:
        return head + "（中间轮次无关键工具结果可摘）"
    return head + "已完成的工作（工具结果摘要）：\n" + "\n".join(lines)


def make_condense_node(
    trigger_count: int = 40,
    keep_recent: int = 20,
    context_limit: int | None = None,
    reserve_tokens: int | None = None,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> Callable[[dict], dict[str, Any]]:
    """构造 condense 节点。

    trigger_count:   messages 条数超过它才压缩（消息数守卫，防碎消息堆积）。
    keep_recent:     保留的尾部最近消息数（含最新 tool_call ↔ ToolMessage 配对）；
                     强制偶数——头部后消息成对交替，偶数尾窗保证从 pair 边界切。
    context_limit:   模型上下文窗口 token 预算（token 守卫）；None = 关闭 token 守卫，
                     仅消息数触发（与 P4-3-1 行为一致）。通常来自 Settings。
    reserve_tokens:  预算外 headroom（模型输出 + 下一轮新消息 + 估算误差）；
                     None → context_limit // 5（20%）。触发阈值 = context_limit - reserve_tokens。
    count_tokens_fn: token 估算函数（测试注入确定性的 fake；None → count_tokens/tiktoken）。
    返回节点的 state 更新 dict；未触发时返回 {}（no-op，与不加该节点行为一致）。
    """
    if keep_recent % 2:
        keep_recent -= 1  # 偶数化：尾窗从 pair 边界切，不劈开最新 tool_call ↔ ToolMessage
    counter = count_tokens_fn or count_tokens
    threshold_tokens = None
    if context_limit is not None:
        reserve = reserve_tokens if reserve_tokens is not None else context_limit // 5
        threshold_tokens = context_limit - reserve

    def condense_node(state: dict) -> dict[str, Any]:
        msgs = list(state.get("messages") or [])
        msg_over = len(msgs) > trigger_count
        est_tokens = None
        tok_over = False
        if threshold_tokens is not None:
            est_tokens = _count_messages_tokens(msgs, counter)
            tok_over = est_tokens > threshold_tokens
        if not (msg_over or tok_over):
            return {}

        # 保护头部：前导的 SystemMessage（系统提示）与 HumanMessage（任务）
        head_end = 0
        for m in msgs:
            if getattr(m, "type", "") in ("system", "human"):
                head_end += 1
            else:
                break

        keep_start = len(msgs) - keep_recent
        if keep_start <= head_end:
            return {}  # recent 窗口已覆盖到头部，无中间段可压

        removed = [m for m in msgs[head_end:keep_start] if getattr(m, "id", None)]
        if not removed:
            return {}

        summary = SystemMessage(
            content=_summarize_tool_results(removed, len(removed))
        )
        updates: list[Any] = [RemoveMessage(id=m.id) for m in removed] + [summary]
        # 只在真实触发时发事件（CLI 会打印触发时机与结果）；no-op 不发，正好用于观察触发点。
        agent = state.get("current_agent", "Supervisor")
        detail: dict[str, Any] = {
            "before": len(msgs),
            "after": len(msgs) - len(removed) + 1,
            "removed": len(removed),
            "summary": summary.content,
        }
        reason_parts: list[str] = []
        if msg_over:
            reason_parts.append("消息数")
        if tok_over:
            reason_parts.append("token")
        tokens_part = ""
        if threshold_tokens is not None:
            removed_ids = {m.id for m in removed}
            kept = [m for m in msgs if getattr(m, "id", None) not in removed_ids]
            detail["before_tokens"] = est_tokens
            detail["after_tokens"] = _count_messages_tokens(kept + [summary], counter)
            tokens_part = f"；{detail['before_tokens']} → {detail['after_tokens']} tokens"
        emit(
            EventType.CONDENSE,
            agent=agent,
            message=(
                f"压缩历史: {len(msgs)} → {len(msgs) - len(removed) + 1} 条"
                f"（触发: {'+'.join(reason_parts)}；压缩中间 {len(removed)} 条为摘要，"
                f"保留头部 + 最近 {keep_recent} 条{tokens_part}）"
            ),
            detail=detail,
        )
        return {"messages": updates}

    return condense_node


def recursion_limit_for(max_iterations: int) -> int:
    """P4-3: 按 max_iterations 放大 LangGraph recursion_limit。

    condense 节点使每轮迭代从 2 个 superstep（agent→tools）变成 3 个
    （agent→tools→condense）。LangGraph 默认 recursion_limit=25 会让长循环
    先撞框架的 GraphRecursionError，而不是 agent 节点的 max_iterations 守卫
    （P4-1 回归踩到：max_iterations=10 需 31 个 superstep > 25）。
    这里按 3N + 余量 计算调用侧 config 的 recursion_limit，保证 max_iterations
    仍是真正的循环上限——recursion_limit 只是框架安全网，放大不会去除守卫。
    """
    return max(50, max_iterations * 3 + 20)
