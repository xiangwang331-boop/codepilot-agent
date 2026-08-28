"""ReAct 内核：agent 节点 + tools 节点 + 条件路由。

对应 OpenHands 的 Agent.step + classify_response + _execute_action_event：

- agent 节点   = 调 LLM，返回 AIMessage（可能带 tool_calls）
- route 条件边  = 有 tool_calls -> tools；否则 -> end（即 OpenHands 的 classify_response）
- tools 节点   = 逐个解析 tool_call -> 按名查表 -> 执行 -> 回写 ToolMessage（错误也回流，让 LLM 自纠）

不用 LangGraph 的 ToolNode，是因为我们需要：
1. 事件日志（ToolCallStarted/Completed/Failed）
2. 工具执行异常转成 ToolMessage 而不是抛出炸图
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

from agent.state import AgentState
from events.events import EventType, emit


def route(state: AgentState) -> str:
    """决定 agent 之后去哪：还有工具调用就继续，否则结束。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def _brief_args(args: dict) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        s = str(v)
        # if len(s) > 40:
        #     s = s[:37] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def make_agent_node(llm) -> Callable[[AgentState], dict[str, Any]]:
    def agent_node(state: AgentState) -> dict[str, Any]:
        agent_name = state.get("current_agent", "Coder")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 20)

        # 死循环守卫
        if iteration >= max_iter:
            msg = AIMessage(content=f"[stopped] 已达到最大迭代次数 {max_iter}，任务中止。")
            return {"messages": [msg], "status": "error", "error": "max_iterations reached"}

        # 调 LLM
        try:
            resp = llm.invoke(state["messages"])
        except Exception as e:  # noqa: BLE001
            err = f"LLM 调用失败: {type(e).__name__}: {e}"
            return {"messages": [AIMessage(content=err)], "status": "error", "error": err}

        # P4-3-2 可观测：真实 LLM 消耗（计费值）。OpenAI 兼容响应（含 DeepSeek）会把 usage
        # 放进 response_metadata["token_usage"]；FakeLLM 没有该字段 → 不发（测试零影响）。
        usage = (getattr(resp, "response_metadata", None) or {}).get("token_usage")
        if usage:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", prompt + completion)
            emit(
                EventType.TOKEN_USAGE,
                agent=agent_name,
                message=f"LLM 调用 输入 {prompt} → 输出 {completion} = {total} tokens",
                detail={
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                },
            )

        new_iter = iteration + 1
        tool_calls = list(getattr(resp, "tool_calls", None) or [])

        if tool_calls:
            names = ", ".join(tc["name"] for tc in tool_calls)
            emit(
                EventType.AGENT_STEP,
                agent=agent_name,
                message=f"决定调用 {len(tool_calls)} 个工具: {names}",
            )
            return {"messages": [resp], "iteration_count": new_iter, "status": "running"}

        # 无工具调用 = 最终回答
        emit(EventType.AGENT_STEP, agent=agent_name, message="给出最终回答")
        return {
            "messages": [resp],
            "iteration_count": new_iter,
            "status": "finished",
            "result": resp.content,
        }

    return agent_node


def make_tools_node(tools_by_name: dict) -> Callable[[AgentState], dict[str, Any]]:
    def tools_node(state: AgentState) -> dict[str, Any]:
        agent_name = state.get("current_agent", "Coder")
        last = state["messages"][-1]
        tool_calls = list(getattr(last, "tool_calls", None) or [])

        tool_messages: list[ToolMessage] = []
        logged_calls: list[dict] = []
        logged_obs: list[dict] = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            call_id = tc["id"]

            emit(EventType.TOOL_CALL_STARTED, agent=agent_name, message=name, detail={"args": args})
            logged_calls.append({"id": call_id, "name": name, "args": args})

            tool = tools_by_name.get(name)
            if tool is None:
                content = f"ERROR: 未知工具 '{name}'。可用工具: {sorted(tools_by_name)}"
                emit(EventType.TOOL_CALL_FAILED, agent=agent_name, message=f"{name}: 未知工具")
            else:
                try:
                    result = tool.invoke(args)
                    content = result if isinstance(result, str) else str(result)
                    emit(
                        EventType.TOOL_CALL_COMPLETED,
                        agent=agent_name,
                        message=f"{name}({_brief_args(args)})",
                    )
                except GraphBubbleUp:  # P4-2: 放行 LangGraph 框架信号（interrupt 等），不当工具错误
                    raise
                except Exception as e:  # noqa: BLE001
                    content = f"ERROR: {type(e).__name__}: {e}"
                    emit(
                        EventType.TOOL_CALL_FAILED,
                        agent=agent_name,
                        message=f"{name}: {type(e).__name__}: {e}",
                    )

            tool_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
            logged_obs.append({"tool_call_id": call_id, "content": content})

        return {
            "messages": tool_messages,
            "tool_calls": logged_calls,
            "observations": logged_obs,
        }

    return tools_node
