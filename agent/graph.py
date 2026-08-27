"""组装 ReAct 图。

图结构（对应 OpenHands 的 LocalConversation.run 的 while True 循环，交给 LangGraph 执行）：

    entry -> agent -(条件路由)-> tools
               ^                 |
               |_________________|   (tools 结果回流 agent，继续决策)

    agent 无工具调用时 -> END

checkpointer 可插拔：不传默认 MemorySaver（进程内，测试用）；
传 SqliteSaver 则 checkpoint 落盘（P1，CLI 用，支持 --resume 续跑）。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent import core
from agent.state import AgentState


def build_agent_graph(llm, tools: list, checkpointer=None):
    """组装 ReAct 图。

    - llm 需已 .bind_tools(tools)（或测试里的 FakeLLM）。
    - checkpointer：LangGraph BaseCheckpointSaver；为 None 时默认 MemorySaver，
      生产 CLI 传入 SqliteSaver 实现磁盘持久化。
    """
    tools_by_name = {t.name: t for t in tools}

    graph = StateGraph(AgentState)
    graph.add_node("agent", core.make_agent_node(llm))
    graph.add_node("tools", core.make_tools_node(tools_by_name))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", core.route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or MemorySaver())
