"""P2: Supervisor 编排 —— delegate 工具 + supervisor 图装配。

架构：supervisor 与 specialist 都复用现有 ReAct 内核（build_agent_graph）。
- supervisor 的工具集 = 只读工具（list_files/read_file/search_code）+ delegate。
- tools 节点执行 delegate 时，闭包同步 invoke 对应 specialist subgraph，
  结果文本作为 ToolMessage 回流给 supervisor，由它决定下一步。
- 父子完全隔离：task 字符串下去、result 字符串上来；父 messages 只多一条
  delegate 的 ToolMessage（含子 agent 最终报告），不含子图内部过程。

关键点：
- 子图用 MemorySaver + 每次委派唯一 thread_id，避免共享编译图下串状态。
- 子图失败返回 "ERROR: 子 agent 'x' 失败: ..."（遵守工具契约，让 supervisor 自纠）。
- AgentState 零新增：委派历史由父图 messages 的 delegate ToolMessage + observations 表达。
"""
from __future__ import annotations

from itertools import count
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.condense import make_condense_node, recursion_limit_for
from agent.graph import build_agent_graph
from agent.specialists import SPECIALISTS, Specialist, specialist_listing
from config.settings import Settings
from events.events import EventType, emit
from tools.registry import build_tools_subset
from workspace.manager import WorkspaceManager

# supervisor 自己的工具（只读 + delegate，不直接写文件/跑命令）
SUPERVISOR_TOOL_NAMES = ("list_files", "read_file", "search_code")


class DelegateArgs(BaseModel):
    specialist: str = Field(
        description="要委派的 specialist 名（小写）。可用 specialist:\n" + specialist_listing()
    )
    task: str = Field(description="自包含的任务描述：目标、相关文件、验收标准")


def build_specialist_subgraph(
    spec: Specialist,
    ws: WorkspaceManager,
    make_llm: Callable[[str], Any],
    condense_node: Callable | None = None,
) -> Any:
    """构造一个 specialist 的 ReAct subgraph（复用内核，MemorySaver）。

    condense_node（P4-3）：透传给子图，长 specialist 会话同样受益；None 表示不加。
    """
    subset = build_tools_subset(ws, spec.tool_names)
    llm = make_llm(spec.name).bind_tools(subset)
    return build_agent_graph(llm, subset, condense_node=condense_node)


def make_delegate_tool(
    specialist_graphs: dict[str, Any],
    settings: Settings,
    require_approval_for: tuple[str, ...] = ("coder",),
) -> StructuredTool:
    """delegate 工具：同步 invoke specialist subgraph，返回其最终报告文本。

    require_approval_for: 需要人类批准（interrupt 挂起）的 specialist 名集合，
    默认仅 coder（负责实际改业务代码）；tester 与只读 specialist 不打断。测试可传 () 关闭。
    """
    counter = count(1)

    def delegate(specialist: str, task: str) -> str:
        name = specialist.lower().strip()
        spec = SPECIALISTS.get(name)
        sub = specialist_graphs.get(name)
        if spec is None or sub is None:
            return f"ERROR: 未知 specialist '{specialist}'。可用: {specialist_listing()}"

        # P4-2: 委派需批准的 specialist 前，interrupt 挂起请求人类确认。
        # 挂起不是错误：不 emit AGENT_FAILED；恢复后 tools 节点重跑，interrupt() 返回 resume 值。
        if name in require_approval_for:
            answer = interrupt({
                "type": "approval",
                "specialist": name,
                "task": task,
                "question": f"是否批准委派 {name} 执行任务？\n任务: {task}",
            })
            if answer != "yes":
                return f"ERROR: 用户拒绝批准委派 {name}（输入: {answer}）。任务未执行。"

        child_state = {
            "messages": [SystemMessage(content=spec.system_prompt), HumanMessage(content=task)],
            "current_agent": spec.name,
            "current_task": task,
            "iteration_count": 0,
            "max_iterations": spec.max_iterations or settings.max_iterations,
            "status": "running",
            "tool_calls": [],
            "observations": [],
        }
        child_config = {
            "configurable": {"thread_id": f"delegate-{next(counter)}"},
            # P4-3: condense 使每轮迭代多 1 个 superstep（agent→tools→condense），
            # 放大 recursion_limit，否则长循环先撞框架 GraphRecursionError
            # 而非 max_iterations 守卫（P4-1「子 agent 失败」语义会被破坏）。
            "recursion_limit": recursion_limit_for(
                spec.max_iterations or settings.max_iterations
            ),
        }

        emit(EventType.AGENT_STARTED, agent=spec.name, message="")
        try:
            child_result = sub.invoke(child_state, child_config)
        except Exception as e:  # noqa: BLE001  (P4-1: 子图未预期异常兜底，不让它冒泡炸父图)
            err = f"子 agent '{name}' 执行抛异常: {type(e).__name__}: {e}"
            emit(EventType.AGENT_FAILED, agent=spec.name, message=err)
            return f"ERROR: {err}"

        if child_result.get("status") == "finished":
            text = child_result.get("result") or "(无最终回答)"
            emit(EventType.AGENT_COMPLETED, agent=spec.name, message=text[:80])
            return f"[委派完成] specialist={name}, status=finished\n子 agent 最终回答:\n{text}"
        err = child_result.get("error") or f"状态 {child_result.get('status')}"
        emit(EventType.AGENT_FAILED, agent=spec.name, message=err)
        return f"ERROR: 子 agent '{name}' 失败: {err}"

    return StructuredTool.from_function(
        name="delegate",
        description=(
            "委派一个自包含任务给 specialist 完成，返回其最终报告。可用 specialist:\n"
            f"{specialist_listing()}\n"
            "task 要写清目标、相关文件、验收标准。返回以 [委派完成] 开头为成功，ERROR 开头为失败。"
        ),
        args_schema=DelegateArgs,
        func=delegate,
    )


def build_supervisor_graph(
    settings: Settings,
    ws: WorkspaceManager,
    make_llm: Callable[[str], Any] | None = None,
    checkpointer=None,
    require_approval_for: tuple[str, ...] = ("coder",),
    condense_node: Callable | None = None,
):
    """组装 supervisor 图（复用 ReAct 内核）+ specialists subgraph。

    - make_llm(role) 返回该角色未 bind 的 LLM（生产：ChatOpenAI；测试：FakeLLM）。
      不传则用 settings 构造 ChatOpenAI。
    - require_approval_for: 委派前需人类批准（interrupt）的 specialist 集合，默认仅 coder。
    - condense_node（P4-3 长会话压缩）：None 时用默认 make_condense_node()，supervisor
      与每个 specialist 子图都挂上；测试可传自定义阈值节点（或高阈值关闭）。
    - supervisor 工具 = 只读 + delegate；挂 checkpointer（P1 SqliteSaver）。
    """
    if make_llm is None:
        from langchain_openai import ChatOpenAI

        def make_llm(role: str) -> Any:
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
                temperature=0,
            )

    if condense_node is None:
        # P4-3-2: 默认 condense 节点带 Settings 的 token 预算（context_limit/reserve_tokens；
        # 都是 None 时 token 守卫关闭，行为与 P4-3-1 一致）。测试仍可传自定义节点覆盖。
        condense_node = make_condense_node(
            context_limit=settings.context_limit,
            reserve_tokens=settings.reserve_tokens,
        )

    specialist_graphs = {
        name: build_specialist_subgraph(spec, ws, make_llm, condense_node)
        for name, spec in SPECIALISTS.items()
    }
    delegate_tool = make_delegate_tool(specialist_graphs, settings, require_approval_for)
    supervisor_tools = build_tools_subset(ws, SUPERVISOR_TOOL_NAMES) + [delegate_tool]
    supervisor_llm = make_llm("Supervisor").bind_tools(supervisor_tools)
    return build_agent_graph(
        supervisor_llm, supervisor_tools, checkpointer=checkpointer, condense_node=condense_node
    )
