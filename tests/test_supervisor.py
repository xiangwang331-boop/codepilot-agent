"""P2: Supervisor 编排测试 —— 按角色注入 FakeLLM，确定性驱动多 agent 协作。

每个测试构造一个真实 WorkspaceManager + 各角色 FakeLLM script：
supervisor 的 script 决定何时 delegate、委派给谁；specialist 的 script
决定子图内部动作。全程无需 API key / 网络。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
import pytest

from agent.condense import make_condense_node
from agent.specialists import SPECIALISTS, build_supervisor_prompt, specialist_listing
from agent.supervisor import build_supervisor_graph, make_delegate_tool
from conftest import FakeLLM, QUICKSORT_CODE, TEST_CODE, _tool_call
from config.settings import Settings
from events.events import EventType, emitter
from tools.registry import build_tools_subset
from workspace.manager import WorkspaceManager

# 编排类测试显式退出 condense：断言完整委派序列/历史，不让压缩把中间记录收进摘要。
# 与 require_approval_for=() 退出 interrupt 同理（P4-2 先例）。
_NO_CONDENSE = make_condense_node(trigger_count=10**9)


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="",
        llm_base_url="",
        llm_model="fake",
        workspace_root=tmp_path / "ws",
        max_iterations=10,
    )


def _initial(task="多 agent 协作完成快排"):
    return {
        "messages": [
            SystemMessage(content=build_supervisor_prompt()),
            HumanMessage(content=task),
        ],
        "current_agent": "Supervisor",
        "current_task": task,
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }


def _make_fakes(scripts: dict[str, list]) -> tuple:
    """构造按角色分发的 FakeLLM factory；未规划的角色给空脚本（fallback）。"""
    fakes = {role: FakeLLM(script) for role, script in scripts.items()}

    def make(role: str) -> FakeLLM:
        return fakes.setdefault(role, FakeLLM([]))

    return make, fakes


def _delegate_tool_msgs(state) -> list:
    return [m for m in state["messages"] if getattr(m, "type", "") == "tool"]


def test_single_delegate_roundtrip(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入 main.py"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写一个 quicksort 到 main.py"}),
        AIMessage(content="完成：Coder 已写好快排。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "s1"}})

    # 循环正常结束，最终报告可见
    assert result["status"] == "finished"
    assert "快排" in result["result"]
    # 文件真的由子 agent 写进了 workspace
    assert "quicksort" in ws.read_text("main.py")
    # 父 messages 恰好多一条 delegate 的 ToolMessage
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1
    assert "[委派完成]" in tmsgs[0].content
    # 事件流带出 [Supervisor] 与 [coder] 两个角色
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "coder"} <= agents


def test_relay_chain_coder_tester_reviewer(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="main.py 已写入"),
        ],
        "tester": [
            _tool_call(1, "write_file", {"path": "test_main.py", "content": TEST_CODE}),
            AIMessage(content="test_main.py 已写入，pytest 通过"),
        ],
        "reviewer": [
            _tool_call(1, "read_file", {"path": "main.py"}),
            AIMessage(content="审查通过，未发现问题。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排"}),
        _tool_call(2, "delegate", {"specialist": "tester", "task": "为快排写测试"}),
        _tool_call(3, "delegate", {"specialist": "reviewer", "task": "审查代码与测试"}),
        AIMessage(content="完成：代码、测试、审查齐全。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "s2"}})

    assert result["status"] == "finished"
    assert "quicksort" in ws.read_text("main.py")
    assert "test_quicksort" in ws.read_text("test_main.py")
    assert len(_delegate_tool_msgs(result)) == 3
    # 三个 specialist 都在事件流里露过面
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "coder", "tester", "reviewer"} <= agents


def test_reviewer_loopback_redelegates_coder(tmp_path):
    """Reviewer 打回 → supervisor 再次委派 Coder；第二次委派状态必须隔离。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="第一版已写入"),
            _tool_call(2, "edit_file", {"path": "main.py", "content": QUICKSORT_CODE + "# v2\n"}),
            AIMessage(content="已按审查意见修改"),
        ],
        "reviewer": [
            _tool_call(1, "read_file", {"path": "main.py"}),
            AIMessage(content="结论：有问题。空数组边界未覆盖。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排第一版"}),
        _tool_call(2, "delegate", {"specialist": "reviewer", "task": "审查第一版"}),
        _tool_call(3, "delegate", {"specialist": "coder", "task": "根据审查意见修改"}),
        AIMessage(content="完成：Coder 已按 Review 意见修改。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "s3"}})

    assert result["status"] == "finished"
    coder_msgs = [m for m in _delegate_tool_msgs(result) if "specialist=coder" in m.content]
    assert len(coder_msgs) == 2
    # 第二次委派成功（唯一 thread_id 隔离，不继承第一次的迭代计数/消息历史）
    assert "[委派完成]" in coder_msgs[1].content


def test_unknown_specialist_error_and_self_correct(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "mystery", "task": "做个计划"}),
        _tool_call(2, "delegate", {"specialist": "coder", "task": "写快排"}),
        AIMessage(content="mystery 不可用，改派 Coder，完成。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "s4"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert tmsgs[0].content.startswith("ERROR")
    assert "未知 specialist" in tmsgs[0].content
    assert "[委派完成]" in tmsgs[1].content
    assert "quicksort" in ws.read_text("main.py")


def test_supervisor_tool_subset(tmp_path):
    ws = WorkspaceManager(tmp_path / "ws")
    subset = build_tools_subset(ws, ["list_files", "read_file", "search_code"])
    assert {t.name for t in subset} == {"list_files", "read_file", "search_code"}


def test_build_tools_subset_rejects_unknown(tmp_path):
    ws = WorkspaceManager(tmp_path / "ws")
    with pytest.raises(ValueError):
        build_tools_subset(ws, ["no_such_tool"])


def test_analyst_delegate_roundtrip(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    ws.write_text("main.py", QUICKSORT_CODE)  # 预置文件供 Analyst 只读分析
    make, fakes = _make_fakes({
        "analyst": [
            _tool_call(1, "list_files", {}),
            _tool_call(2, "read_file", {"path": "main.py"}),
            AIMessage(content="分析结论：main.py 已有 quicksort，结构清晰。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "analyst", "task": "分析 main.py 结构"}),
        AIMessage(content="完成：Analyst 已给出分析。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "a1"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1
    assert "分析结论" in tmsgs[0].content
    assert {"Supervisor", "analyst"} <= {e.agent for e in emitter.events}


def test_planner_delegate_roundtrip(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    ws.write_text("main.py", QUICKSORT_CODE)
    make, fakes = _make_fakes({
        "planner": [
            _tool_call(1, "search_code", {"query": "def quicksort"}),
            AIMessage(content="计划：修改 main.py 增加边界处理；步骤 1/2/3；验证 pytest。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "planner", "task": "制定快排实现计划"}),
        AIMessage(content="完成：Planner 已给出计划。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "p1"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1
    assert "计划" in tmsgs[0].content


def test_analyst_planner_coder_tester_relay(tmp_path):
    """接力链：Analyst → Planner → Coder → Tester 依次委派。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "analyst": [AIMessage(content="分析：项目为空，需新建 quicksort。")],
        "planner": [AIMessage(content="计划：新建 main.py 实现 quicksort。")],
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="main.py 已写入"),
        ],
        "tester": [
            _tool_call(1, "write_file", {"path": "test_main.py", "content": TEST_CODE}),
            AIMessage(content="test_main.py 已写入"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "analyst", "task": "分析现状"}),
        _tool_call(2, "delegate", {"specialist": "planner", "task": "制定计划"}),
        _tool_call(3, "delegate", {"specialist": "coder", "task": "实现"}),
        _tool_call(4, "delegate", {"specialist": "tester", "task": "测试"}),
        AIMessage(content="完成：分析、计划、实现、测试齐全。"),
    ])

    graph = build_supervisor_graph(
        _settings(tmp_path), ws, make_llm=make, require_approval_for=(),
        condense_node=_NO_CONDENSE,  # 编排序列断言：不掺压缩
    )
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "r1"}})

    assert result["status"] == "finished"
    assert len(_delegate_tool_msgs(result)) == 4
    assert "quicksort" in ws.read_text("main.py")
    assert "test_quicksort" in ws.read_text("test_main.py")
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "analyst", "planner", "coder", "tester"} <= agents


def test_readonly_specialists_have_no_write_tools(tmp_path):
    """Analyst/Planner/Reviewer 只读：工具集不含写/删除/执行工具（硬性边界）。"""
    ws = WorkspaceManager(tmp_path / "ws")
    for name in ("analyst", "planner", "reviewer"):
        spec = SPECIALISTS[name]
        names = {t.name for t in build_tools_subset(ws, spec.tool_names)}
        assert names == {"list_files", "read_file", "search_code"}
        assert not (names & {"write_file", "edit_file", "delete_file", "run_command"})


def test_specialist_listing_includes_new_roles():
    """注册表自动传播：prompt 与 delegate 描述含 analyst/planner/debugger。"""
    listing = specialist_listing()
    assert "analyst" in listing and "planner" in listing
    assert "debugger" in listing
    prompt = build_supervisor_prompt()
    assert "analyst" in prompt and "planner" in prompt
    assert "debugger" in prompt


def test_delegate_accepts_mixed_case_specialist(tmp_path):
    """specialist 参数大小写归一化：真实 LLM 传 'Coder' 也应成功。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "Coder", "task": "写快排"}),
        AIMessage(content="完成"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "case1"}})

    assert result["status"] == "finished"
    assert "[委派完成]" in _delegate_tool_msgs(result)[0].content
    assert "quicksort" in ws.read_text("main.py")


def test_debugger_delegate_roundtrip(tmp_path):
    """Debugger 单次委派：复现 + 定位 + 输出修复建议（不改文件）。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    ws.write_text("main.py", QUICKSORT_CODE)
    make, fakes = _make_fakes({
        "debugger": [
            _tool_call(1, "run_command", {"command": "python -m pytest -q"}),
            _tool_call(2, "read_file", {"path": "main.py"}),
            AIMessage(content="定位：quicksort 未处理空数组。根因在 main.py。建议 Coder 增加空数组边界。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "debugger", "task": "定位测试失败根因"}),
        AIMessage(content="完成：Debugger 已定位并给出建议。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "d1"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1
    assert "定位" in tmsgs[0].content
    assert "建议" in tmsgs[0].content
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "debugger"} <= agents


def test_reviewer_blocking_delegates_debugger_then_coder(tmp_path):
    """P3-2 闭环：Coder 第一版 → Reviewer 发现问题(BLOCKING) → Debugger 定位
    → Coder 修复 → Tester 回归 → Reviewer 复审通过。全程由 Supervisor 编排。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="第一版已写入"),
            _tool_call(2, "edit_file", {
                "path": "main.py",
                "old_string": "    return quicksort(left) + middle + quicksort(right)",
                "new_string": "    return quicksort(left) + middle + quicksort(right)\n    # 已按 Debugger 建议修复空数组边界",
            }),
            AIMessage(content="已按 Debugger 建议修复"),
        ],
        "reviewer": [
            _tool_call(1, "read_file", {"path": "main.py"}),
            AIMessage(content="结论：有问题。BLOCKING：空数组边界未覆盖。"),
            _tool_call(2, "read_file", {"path": "main.py"}),
            AIMessage(content="结论：通过，未发现问题。"),
        ],
        "debugger": [
            _tool_call(1, "run_command", {"command": "python -m pytest -q"}),
            AIMessage(content="定位：空数组边界未覆盖，根因在 main.py。建议 Coder 增加边界判断。"),
        ],
        "tester": [
            _tool_call(1, "run_command", {"command": "python -m pytest -q"}),
            AIMessage(content="pytest 全部通过。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排第一版"}),
        _tool_call(2, "delegate", {"specialist": "reviewer", "task": "审查第一版"}),
        _tool_call(3, "delegate", {"specialist": "debugger", "task": "定位 Reviewer 发现的问题"}),
        _tool_call(4, "delegate", {"specialist": "coder", "task": "按 Debugger 建议修复"}),
        _tool_call(5, "delegate", {"specialist": "tester", "task": "回归测试"}),
        _tool_call(6, "delegate", {"specialist": "reviewer", "task": "复审修复结果"}),
        AIMessage(content="完成：Debugger 定位 → Coder 修复 → Tester 回归 → Reviewer 复审通过。"),
    ])

    graph = build_supervisor_graph(
        _settings(tmp_path), ws, make_llm=make, require_approval_for=(),
        condense_node=_NO_CONDENSE,  # 6 步委派序列断言：不掺压缩
    )
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "d2"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 6
    # 委派序列：coder → reviewer(BLOCKING) → debugger → coder → tester → reviewer(通过)
    specialists = [m.content.split("specialist=")[1].split(",")[0] for m in tmsgs]
    assert specialists == ["coder", "reviewer", "debugger", "coder", "tester", "reviewer"]
    # Reviewer 第一次报告含 BLOCKING，复审通过；Debugger 报告含定位/建议
    assert "BLOCKING" in tmsgs[1].content
    assert "定位" in tmsgs[2].content and "建议" in tmsgs[2].content
    assert "通过" in tmsgs[5].content
    # 最终文件已被修复
    assert "已按 Debugger 建议修复" in ws.read_text("main.py")
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "coder", "tester", "reviewer", "debugger"} <= agents


class _BlockingAwareLLM(FakeLLM):
    """Supervisor 用消息感知 FakeLLM：第 2 次决策必须看到 Reviewer 的 BLOCKING
    报告（ToolMessage 已回流），才允许它委派 Debugger。否则 assert 失败。"""

    def __init__(self):
        super().__init__([])
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        joined = "".join(getattr(m, "content", "") or "" for m in messages)
        if self.calls == 1:
            return _tool_call(1, "delegate", {"specialist": "reviewer", "task": "审查代码与测试"})
        if self.calls == 2:
            assert "BLOCKING" in joined, (
                f"supervisor 第二次决策必须看到 Reviewer 的 BLOCKING 报告，实际: {joined[:200]}"
            )
            return _tool_call(2, "delegate", {"specialist": "debugger", "task": "定位问题根因"})
        return AIMessage(content="完成：已根据审查意见委派 Debugger。")


def test_supervisor_redescides_from_blocking_toolmessage(tmp_path):
    """最小转折点：Supervisor 收到 Reviewer 的 BLOCKING ToolMessage → 再次 LLM 决策
    → 委派 Debugger。第 2 次决策的 messages 必须真实含 BLOCKING（由 assert 证明）。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "reviewer": [
            AIMessage(content="结论：有问题。BLOCKING：空数组边界未覆盖。"),
        ],
        "debugger": [
            AIMessage(content="定位：根因在 main.py 空数组未处理。"),
        ],
    })
    fakes["Supervisor"] = _BlockingAwareLLM()

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "d3"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 2
    # 第二条委派对象是 debugger（supervisor 已根据 BLOCKING 重新决策）
    assert "specialist=debugger" in tmsgs[1].content
    agents = {e.agent for e in emitter.events}
    assert {"Supervisor", "reviewer", "debugger"} <= agents


def test_debugger_tool_permissions(tmp_path):
    """Debugger 只读 + run_command：可复现验证，但无写/删工具（硬性边界）。"""
    ws = WorkspaceManager(tmp_path / "ws")
    spec = SPECIALISTS["debugger"]
    names = {t.name for t in build_tools_subset(ws, spec.tool_names)}
    assert names == {"list_files", "read_file", "search_code", "run_command"}
    assert not (names & {"write_file", "edit_file", "delete_file"})


class _ThrowingSubgraph:
    """模拟 specialist 子图在 invoke 时抛未预期异常（P4-1 delegate 兜底测试用）。"""

    def invoke(self, state, config):
        raise RuntimeError("boom: 子图内部未预期异常")


class _ThrowingLLM:
    """invoke 抛 RuntimeError（Exception 子类）：被子图 agent 节点捕获转 error 状态。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError("api down")


def test_delegate_swallows_subgraph_exception(tmp_path):
    """P4-1: delegate 兜底——子图 invoke 直接抛异常 → AGENT_FAILED + ERROR 回流，不冒泡到父图。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    delegate_tool = make_delegate_tool(
        {"coder": _ThrowingSubgraph()}, _settings(tmp_path), require_approval_for=()
    )
    result = delegate_tool.invoke({"specialist": "coder", "task": "写快排"})
    # 返回 ERROR 字符串，异常被吞掉而非抛出
    assert result.startswith("ERROR: 子 agent 'coder' 执行抛异常")
    assert "RuntimeError" in result and "boom" in result
    # AGENT_FAILED 已触发（与 AGENT_STARTED 对称），agent 指向 coder
    failed = [e for e in emitter.events if e.type is EventType.AGENT_FAILED]
    assert len(failed) == 1 and failed[0].agent == "coder"


def test_child_agent_failure_reflows_as_error_then_supervisor_continues(tmp_path):
    """P4-1: 子 agent LLM 失败 → 子图 status=error → delegate 返回 ERROR ToolMessage
    → Supervisor 收到后继续决策并正常收尾。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({})
    fakes["coder"] = _ThrowingLLM()
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排"}),
        AIMessage(content="coder 失败，已记录错误，任务中止。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "er1"}})

    # Supervisor 未被拉垮，正常收尾
    assert result["status"] == "finished"
    assert "任务中止" in result["result"]
    # ERROR ToolMessage 已回流，且包含子 agent 的失败语义
    tmsgs = _delegate_tool_msgs(result)
    assert tmsgs[0].content.startswith("ERROR")
    assert "子 agent 'coder' 失败" in tmsgs[0].content
    assert "LLM 调用失败" in tmsgs[0].content
    # 事件流：AGENT_FAILED 一次，agent=coder；Supervisor 继续运行过
    failed = [e for e in emitter.events if e.type is EventType.AGENT_FAILED]
    assert len(failed) == 1 and failed[0].agent == "coder"
    agents = {e.agent for e in emitter.events}
    assert "Supervisor" in agents


def test_child_agent_max_iterations_reflows_as_error_then_supervisor_continues(tmp_path):
    """P4-1: 子 agent 撞 max_iterations → 子图 status=error → delegate 返回 ERROR
    ToolMessage → Supervisor 收到后继续决策并正常收尾。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    # coder 无限工具调用，脚本足够长保证先撞 max_iterations(10)
    loop_script = [_tool_call(i, "list_files", {}) for i in range(1, 30)]
    make, fakes = _make_fakes({"coder": loop_script})
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排"}),
        AIMessage(content="coder 迭代超限，已记录错误。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make, require_approval_for=())
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "er2"}})

    assert result["status"] == "finished"
    tmsgs = _delegate_tool_msgs(result)
    assert tmsgs[0].content.startswith("ERROR")
    assert "子 agent 'coder' 失败" in tmsgs[0].content
    assert "max_iterations" in tmsgs[0].content
    failed = [e for e in emitter.events if e.type is EventType.AGENT_FAILED]
    assert len(failed) == 1 and failed[0].agent == "coder"
    agents = {e.agent for e in emitter.events}
    assert "Supervisor" in agents


# ─────────────────────────── P4-2: Human Approval (interrupt) ───────────────────────────

def test_delegate_interrupts_pending_approval(tmp_path):
    """P4-2: 委派 coder（默认需批准）→ interrupt 挂起；resume=yes 后继续执行并落盘。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入 main.py"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写一个 quicksort 到 main.py"}),
        AIMessage(content="完成：Coder 已写好快排。"),
    ])

    # 默认 require_approval_for=("coder",)，不显式传参
    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
    config = {"configurable": {"thread_id": "app1"}}

    result = graph.invoke(_initial(), config)
    # 未到 finished：挂在 tools 节点等人类批准（invoke 正常返回部分状态）
    assert result.get("status") == "running"
    snap = graph.get_state(config)
    assert snap.next == ("tools",)
    assert len(snap.interrupts) == 1
    payload = snap.interrupts[0].value
    assert payload["type"] == "approval"
    assert payload["specialist"] == "coder"
    assert "写一个 quicksort" in payload["task"]
    # 批准前子图未执行：文件不存在
    assert not (tmp_path / "ws" / "main.py").exists()

    # 批准后继续：子图真正执行，文件落盘
    result = graph.invoke(Command(resume="yes"), config)
    assert result["status"] == "finished"
    assert "快排" in result["result"]
    assert "quicksort" in ws.read_text("main.py")
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1 and "[委派完成]" in tmsgs[0].content


def test_delegate_approval_rejected_then_supervisor_continues(tmp_path):
    """P4-2: 拒绝批准 → delegate 返回 ERROR ToolMessage 回流，Supervisor 继续决策收尾。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({})
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写一个 quicksort 到 main.py"}),
        AIMessage(content="用户拒绝委派，任务中止。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
    config = {"configurable": {"thread_id": "app2"}}
    graph.invoke(_initial(), config)
    snap = graph.get_state(config)
    assert len(snap.interrupts) == 1
    assert snap.interrupts[0].value["specialist"] == "coder"

    result = graph.invoke(Command(resume="no"), config)
    assert result["status"] == "finished"
    assert "拒绝委派" in result["result"]
    tmsgs = _delegate_tool_msgs(result)
    assert tmsgs[0].content.startswith("ERROR")
    assert "用户拒绝批准" in tmsgs[0].content
    # 拒绝后子图未执行：文件未被写入
    assert not (tmp_path / "ws" / "main.py").exists()


def test_readonly_specialist_does_not_interrupt(tmp_path):
    """P4-2: 只读 specialist（debugger）不在默认批准集 → 不 interrupt，一次跑完。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    ws.write_text("main.py", QUICKSORT_CODE)
    make, fakes = _make_fakes({
        "debugger": [
            _tool_call(1, "run_command", {"command": "python -m pytest -q"}),
            AIMessage(content="定位：空数组边界。建议 Coder 修复。"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "debugger", "task": "定位失败根因"}),
        AIMessage(content="完成"),
    ])

    # 默认批准集 = ("coder",)，debugger 不在其中
    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
    config = {"configurable": {"thread_id": "app3"}}
    result = graph.invoke(_initial(), config)
    assert result["status"] == "finished"
    snap = graph.get_state(config)
    assert not snap.interrupts
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1 and "[委派完成]" in tmsgs[0].content


def test_tester_does_not_interrupt_by_default(tmp_path):
    """P4-2: 默认批准集仅 coder——tester（测试验证）不 interrupt，一次跑完。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "tester": [
            _tool_call(1, "write_file", {"path": "test_main.py", "content": TEST_CODE}),
            AIMessage(content="test_main.py 已写入，pytest 通过"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "tester", "task": "为快排写测试"}),
        AIMessage(content="完成"),
    ])

    # 默认批准集 = ("coder",)，tester 不在其中 → 不 interrupt
    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
    config = {"configurable": {"thread_id": "app5"}}
    result = graph.invoke(_initial(), config)
    assert result["status"] == "finished"
    snap = graph.get_state(config)
    assert not snap.interrupts
    tmsgs = _delegate_tool_msgs(result)
    assert len(tmsgs) == 1 and "[委派完成]" in tmsgs[0].content
    assert "test_quicksort" in ws.read_text("test_main.py")


def test_delegate_interrupts_on_configured_only(tmp_path):
    """P4-2: 只对配置集中断——coder 不中断、tester 中断；且只挂 1 个 interrupt。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="main.py 已写入"),
        ],
        "tester": [
            _tool_call(1, "write_file", {"path": "test_main.py", "content": TEST_CODE}),
            AIMessage(content="test_main.py 已写入"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排"}),
        _tool_call(2, "delegate", {"specialist": "tester", "task": "为快排写测试"}),
        AIMessage(content="完成"),
    ])

    graph = build_supervisor_graph(
        _settings(tmp_path), ws, make_llm=make, require_approval_for=("tester",)
    )
    config = {"configurable": {"thread_id": "app4"}}
    graph.invoke(_initial(), config)
    snap = graph.get_state(config)
    # 只挂在 tester 这一个 interrupt 上；coder 已先完成并提交 ToolMessage
    assert snap.next == ("tools",)
    assert len(snap.interrupts) == 1
    assert snap.interrupts[0].value["specialist"] == "tester"
    assert "写测试" in snap.interrupts[0].value["task"]

    # 批准后全部完成
    result = graph.invoke(Command(resume="yes"), config)
    assert result["status"] == "finished"
    assert "quicksort" in ws.read_text("main.py")
    assert "test_quicksort" in ws.read_text("test_main.py")
