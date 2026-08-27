"""P2: Supervisor 编排测试 —— 按角色注入 FakeLLM，确定性驱动多 agent 协作。

每个测试构造一个真实 WorkspaceManager + 各角色 FakeLLM script：
supervisor 的 script 决定何时 delegate、委派给谁；specialist 的 script
决定子图内部动作。全程无需 API key / 网络。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from agent.specialists import build_supervisor_prompt
from agent.supervisor import build_supervisor_graph
from conftest import FakeLLM, QUICKSORT_CODE, TEST_CODE, _tool_call
from config.settings import Settings
from events.events import emitter
from tools.registry import build_tools_subset
from workspace.manager import WorkspaceManager


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

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
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

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
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

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
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
        _tool_call(1, "delegate", {"specialist": "planner", "task": "做个计划"}),
        _tool_call(2, "delegate", {"specialist": "coder", "task": "写快排"}),
        AIMessage(content="planner 不可用，改派 Coder，完成。"),
    ])

    graph = build_supervisor_graph(_settings(tmp_path), ws, make_llm=make)
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
