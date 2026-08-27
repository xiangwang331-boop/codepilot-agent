"""工具层的 schema / 参数校验 / 异常处理测试。"""
from __future__ import annotations

import pytest

from tools.registry import build_tools, build_tools_map
from workspace.manager import WorkspaceManager


@pytest.fixture
def ws(tmp_path):
    return WorkspaceManager(tmp_path / "ws")


@pytest.fixture
def tools_map(ws):
    return build_tools_map(ws)


def test_all_expected_tools_present(tools_map):
    expected = {
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "delete_file",
        "search_code",
        "run_command",
    }
    assert expected <= set(tools_map)


def test_every_tool_has_schema(ws):
    for t in build_tools(ws):
        assert t.args_schema is not None
        schema = t.args_schema.model_json_schema()
        assert "properties" in schema


def test_write_then_read(tools_map):
    out = tools_map["write_file"].invoke({"path": "main.py", "content": "print('hi')"})
    assert "已写入" in out
    content = tools_map["read_file"].invoke({"path": "main.py"})
    assert "print('hi')" in content


def test_missing_required_arg_is_rejected(tools_map):
    with pytest.raises(Exception):
        tools_map["read_file"].invoke({})


def test_unknown_path_returns_error_string(tools_map):
    out = tools_map["read_file"].invoke({"path": "nope.txt"})
    assert out.startswith("ERROR:")


def test_edit_file(tools_map):
    tools_map["write_file"].invoke({"path": "a.py", "content": "x = 1\n"})
    out = tools_map["edit_file"].invoke(
        {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}
    )
    assert "替换" in out
    assert tools_map["read_file"].invoke({"path": "a.py"}) == "x = 2\n"


def test_search_code(tools_map):
    tools_map["write_file"].invoke({"path": "a.py", "content": "def quicksort():\n    pass\n"})
    out = tools_map["search_code"].invoke({"query": "quicksort"})
    assert "a.py:1" in out


def test_run_command_returns_exit_code(ws):
    tools_map = build_tools_map(ws)
    out = tools_map["run_command"].invoke({"command": "echo hello"})
    assert "exit_code=" in out
