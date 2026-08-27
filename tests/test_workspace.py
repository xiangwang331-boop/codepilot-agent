"""WorkspaceManager 的路径守卫测试。"""
from __future__ import annotations

import pytest

from workspace.manager import WorkspaceError, WorkspaceManager


@pytest.fixture
def ws(tmp_path):
    return WorkspaceManager(tmp_path / "ws")


def test_resolve_relative(ws):
    assert ws.resolve("a/b.py") == ws.root / "a" / "b.py"


def test_resolve_root(ws):
    assert ws.resolve(".") == ws.root


def test_reject_absolute(ws):
    # Windows 绝对路径
    with pytest.raises(WorkspaceError):
        ws.resolve("C:/Windows/System32")


def test_reject_traversal(ws):
    with pytest.raises(WorkspaceError):
        ws.resolve("../secret.txt")
    with pytest.raises(WorkspaceError):
        ws.resolve("a/../../secret.txt")


def test_resolve_never_escapes_root(ws):
    """不变量：任何输入要么被拒绝，要么解析结果仍在 root 内。"""
    adversarial = ["../x", "../../x", "a/../../b/../../../x", "..\\..\\x"]
    for p in adversarial:
        try:
            out = ws.resolve(p)
        except WorkspaceError:
            continue
        assert out.is_relative_to(ws.root), f"逃逸: {p} -> {out}"


def test_write_read_roundtrip(ws):
    ws.write_text("hello/world.txt", "hi")
    assert ws.read_text("hello/world.txt") == "hi"


def test_list_dir(ws):
    ws.write_text("a.py", "1")
    ws.write_text("sub/b.py", "2")
    assert "a.py" in ws.list_dir(".")
    assert "sub" in ws.list_dir(".")
    assert "sub/b.py" in ws.list_dir(".", recursive=True)


def test_delete(ws):
    ws.write_text("x.txt", "x")
    ws.delete("x.txt")
    assert not ws.exists("x.txt")


def test_read_missing_raises(ws):
    with pytest.raises(WorkspaceError):
        ws.read_text("nope.txt")
