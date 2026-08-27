"""共享测试基件：脚本化 FakeLLM + 工具调用构造器 + 公共常量。

P0 各测试文件各自定义了 FakeLLM；P2 起统一放这里，
供 test_agent_loop / test_persistence / test_supervisor 复用。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeLLM:
    """按预设脚本吐 AIMessage，确定性驱动 ReAct 循环。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if not self.script:
            return AIMessage(content="(fallback done)")
        return self.script.pop(0)


def _tool_call(idx, name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
    )


QUICKSORT_CODE = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n"
    "    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n"
)
TEST_CODE = (
    "from main import quicksort\n\n"
    "def test_quicksort():\n"
    "    assert quicksort([3, 1, 2]) == [1, 2, 3]\n"
)
