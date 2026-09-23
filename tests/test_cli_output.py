"""P7-3: CLI 输出契约 —— 抽出 runtime/ 之后，main.py 的 stdout 必须一字不变。

P7 把「装配」和「interrupt 循环」搬进了 `runtime/assembly.py` + `runtime/driver.py`
（服务端复用同一份）。这个重构最容易悄悄改掉的就是**打印的内容和顺序**，
所以这里端到端跑真正的 `main.main()`（FakeLLM 驱动，无网络、无 Docker、无 PG），
把输出钉住。

`test_cli_output_matches_pre_p7_golden` 的基线是重构前的逐字输出
（重构前后各跑一遍同样脚本、diff 为空）。
"""
from __future__ import annotations

import sys

import pytest
from langchain_core.messages import AIMessage

from agent import supervisor
from conftest import FakeLLM, QUICKSORT_CODE, _tool_call
from events.events import bind_thread, emitter
from runtime import assembly as assembly_mod

import main as main_mod

# ---------------------------------------------------------------- 脚手架


def _patch_env(monkeypatch, tmp_path):
    """把 Settings.from_env 能读到的项全指到 tmp，且强制 local/sqlite（不碰 Docker/PG）。"""
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("SANDBOX_MODE", "local")
    monkeypatch.setenv("PERSISTENCE_BACKEND", "sqlite")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "data" / "cp.db"))


def _patch_graph(monkeypatch, per_build: list[dict]):
    """把 assembly 的 build_supervisor_graph 换成「按角色注入 FakeLLM」的版本。

    `per_build` 每次建图消费一项：CLI 每次 `main()` 都会重建图（含 --resume 那次），
    每次都需要一份没被消费过的新脚本。
    """
    real = supervisor.build_supervisor_graph
    queue = list(per_build)

    def wrapper(settings, ws, make_llm=None, **kw):
        fakes = {role: FakeLLM(script) for role, script in queue.pop(0).items()}

        def make(role):
            return fakes.setdefault(role, FakeLLM([]))

        return real(settings, ws, make_llm=make, **kw)

    monkeypatch.setattr(assembly_mod, "build_supervisor_graph", wrapper)


def _delegate_to_coder():
    return {
        "Supervisor": [
            _tool_call(1, "delegate", {"specialist": "coder", "task": "写一个 quicksort 到 main.py"}),
            AIMessage(content="完成：Coder 已写好快排。"),
        ],
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入 main.py"),
        ],
    }


def _reset_globals() -> None:
    """把 main() 动过的进程级状态还原。

    `main.py` 刻意不 reset 线程绑定（生产一个进程只跑一个会话），但测试是在同一个
    进程里反复跑真 `main()`：不清就会把会话 ID 泄漏给后面所有用例（P6 的
    `test_event_carries_bound_thread_id`、P7-1 的 thread_id 归属断言都会被带偏）。
    """
    emitter.clear()
    bind_thread("")


def _run_cli(monkeypatch, tmp_path, capsys, argv: list[str]) -> str:
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    _reset_globals()
    try:
        main_mod.main()
        return capsys.readouterr().out
    finally:
        _reset_globals()


def _assert_order(out: str, *needles: str) -> None:
    pos = -1
    for needle in needles:
        i = out.find(needle, pos + 1)
        assert i > pos, f"输出里 {needle!r} 缺失或顺序不对：\n{out}"
        pos = i


# ---------------------------------------------------------------- 顺序契约


def test_cli_print_order_unchanged(tmp_path, monkeypatch, capsys):
    """`=== 任务 ===` → `会话 ID:` → `[需要批准]` → `=== 结果 ===` 的相对顺序。"""
    _patch_env(monkeypatch, tmp_path)
    _patch_graph(monkeypatch, [_delegate_to_coder()])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")

    out = _run_cli(monkeypatch, tmp_path, capsys, ["多 agent 协作完成快排"])

    _assert_order(
        out,
        "[Supervisor] 开始任务",
        "=== 任务 ===",
        "多 agent 协作完成快排",
        "会话 ID: ",
        "[需要批准] ",
        "=== 结果 ===",
        "完成：Coder 已写好快排。",
    )
    # local 模式不该出现沙箱那一行；sqlite 模式也不该出现持久化那一行
    assert "沙箱模式" not in out
    assert "持久化：" not in out


def test_cli_resume_prints_recovered_message_count(tmp_path, monkeypatch, capsys):
    """--resume 的输出形状：先报已有消息数，再报追加指令。"""
    _patch_env(monkeypatch, tmp_path)
    _patch_graph(monkeypatch, [_delegate_to_coder()])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    out = _run_cli(monkeypatch, tmp_path, capsys, ["多 agent 协作完成快排"])
    thread_id = out.split("会话 ID: ")[1].splitlines()[0].strip()

    _patch_graph(monkeypatch, [{"Supervisor": [AIMessage(content="续跑完成。")]}])
    out = _run_cli(monkeypatch, tmp_path, capsys, ["--resume", thread_id, "再写一个 b.py"])

    _assert_order(
        out,
        f"=== 恢复会话 {thread_id}（已有 5 条消息）===",
        "追加指令: 再写一个 b.py",
        "=== 结果 ===",
    )
    # resume 不重传初始 messages：不该再发 AGENT_STARTED（那是新任务才有的）
    assert "[Supervisor] 开始任务" not in out


def test_cli_resume_missing_session_reports_error(tmp_path, monkeypatch, capsys):
    """恢复一个不存在的会话：报错、不打结果段。"""
    _patch_env(monkeypatch, tmp_path)
    _patch_graph(monkeypatch, [{"Supervisor": [AIMessage(content="x")]}])

    out = _run_cli(monkeypatch, tmp_path, capsys, ["--resume", "no-such-thread"])

    assert "不存在或已清理" in out
    assert "=== 结果 ===" not in out


def test_cli_rejection_keeps_loop_and_finishes(tmp_path, monkeypatch, capsys):
    """拒绝批准走同一条循环：driver 把空值/拒绝值归一化成 "no"。"""
    _patch_env(monkeypatch, tmp_path)
    _patch_graph(monkeypatch, [_delegate_to_coder()])
    monkeypatch.setattr("builtins.input", lambda *a: "no")

    out = _run_cli(monkeypatch, tmp_path, capsys, ["多 agent 协作完成快排"])

    assert "[需要批准] " in out
    assert "=== 结果 ===" in out
    assert not (tmp_path / "ws" / "main.py").exists(), "拒绝后子图不该执行"


def test_cli_unknown_args_still_error_before_assembly(tmp_path, monkeypatch, capsys):
    """没给需求且没 --resume：在装配之前就报错返回（不建图、不碰 workspace）。"""
    _patch_env(monkeypatch, tmp_path)
    out = _run_cli(monkeypatch, tmp_path, capsys, [])
    assert "请提供开发需求" in out
    assert "=== 任务 ===" not in out


# ---------------------------------------------------------------- 逐字基线

# 下面这段**逐字来自重构前的 main.py**：把 `git show HEAD:main.py` 与新版各跑一遍
# 同样的两段场景（新任务+批准、--resume 追加指令），用同一份 FakeLLM 脚本驱动，
# `diff out_old.txt out_new.txt` 为空。它比上面的「顺序断言」强：顺序对了但文案/空行
# 变了同样会挂，是 P7-3「CLI 字节不变」这条硬门的固化。
#
# 两处刻意保留的「怪味」：
# 1. RESUME 段里 `给出最终回答` / `完成` 各出现两次 —— 一个进程里跑第二次 main()
#    会再挂一个 print 监听器（真实 CLI 不会这样跑，这里为了在一段输出里覆盖两条路径）。
# 2. 审批那行的 prompt 不在输出里 —— 测试把 `input` 换成了 stub，不写 prompt。
PRE_P7_GOLDEN_LINES = [
    '[Supervisor] 开始任务',
    '',
    '=== 任务 ===',
    '多 agent 协作完成快排',
    '',
    '会话 ID: {thread_id}',
    '[Supervisor] 决定调用 1 个工具: delegate',
    '[Supervisor] 调用 tool: delegate',
    '',
    '[需要批准] 是否批准委派 coder 执行任务？',
    '任务: 写一个 quicksort 到 main.py',
    '[Supervisor] 调用 tool: delegate',
    '[coder] 开始任务',
    '[coder] 决定调用 1 个工具: write_file',
    '[coder] 调用 tool: write_file',
    "    [Tool] write_file(path='main.py', content='def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    pivot = arr[len(arr) // 2]\\n    left = [x for x in arr if x < pivot]\\n    middle = [x for x in arr if x == pivot]\\n    right = [x for x in arr if x > pivot]\\n    return quicksort(left) + middle + quicksort(right)\\n')",
    '[coder] 给出最终回答',
    '[coder] 完成',
    "    [Tool] delegate(specialist='coder', task='写一个 quicksort 到 main.py')",
    '[Supervisor] 给出最终回答',
    '',
    '=== 结果 ===',
    '[Supervisor] 完成',
    '完成：Coder 已写好快排。',
    '',
    '########## RESUME ##########',
    '',
    '=== 恢复会话 {thread_id}（已有 5 条消息）===',
    '追加指令: 再写一个 b.py',
    '[Supervisor] 给出最终回答',
    '[Supervisor] 给出最终回答',
    '',
    '=== 结果 ===',
    '[Supervisor] 完成',
    '[Supervisor] 完成',
    '续跑完成。',
    '',
]


def test_cli_output_matches_pre_p7_golden(tmp_path, monkeypatch, capsys):
    """P7-3 的硬门：抽出 runtime/ 之后 CLI 输出必须逐字不变。"""
    _patch_env(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    _patch_graph(
        monkeypatch,
        [
            _delegate_to_coder(),
            {"Supervisor": [AIMessage(content="续跑完成。")]},
        ],
    )
    _reset_globals()  # 只清一次：第二次 main() 会再挂一个 print 监听器，与基线一致
    try:
        monkeypatch.setattr(sys, "argv", ["main.py", "多 agent 协作完成快排"])
        main_mod.main()
        out_new = capsys.readouterr().out
        thread_id = out_new.split("会话 ID: ")[1].splitlines()[0].strip()

        monkeypatch.setattr(sys, "argv", ["main.py", "--resume", thread_id, "再写一个 b.py"])
        main_mod.main()
        out_resume = capsys.readouterr().out
    finally:
        _reset_globals()

    combined = (out_new + "\n########## RESUME ##########\n" + out_resume).replace(
        thread_id, "{thread_id}"
    )
    # "\n".join 而非 "".join(line + "\n")：列表末元素是 split 出来的空串，
    # 它本身就代表最后那个换行，再加一个会多出空行。
    assert combined == "\n".join(PRE_P7_GOLDEN_LINES)
