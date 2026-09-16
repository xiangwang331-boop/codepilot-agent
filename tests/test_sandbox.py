"""P5: CommandRunner 单元测试 —— Local/Docker 双宿主 + 注入链路，全 fake 不碰 docker daemon。

覆盖（无需 daemon，monkeypatch runner._run_cli 为假 CLI）：
- 默认 local：不传 runner = 本机 subprocess（P0–P4 行为零变化）。
- Local runner 的 stderr 段 / exit_code 契约。
- terminal/registry/supervisor 三层 runner 注入穿透。
- DockerCommandRunner：docker run -d 参数（-v bind mount / -w /workspace / sleep infinity）、
  image inspect、exec 参数、exit_code 透传、infra 错误回流 ERROR 文本不炸图、
  镜像缺失 build 提示、with 进出启动/清理幂等。
- docker 模式下 run_command description 含容器路径引导。
- Settings SANDBOX_MODE/SANDBOX_IMAGE 读取 + 非法值 fallback local。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from agent.condense import make_condense_node
from agent.specialists import build_supervisor_prompt
from agent.supervisor import build_supervisor_graph
from conftest import FakeLLM, _tool_call
from config import settings as settings_mod
from config.settings import Settings
from tools.command_runner import (
    CONTAINER_LABEL,
    DEFAULT_IMAGE,
    DEFAULT_WORKDIR,
    DockerCommandRunner,
    DockerSandboxError,
    LocalCommandRunner,
    _format_run_output,
    sweep_orphan_containers,
)
from tools.registry import build_tools, build_tools_map
from workspace.manager import WorkspaceManager

_NO_CONDENSE = make_condense_node(trigger_count=10**9)


# ---------- 小工具 ----------

def _settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="",
        llm_base_url="",
        llm_model="fake",
        workspace_root=tmp_path / "ws",
        max_iterations=10,
    )


def _initial(task="跑一次 run_command"):
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


class FakeRunner:
    """记录 (command, timeout)，返回固定契约文本；模拟任一台 run_command 宿主。"""

    def __init__(self, result="exit_code=0\n(无输出)"):
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def run(self, command: str, timeout: int = 60) -> str:
        self.calls.append((command, timeout))
        return self.result


def _docker_cli(responses: dict) -> tuple:
    """假 docker CLI：按子命令返回脚本化结果，并记录所有调用。"""
    calls: list[list[str]] = []

    def fake(args: list[str], timeout: int) -> subprocess.CompletedProcess:
        calls.append(list(args))
        sub = args[1] if len(args) > 1 else ""
        rc, out, err = responses.get(sub, (0, "", ""))
        return subprocess.CompletedProcess(args=list(args), returncode=rc, stdout=out, stderr=err)

    return fake, calls


# ---------- 纯函数 / Local ----------

def test_format_run_output_stderr_segment_and_empty_body():
    assert _format_run_output(0, "hello", "oops") == "exit_code=0\nhello\n[stderr]\noops"
    assert _format_run_output(1, "", "") == "exit_code=1\n(无输出)"


def test_local_runner_echo_exit_code(tmp_path):
    ws = WorkspaceManager(tmp_path / "ws")
    out = LocalCommandRunner(ws.root).run("echo hello")
    assert "exit_code=0" in out
    assert "hello" in out


def test_default_no_runner_is_local(tmp_path):
    """不传 runner → run_command 走本机 subprocess（行为与 P0–P4 一致）。"""
    ws = WorkspaceManager(tmp_path / "ws")
    tool = next(t for t in build_tools(ws) if t.name == "run_command")
    out = tool.invoke({"command": "echo hello", "timeout": 30})
    assert "exit_code=0" in out
    assert "hello" in out


# ---------- runner 注入穿透（terminal / registry） ----------

def test_terminal_build_tools_injects_runner(tmp_path):
    ws = WorkspaceManager(tmp_path / "ws")
    fake = FakeRunner()
    tool = next(t for t in build_tools(ws, runner=fake) if t.name == "run_command")
    out = tool.invoke({"command": "echo hi", "timeout": 30})
    assert out == "exit_code=0\n(无输出)"
    assert fake.calls == [("echo hi", 30)]


def test_registry_map_and_subset_thread_runner(tmp_path):
    ws = WorkspaceManager(tmp_path / "ws")
    fake = FakeRunner()
    names = set(build_tools_map(ws, runner=fake))  # map 是 dict：迭代 keys
    assert "run_command" in names
    subset = build_tools(ws, runner=fake)
    assert any(t.name == "run_command" for t in subset)


# ---------- DockerCommandRunner：单测（全 fake CLI） ----------

def test_docker_run_creates_container_then_exec_cleanup(tmp_path):
    """docker run -d 参数（-v mount / -w /workspace / sleep infinity）+ exec + 清理顺序。"""
    wsroot = Path(tmp_path) / "ws"
    host = str(wsroot).replace("\\", "/")
    runner = DockerCommandRunner(wsroot, image="img:1", container_name="ctr-abc")
    fake, calls = _docker_cli({
        "image": (0, "", ""),                 # inspect 通过
        "run": (0, "abc123\n", ""),           # 容器启动
        "exec": (0, "/workspace\n", ""),      # pwd
        "rm": (0, "ctr-abc\n", ""),           # 清理
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    with runner as r:
        out = r.run("pwd")

    assert out == "exit_code=0\n/workspace"
    kinds = [c[1] for c in calls]
    assert kinds == ["image", "run", "exec", "rm"]  # inspect → run → exec → 退出清理
    run_call = next(c for c in calls if c[1] == "run")
    assert run_call == [
        "docker", "run", "-d", "--name", "ctr-abc",
        # P7: 每个沙箱容器都带归属 label，服务端起服务时按它清扫孤儿
        "--label", CONTAINER_LABEL,
        "-v", f"{host}:{DEFAULT_WORKDIR}", "-w", DEFAULT_WORKDIR,
        "img:1", "sleep", "infinity",
    ]
    exec_call = next(c for c in calls if c[1] == "exec")
    assert exec_call == ["docker", "exec", "ctr-abc", "bash", "-lc", "pwd"]
    # bind mount 用的是 workspace 真实路径（run_call 已断言 -v {host}:/workspace）


def test_docker_command_exit_code_passthrough(tmp_path):
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-exit")
    fake, _ = _docker_cli({
        "image": (0, "", ""),
        "run": (0, "id\n", ""),
        "exec": (2, "compile error\n", ""),   # 非零退出 → 契约原样带出 exit_code
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    with runner as r:
        out = r.run("python x.py")
    assert out == "exit_code=2\ncompile error"


def test_docker_daemon_down_returns_error_text(tmp_path):
    """daemon 挂：run() 返回 "ERROR: Docker sandbox 启动失败"，绝不抛异常炸图。"""
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-down")
    fake, _ = _docker_cli({
        "image": (1, "", "Cannot connect to the Docker daemon. Is the docker daemon running?"),
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    out = runner.run("pwd")  # 不进 with，直接触达 _ensure_started
    assert out.startswith("ERROR: Docker sandbox 启动失败")
    assert "daemon" in out
    with pytest.raises(DockerSandboxError):
        runner.__enter__()


def test_docker_missing_image_gives_build_hint(tmp_path):
    """镜像缺失：提示先 build（不是误报 daemon 挂）。"""
    runner = DockerCommandRunner(tmp_path / "ws", image=DEFAULT_IMAGE, container_name="ctr-noimg")
    fake, _ = _docker_cli({
        "image": (1, "", f"Error response from daemon: No such image: {DEFAULT_IMAGE}"),
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    out = runner.run("pwd")
    assert out.startswith("ERROR: Docker sandbox 启动失败")
    assert "请先构建" in out
    assert "docker build" in out


def test_docker_stop_idempotent(tmp_path):
    """with 退出 rm 一次；再 stop 幂等（不再重复 rm）。"""
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-clean")
    fake, calls = _docker_cli({
        "image": (0, "", ""),
        "run": (0, "id\n", ""),
        "rm": (0, "gone\n", ""),
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    with runner:
        pass
    assert runner._started is False
    runner.stop()  # 幂等：不再触发 rm
    assert sum(1 for c in calls if c[1] == "rm") == 1


def test_docker_is_already_in_use_retries_once(tmp_path):
    """同名容器遗留：docker run 报 is already in use → rm -f 后重试一次成功。"""
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-stale")
    calls: list[list[str]] = []
    run_attempts = {"n": 0}

    def fake(args: list[str], timeout: int) -> subprocess.CompletedProcess:
        calls.append(list(args))
        sub = args[1] if len(args) > 1 else ""
        if sub == "image":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if sub == "run":
            run_attempts["n"] += 1
            if run_attempts["n"] == 1:  # 第一次：遗留容器撞名
                return subprocess.CompletedProcess(
                    args=args, returncode=1, stdout="", stderr=(
                        "docker: Error response from daemon: Conflict. The container name "
                        "ctr-stale is already in use by container x. You have to remove ..."
                    )
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="id\n", stderr="")
        if sub == "exec":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")
        if sub == "rm":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ctr-stale\n", stderr="")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    runner._run_cli = fake  # type: ignore[method-assign]

    with runner as r:
        out = r.run("echo ok")
    assert sum(1 for c in calls if c[1] == "run") == 2  # 首次撞名 + rm 后重试
    assert sum(1 for c in calls if c[1] == "rm") == 2  # 撞名清理一次 + 会话结束清理一次
    assert out == "exit_code=0\nok"


def test_docker_container_died_reports_infra_error(tmp_path):
    """exec 报 is not running（容器中途死了）→ 回流 "ERROR: Docker ..."，不是裸 exit_code。"""
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-died")
    fake, _ = _docker_cli({
        "image": (0, "", ""),
        "run": (0, "id\n", ""),
        "exec": (1, "", "Error response from daemon: container ctr-died is not running"),
    })
    runner._run_cli = fake  # type: ignore[method-assign]

    with runner as r:
        out = r.run("pwd")
    assert out.startswith("ERROR: Docker ")
    assert "not running" in out


def test_docker_description_mentions_workspace_relative_path(tmp_path):
    """docker 模式 description 要引导模型用相对路径、识别 /workspace/xxx。"""
    ws = WorkspaceManager(tmp_path / "ws")
    runner = DockerCommandRunner(ws.root)
    tool = next(t for t in build_tools(ws, runner=runner) if t.name == "run_command")
    assert "/workspace" in tool.description
    assert "相对路径" in tool.description
    # local 模式不带容器引导
    tool_local = next(t for t in build_tools(ws) if t.name == "run_command")
    assert "/workspace" not in tool_local.description


# ---------- Settings ----------

def test_settings_sandbox_mode_from_env(monkeypatch, tmp_path):
    # 屏蔽 .env 加载（关键坑 #28）：本机 .env 现在设了 SANDBOX_MODE=docker，
    # 不屏蔽的话这个用例测的是「本机 .env 写了什么」而不是「env 未设置 → 默认 local」。
    monkeypatch.setattr(settings_mod, "_load_dotenv", lambda: None)
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("SANDBOX_IMAGE", "my-img:1")
    s = Settings.from_env()
    assert s.sandbox_mode == "local"          # 未设置 → 默认 local
    assert s.sandbox_image == "my-img:1"

    monkeypatch.setenv("SANDBOX_MODE", "docker")
    assert Settings.from_env().sandbox_mode == "docker"
    assert Settings.from_env().sandbox_image == "my-img:1"


def test_settings_sandbox_mode_invalid_falls_back_local(monkeypatch):
    monkeypatch.setattr(settings_mod, "_load_dotenv", lambda: None)  # 同上，见关键坑 #28
    monkeypatch.setenv("SANDBOX_MODE", "kubernetes")
    s = Settings.from_env()
    assert s.sandbox_mode == "local"          # 非法值 → 告警回退 local，不静默


# ---------- supervisor 装配：runner 穿透到 specialist 子图 ----------

def test_supervisor_threads_runner_into_specialist(tmp_path):
    """command_runner 注入 → coder 子图的 run_command 真的打到注入的宿主。"""
    ws = WorkspaceManager(tmp_path / "ws")
    fake = FakeRunner()
    fakes = {
        "coder": FakeLLM([
            _tool_call(1, "run_command", {"command": "pytest -q"}),
            AIMessage(content="pytest 通过"),
        ]),
        "Supervisor": FakeLLM([
            _tool_call(1, "delegate", {"specialist": "coder", "task": "跑一遍测试"}),
            AIMessage(content="完成：Coder 已跑完测试。"),
        ]),
    }

    def make(role: str) -> FakeLLM:
        return fakes.setdefault(role, FakeLLM([]))

    graph = build_supervisor_graph(
        _settings(tmp_path), ws,
        make_llm=make,
        require_approval_for=(),
        condense_node=_NO_CONDENSE,
        command_runner=fake,
    )
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "sb1"}})

    assert result["status"] == "finished"
    assert fake.calls == [("pytest -q", 60)]  # 子图里的 run_command 走了注入的宿主


# ---------- P7-6 启动清扫（按 label 删遗留容器，全 fake 不碰 daemon） ----------

def test_sweep_removes_labelled_containers_and_filters_by_label():
    """清扫删掉带 label 的遗留容器，且**筛的是 label 不是名字前缀**。"""
    fake, calls = _docker_cli({"ps": (0, "aaa111\nbbb222\n", ""), "rm": (0, "", "")})

    removed, error = sweep_orphan_containers(run_cli=fake)

    assert (removed, error) == (2, None)
    listed = calls[0]
    assert listed[:2] == ["docker", "ps"]
    assert "-aq" in listed
    assert f"label={CONTAINER_LABEL}" in listed
    # 名字前缀筛会误伤用户自己起的同名容器（本机 5432 那个 codepilot-postgres 是别人的）
    assert not any("name=" in a for a in listed)
    assert calls[1:] == [["docker", "rm", "-f", "aaa111"], ["docker", "rm", "-f", "bbb222"]]


def test_sweep_no_orphans_is_a_silent_noop():
    fake, calls = _docker_cli({"ps": (0, "", "")})

    assert sweep_orphan_containers(run_cli=fake) == (0, None)
    assert len(calls) == 1  # 只查了一次，没发任何 rm


def test_sweep_never_raises_when_daemon_is_down():
    """清扫是尽力而为的卫生工作：daemon 挂了给提示即可，绝不挡住服务启动。"""
    fake, _ = _docker_cli({"ps": (1, "", "Cannot connect to the Docker daemon at ...")})
    removed, error = sweep_orphan_containers(run_cli=fake)
    assert removed == 0
    assert "Cannot connect to the Docker daemon" in error

    def boom(args, timeout):
        raise FileNotFoundError("docker 不在 PATH 里")

    removed, error = sweep_orphan_containers(run_cli=boom)
    assert removed == 0
    assert "FileNotFoundError" in error


def test_sweep_reports_partial_failure_without_raising():
    fake, _ = _docker_cli({"ps": (0, "aaa111\nbbb222\n", ""), "rm": (1, "", "container is locked")})

    removed, error = sweep_orphan_containers(run_cli=fake)

    assert removed == 0
    assert "container is locked" in error
