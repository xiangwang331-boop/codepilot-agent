"""CommandRunner：run_command 的执行宿主（local subprocess / docker sandbox）。

P5 起 subprocess 的唯一宿主从 tools/terminal.py 迁到这里：
- LocalCommandRunner  = 原 run_command 的本机 subprocess 实现（行为字节不变，默认）。
- DockerCommandRunner = 把命令丢进长驻 Docker 容器执行（SANDBOX_MODE=docker）。
文件操作仍由 WorkspaceManager 负责（不改）；这里只负责"执行命令"。

工具契约保持：command + timeout → "exit_code=..\n输出" / "ERROR: .."（异常回流给 LLM 自纠）。
Docker 只用 Docker CLI（argv 直传，不 shell），不新增 docker-py 等 Python 依赖。
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable
from uuid import uuid4

DEFAULT_IMAGE = "codepilot-sandbox:py3.12"
DEFAULT_WORKDIR = "/workspace"

# P7: 所有 CodePilot 起的沙箱容器都带这个 label，服务端启动时按 label 清扫孤儿
# （`docker ps -a --filter label=codepilot.managed=1`），比按名字前缀筛安全。
CONTAINER_LABEL = "codepilot.managed=1"

# docker CLI 基础设施错误标记（daemon 挂 / 镜像缺 / 容器不存在等），
# 命中即判定为"基础设施问题"而非 agent 代码问题，回流 "ERROR: Docker ..."。
_INFRA_MARKERS = (
    "Error response from daemon",
    "Cannot connect to the Docker daemon",
    "error during connect",
    "Is the docker daemon running",
    "No such image",
    "No such container",
    "is not running",
    "is already in use",
)


class DockerSandboxError(RuntimeError):
    """Docker 基础设施问题（daemon 未起 / 镜像缺失等），不是 agent 代码的问题。"""


def _docker_cli(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """直调 docker CLI（argv 不 shell）。模块级函数 = 清扫逻辑的测试注入点。"""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def sweep_orphan_containers(
    *, run_cli: Callable[[list[str], int], subprocess.CompletedProcess] | None = None
) -> tuple[int, str | None]:
    """删掉**所有带 `CONTAINER_LABEL` 的容器**，返回 (删除数量, 错误信息或 None)。

    服务端**启动时**跑一次：那一刻本进程还没建过任何容器，所以带 label 的必然是上一个
    进程崩溃 / 被硬杀（`atexit` 没跑到）留下的孤儿 —— 它们各占一份 bind mount 和一份
    workspace 目录，不清就会随重启累积。

    刻意按 **label** 筛而不是 `--filter name=codepilot-sandbox-`：名字前缀会把用户自己
    手起的同名容器一并误删（label 是「我们起的」这件事本身，名字不是）。

    ⚠️ 前提：同一时刻只有一个 CodePilot 进程。多进程并存时后启动的会把先启动的活容器
    当孤儿删掉 —— 本机单用户开发下成立，多实例部署前必须先解决归属问题（见 DESIGN.md）。

    任何失败都**不抛**：清扫是尽力而为的卫生工作，不该因为它挡住服务启动。
    """
    cli = run_cli or _docker_cli
    try:
        listed = cli(["docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}"], 30)
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"
    if listed.returncode != 0:
        return 0, (listed.stderr or "").strip()[:300] or f"exit_code={listed.returncode}"
    ids = [line.strip() for line in (listed.stdout or "").splitlines() if line.strip()]

    removed = 0
    for container_id in ids:
        try:
            proc = cli(["docker", "rm", "-f", container_id], 60)
        except Exception as e:  # noqa: BLE001
            return removed, f"{type(e).__name__}: {e}"
        if proc.returncode == 0:
            removed += 1
        else:
            return removed, (proc.stderr or "").strip()[:300] or f"exit_code={proc.returncode}"
    return removed, None


def _format_run_output(returncode: int, stdout: str, stderr: str) -> str:
    """把一次命令的 returncode/stdout/stderr 拼成 run_command 契约文本（local/docker 共用）。"""
    out: list[str] = []
    if stdout:
        out.append(stdout.rstrip())
    if stderr:
        out.append("[stderr]\n" + stderr.rstrip())
    body = "\n".join(out) if out else "(无输出)"
    return f"exit_code={returncode}\n{body}"


class LocalCommandRunner:
    """本机 subprocess 执行（原 run_command 逻辑原样迁移，行为零变化）。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _env_with_runtime_python() -> dict:
        """把当前 Python 解释器目录放到 PATH 最前，保证 python/pytest 可用。"""
        env = os.environ.copy()
        scripts_dir = str(Path(sys.executable).parent)
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
        return env

    def run(self, command: str, timeout: int = 60) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=self._env_with_runtime_python(),
            )
            return _format_run_output(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return f"ERROR: 命令超时（>{timeout}s）: {command}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"


class DockerCommandRunner:
    """把命令丢进长驻 Docker 容器执行（run_command 的 docker 模式）。

    会话级容器：`with DockerCommandRunner(...) as r:` 进入时 docker run -d（bind-mount
    host workspace → /workspace）启动，__exit__ 时 docker rm -f 清理（正常/异常都走
    with 语义）；另注册 atexit.stop() 兜底进程硬杀。每次 run() 走
    `docker exec <ctr> bash -lc <cmd>`，工作目录为容器内 /workspace。
    任何 Docker 基础设施错误（daemon 挂 / 镜像缺失 / exec 报错）都转成 "ERROR: Docker ..."
    文本回流给 LLM 自纠，不炸父图。
    """

    def __init__(
        self,
        root: Path,
        image: str = DEFAULT_IMAGE,
        container_name: str | None = None,
        container_workdir: str = DEFAULT_WORKDIR,
    ):
        self.root = Path(root)
        self.image = image
        self.container_name = container_name or f"codepilot-sandbox-{uuid4().hex[:12]}"
        self.container_workdir = container_workdir
        self._started = False
        atexit.register(self.stop)

    # ---- docker CLI 薄封装 ----

    def _run_cli(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        """直调 docker CLI（argv 不 shell，避免 Windows host 引号地狱）。"""
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

    def _run_args(self) -> list[str]:
        host = str(self.root).replace("\\", "/")  # Windows drive-letter + 正斜杠，Docker Desktop 最稳形态
        return [
            "docker", "run", "-d",
            "--name", self.container_name,
            # P7: 打上归属标签，服务端启动时按它清扫崩溃遗留的孤儿容器。
            # 刻意不用 `--filter name=codepilot-sandbox-`——那会误伤用户自己起的同名容器。
            "--label", CONTAINER_LABEL,
            "-v", f"{host}:{self.container_workdir}",
            "-w", self.container_workdir,
            self.image, "sleep", "infinity",
        ]

    def _ensure_started(self) -> None:
        if self._started:
            return
        inspect = self._run_cli(["docker", "image", "inspect", self.image], 30)
        if inspect.returncode != 0:
            # image inspect 失败要先判"镜像缺失"（daemon 正常时报 No such image），
            # 再判"daemon 挂了"——两种提示一个指导 build、一个指导起 Docker，不可混淆。
            err = (inspect.stderr or "").strip()
            if "No such image" in err:
                raise DockerSandboxError(
                    f"镜像 {self.image} 不存在，请先构建: "
                    f"docker build -t {self.image} -f Dockerfile.sandbox ."
                )
            if any(m in err for m in _INFRA_MARKERS):
                raise DockerSandboxError(f"Docker daemon 不可用: {err[:300]}")
            raise DockerSandboxError(f"docker image inspect 失败: {err[:300]}")
        run = self._run_cli(self._run_args(), 120)
        if run.returncode != 0:
            # 崩溃遗留的同名容器 → 删掉重试一次
            if "is already in use" in (run.stderr or ""):
                self._run_cli(["docker", "rm", "-f", self.container_name], 60)
                run = self._run_cli(self._run_args(), 120)
            if run.returncode != 0:
                raise DockerSandboxError(f"docker run 失败: {(run.stderr or '').strip()[:500]}")
        self._started = True

    def stop(self) -> None:
        try:
            # P7: 释放 atexit 持有的强引用。`atexit.register(self.stop)` 注册的是绑定方法，
            # 它让**对象本身永不回收**（实测 alive=True）——CLI 一个进程一个容器无所谓，
            # 但服务端每会话一个容器，不摘就是持续泄漏。stop 幂等，重复调用无副作用。
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001
            pass
        if not self._started:
            return
        try:
            self._run_cli(["docker", "rm", "-f", self.container_name], 60)
        except Exception:  # noqa: BLE001  清理失败静默（进程将退出 / 下次 with 再兜）
            pass
        finally:
            self._started = False

    def __enter__(self) -> "DockerCommandRunner":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def run(self, command: str, timeout: int = 60) -> str:
        try:
            self._ensure_started()
        except DockerSandboxError as e:
            return f"ERROR: Docker sandbox 启动失败: {e}"
        try:
            proc = self._run_cli(
                ["docker", "exec", self.container_name, "bash", "-lc", command], timeout
            )
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: Docker 命令超时（>{timeout}s，容器内进程可能仍在运行，"
                f"会话结束时会 rm -f 连根清理）: {command}"
            )
        stderr = proc.stderr or ""
        if proc.returncode != 0 and any(m in stderr for m in _INFRA_MARKERS):
            return f"ERROR: Docker {stderr.strip()[:500]}"
        return _format_run_output(proc.returncode, proc.stdout or "", stderr)
