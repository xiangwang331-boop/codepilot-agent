"""P5: Docker 沙箱集成测试 —— 真实 daemon + 镜像（模块级 skipif 自动跳过）。

每个测试独立开一个会话级容器（DockerCommandRunner with 块），验证：
- 容器内 pwd == /workspace
- bind mount 双向可见：host 写文件 → 容器读到；容器写文件 → host 读到
- pytest 在容器内能跑 host bind 进来的测试并全过
- 会话结束容器已被删除（rm -f 生效）
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from tools.command_runner import (
    CONTAINER_LABEL,
    DEFAULT_IMAGE,
    DockerCommandRunner,
    sweep_orphan_containers,
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
    "from quick import quicksort\n\n"
    "def test_quicksort():\n"
    "    assert quicksort([3, 1, 2]) == [1, 2, 3]\n"
)


def _docker_ready() -> bool:
    """daemon 可达且沙箱镜像已构建才跑集成测试（无 daemon/镜像自动 skip）。"""
    try:
        p = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_IMAGE],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return p.returncode == 0
    except Exception:  # noqa: BLE001  docker CLI 缺失 / 超时
        return False


pytestmark = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"Docker daemon 或镜像 {DEFAULT_IMAGE} 不可用（先 docker build -f Dockerfile.sandbox .）",
)


def test_docker_pwd_is_workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    with DockerCommandRunner(root) as r:
        out = r.run("pwd")
    assert "exit_code=0" in out
    assert "/workspace" in out


def test_bind_mount_bidirectional(tmp_path):
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    # host → 容器可见
    (root / "from_host.txt").write_text("hello from host", encoding="utf-8")
    with DockerCommandRunner(root) as r:
        assert "hello from host" in r.run("cat from_host.txt")
        # 容器 → host 可见（写文件后再 cat 自证，随后 host 侧读）
        assert "hi from container" in r.run(
            "printf 'hi from container' > from_container.txt && cat from_container.txt"
        )
    assert (root / "from_container.txt").read_text(encoding="utf-8") == "hi from container"


def test_pytest_runs_inside_sandbox(tmp_path):
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    (root / "quick.py").write_text(QUICKSORT_CODE, encoding="utf-8")
    (root / "test_quick.py").write_text(TEST_CODE, encoding="utf-8")
    with DockerCommandRunner(root) as r:
        out = r.run("pytest -q")
    assert "exit_code=0" in out
    assert "passed" in out


def test_container_removed_after_session(tmp_path):
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    runner = DockerCommandRunner(root)
    with runner as r:
        assert r.run("echo ok").startswith("exit_code=0")
    # with 退出后容器应已被 rm -f：container inspect 必须失败
    gone = subprocess.run(
        ["docker", "container", "inspect", runner.container_name],
        capture_output=True,
        text=True,
    )
    assert gone.returncode != 0


# ---------- P7-6 启动清扫（真 daemon） ----------

def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60)


def _exists(name: str) -> bool:
    return _docker("container", "inspect", name).returncode == 0


def test_sweep_removes_orphans_but_spares_foreign_containers():
    """带 label 的遗留容器被回收；**不带 label 的容器毫发无损**。

    ⚠️ 本用例会真删带 label 的容器 —— 别在本地正跑着 CodePilot 服务（有活会话）时跑它。
    """
    orphan = f"codepilot-sweep-orphan-{uuid4().hex[:8]}"
    foreign = f"codepilot-sweep-foreign-{uuid4().hex[:8]}"
    assert _docker(
        "run", "-d", "--name", orphan, "--label", CONTAINER_LABEL,
        DEFAULT_IMAGE, "sleep", "infinity",
    ).returncode == 0
    # 用户自己起的容器：没有 label，清扫必须看不见它
    assert _docker("run", "-d", "--name", foreign, DEFAULT_IMAGE, "sleep", "infinity").returncode == 0

    try:
        removed, error = sweep_orphan_containers()
        assert error is None
        assert removed >= 1
        assert not _exists(orphan), "带 label 的孤儿容器应被清扫"
        assert _exists(foreign), "不带 label 的容器不该被误删"
    finally:
        _docker("rm", "-f", orphan)
        _docker("rm", "-f", foreign)
