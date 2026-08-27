"""Workspace 管理器。

所有文件操作都被限制在根目录内，防止 `../`、绝对路径、符号链接穿越到
workspace 之外。Agent 不允许直接碰 os/subprocess，只能经由这里的受控接口。
"""
from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """路径或操作超出 workspace 允许范围。"""


class WorkspaceManager:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def relpath(self, p: Path) -> str:
        """绝对路径 -> 相对 root 的 POSIX 风格字符串。"""
        return p.relative_to(self.root).as_posix()

    def resolve(self, path: str | Path) -> Path:
        """把 workspace 相对路径解析成受控的绝对路径。

        拒绝两类危险输入：
        1. 绝对路径（如 C:/...、/etc/...）
        2. 解析后仍越过 root 边界的路径（../ 穿越、符号链接指向外部）
        """
        p = Path(path)
        if p.is_absolute():
            raise WorkspaceError(f"禁止绝对路径: {path}")
        candidate = (self.root / p).resolve()
        # .resolve() 会展开 .. 与符号链接；只要结果仍在 root 内即合法
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"路径越过 workspace 边界: {path}")
        return candidate

    # ---- 目录 ----
    def list_dir(self, path: str = ".", recursive: bool = False) -> list[str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise WorkspaceError(f"不是目录: {path}")
        if recursive:
            return sorted(self.relpath(item) for item in target.rglob("*"))
        return sorted(self.relpath(item) for item in target.iterdir())

    # ---- 文件 ----
    def exists(self, path: str) -> bool:
        return self.resolve(path).exists()

    def is_dir(self, path: str) -> bool:
        return self.resolve(path).is_dir()

    def read_text(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"不是文件: {path}")
        return target.read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> Path:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def delete(self, path: str) -> None:
        target = self.resolve(path)
        if target.is_dir():
            target.rmdir()  # 只删空目录，避免误删整棵树
        elif target.is_file():
            target.unlink()
        else:
            raise WorkspaceError(f"路径不存在: {path}")
