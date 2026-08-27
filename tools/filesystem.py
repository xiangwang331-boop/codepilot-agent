"""文件系统工具。

工具契约（对应 OpenHands 的 name + description + action_schema + executor）：
每个工具 = Pydantic args_schema（参数校验 + 生成 JSON Schema）+ 一个真正执行的函数。
执行函数只通过 WorkspaceManager 操作文件，绝不直接碰 os。
异常一律捕获并返回 "ERROR: ..." 字符串，让 LLM 能据此自我纠正。
"""
from __future__ import annotations

import re
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from workspace.manager import WorkspaceManager

# ---- 参数 schema（即 OpenAI function 的 JSON Schema）----


class ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="相对 workspace 根目录的路径，'.' 表示根目录")
    recursive: bool = Field(default=False, description="是否递归列出子目录")


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="要读取的文件路径（相对 workspace 根目录）")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="要写入的文件路径（相对 workspace 根目录）")
    content: str = Field(..., description="完整的文件内容")


class EditFileArgs(BaseModel):
    path: str = Field(..., description="要编辑的文件路径")
    old_string: str = Field(..., description="要替换的原文（精确匹配）")
    new_string: str = Field(..., description="替换后的新文本")
    replace_all: bool = Field(default=False, description="是否替换所有出现（默认只替换第一处）")


class DeleteFileArgs(BaseModel):
    path: str = Field(..., description="要删除的文件路径")


class SearchCodeArgs(BaseModel):
    query: str = Field(..., description="要搜索的字符串或正则")
    path: str = Field(default=".", description="在哪个目录下搜索")
    regex: bool = Field(default=False, description="query 是否为正则表达式")


# ---- 执行函数 ----

_MAX_LIST = 200
_MAX_SEARCH = 200


def _err(e: Exception) -> str:
    return f"ERROR: {type(e).__name__}: {e}"


def _list_files(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(path: str = ".", recursive: bool = False) -> str:
        try:
            entries = ws.list_dir(path, recursive)
            if not entries:
                return f"(空目录) {path}"
            if len(entries) > _MAX_LIST:
                return "\n".join(entries[:_MAX_LIST]) + f"\n...(共 {len(entries)} 项，已截断)"
            return "\n".join(entries)
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def _read_file(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(path: str) -> str:
        try:
            return ws.read_text(path)
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def _write_file(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(path: str, content: str) -> str:
        try:
            target = ws.write_text(path, content)
            return f"已写入 {ws.relpath(target)}（{len(content)} 字符）"
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def _edit_file(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        try:
            text = ws.read_text(path)
            count = text.count(old_string)
            if count == 0:
                return f"ERROR: 未找到要替换的原文 {old_string[:40]!r}，文件未修改"
            new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
            ws.write_text(path, new_text)
            n = count if replace_all else 1
            return f"已编辑 {path}：替换 {n} 处"
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def _delete_file(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(path: str) -> str:
        try:
            ws.delete(path)
            return f"已删除 {path}"
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def _search_code(ws: WorkspaceManager) -> Callable[..., str]:
    def fn(query: str, path: str = ".", regex: bool = False) -> str:
        try:
            target = ws.resolve(path)
            if not target.is_dir():
                return f"ERROR: 不是目录: {path}"
            matches: list[str] = []
            for f in sorted(target.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.startswith(".") or "__pycache__" in f.parts:
                    continue
                try:
                    lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for i, line in enumerate(lines, 1):
                    hit = re.search(query, line) if regex else (query in line)
                    if hit:
                        matches.append(f"{ws.relpath(f)}:{i}: {line.strip()[:120]}")
                        if len(matches) >= _MAX_SEARCH:
                            return "\n".join(matches) + "\n...(结果过多，已截断)"
            if not matches:
                return f"(未找到匹配 '{query}')"
            return "\n".join(matches)
        except Exception as e:  # noqa: BLE001
            return _err(e)

    return fn


def build_tools(ws: WorkspaceManager) -> list[StructuredTool]:
    """把一个 WorkspaceManager 绑定成一组文件系统工具。"""
    return [
        StructuredTool.from_function(
            name="list_files",
            description="列出 workspace 内的文件和目录。用于查看项目结构，确认当前有哪些文件。",
            args_schema=ListFilesArgs,
            func=_list_files(ws),
        ),
        StructuredTool.from_function(
            name="read_file",
            description="读取一个文件的完整内容。写代码或改代码前先看现状。",
            args_schema=ReadFileArgs,
            func=_read_file(ws),
        ),
        StructuredTool.from_function(
            name="write_file",
            description="创建或覆盖一个文件。content 为完整内容。用于新建代码文件。",
            args_schema=WriteFileArgs,
            func=_write_file(ws),
        ),
        StructuredTool.from_function(
            name="edit_file",
            description="对文件做精确的字符串替换（先匹配 old_string，替换成 new_string）。用于小范围修改已有代码。",
            args_schema=EditFileArgs,
            func=_edit_file(ws),
        ),
        StructuredTool.from_function(
            name="delete_file",
            description="删除一个文件（或空目录）。",
            args_schema=DeleteFileArgs,
            func=_delete_file(ws),
        ),
        StructuredTool.from_function(
            name="search_code",
            description="在 workspace 内按字符串或正则搜索代码，返回 文件:行号:内容。用于定位某个符号/函数/报错来源。",
            args_schema=SearchCodeArgs,
            func=_search_code(ws),
        ),
    ]
