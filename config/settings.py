"""配置读取。

所有可配置项都来自环境变量，让 Agent Core 与具体模型供应商解耦：
- LLM_API_KEY  : 供应商 API Key
- LLM_BASE_URL : OpenAI 兼容端点（DeepSeek/OpenAI/其他），留空走 OpenAI 默认
- LLM_MODEL    : 模型名
- WORKSPACE_ROOT : workspace 根目录，默认 codepilot/workspace_data
- MAX_ITERATIONS: 最大迭代次数，防死循环
- CHECKPOINT_DB_PATH : checkpoint SQLite 库路径，默认 data/checkpoints.db
- CONTEXT_LIMIT : 模型上下文窗口 token 预算（P4-3-2 token 守卫）；None 关闭
- RESERVE_TOKENS : token 预算外 headroom（输出 + 下轮新消息 + 估算误差）；None 用 context_limit//5
- SANDBOX_MODE : run_command 执行宿主（P5）。local = 本机 subprocess（默认）；docker = Docker 沙箱
- SANDBOX_IMAGE : docker 模式镜像 tag；None 用 tools.command_runner.DEFAULT_IMAGE
- PERSISTENCE_BACKEND : 持久化后端（P6）。sqlite = SqliteSaver + 事件仅内存（默认，P0–P5 行为）；
  postgres = PostgresSaver + 事件落库，共用 DATABASE_URL
- DATABASE_URL : postgres 后端的连接串；sqlite 后端下不使用

启动时会加载项目根目录的 `.env`（不存在则退回 `.env.example`），
但已存在的环境变量优先，不会被覆盖。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int | None = None) -> int | None:
    """读环境变量为 int；未设置/空串返回 default（默认 None = 关闭该项）。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_sandbox_mode() -> str:
    """P5: SANDBOX_MODE → 'local' | 'docker'；非法值告警回退 local（不静默）。"""
    raw = (os.getenv("SANDBOX_MODE") or "local").strip().lower()
    if raw not in ("local", "docker"):
        print(
            f"警告: SANDBOX_MODE={raw!r} 非法，回退 local（可选: local | docker）",
            file=sys.stderr,
        )
        return "local"
    return raw


def _env_persistence_backend() -> str:
    """P6: PERSISTENCE_BACKEND → 'sqlite' | 'postgres'；非法值告警回退 sqlite（不静默）。"""
    raw = (os.getenv("PERSISTENCE_BACKEND") or "sqlite").strip().lower()
    if raw not in ("sqlite", "postgres"):
        print(
            f"警告: PERSISTENCE_BACKEND={raw!r} 非法，回退 sqlite（可选: sqlite | postgres）",
            file=sys.stderr,
        )
        return "sqlite"
    return raw


def _load_dotenv() -> None:
    """加载 .env（优先）或 .env.example，但不覆盖已有环境变量。"""
    project_root = Path(__file__).resolve().parent.parent
    for name in (".env", ".env.example"):
        path = project_root / name
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return  # 只加载第一个存在的文件


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    workspace_root: Path
    max_iterations: int = 20
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    # P4-3-2 token-aware context budget：None = 关闭 token 守卫（仅消息数触发，P4-3-1 行为）
    context_limit: int | None = None
    reserve_tokens: int | None = None
    # P5 run_command 执行宿主：'local'（本机 subprocess，默认）= 原行为；'docker' = 沙箱
    sandbox_mode: str = "local"
    # P5 docker 模式镜像 tag；None → tools.command_runner.DEFAULT_IMAGE
    sandbox_image: str | None = None
    # P6 持久化后端：'sqlite'（默认，checkpoint 落 data/checkpoints.db，事件仅内存）
    # 或 'postgres'（checkpoint + 事件都落 DATABASE_URL 指向的库）
    persistence_backend: str = "sqlite"
    # P6 postgres 模式连接串；sqlite 模式不使用
    database_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        project_root = Path(__file__).resolve().parent.parent
        workspace_root = Path(
            os.getenv("WORKSPACE_ROOT", str(project_root / "workspace_data"))
        ).resolve()
        checkpoint_db_path = Path(
            os.getenv("CHECKPOINT_DB_PATH", str(project_root / "data" / "checkpoints.db"))
        ).resolve()
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            workspace_root=workspace_root,
            max_iterations=int(os.getenv("MAX_ITERATIONS", "20")),
            checkpoint_db_path=checkpoint_db_path,
            context_limit=_env_int("CONTEXT_LIMIT"),
            reserve_tokens=_env_int("RESERVE_TOKENS"),
            sandbox_mode=_env_sandbox_mode(),
            sandbox_image=os.getenv("SANDBOX_IMAGE") or None,
            persistence_backend=_env_persistence_backend(),
            database_url=(os.getenv("DATABASE_URL") or "").strip(),
        )
