"""服务端日志配置：终端（按级别分流）+ 落文件 + 把 uvicorn 并进同一套格式。

## 为什么是「服务端专属」——`main.py` 一行不动

`main.py` 的 31 个 `print` 是**字节锚点**：它的 docstring 写着「所有 print 及其相对顺序刻意
保持不变」，`tests/test_cli_output.py` 里既有顺序断言、也有 36 行冻结基线
（`test_cli_output_matches_pre_p7_golden`，整段 stdout 精确比对）。所以本模块
**只由 `create_app` 调用**，`main.py` 永远不配置 logging。

## ⛔ 为什么 handler 必须挂在 `codepilot` 命名空间上，不能挂 root

这是本模块最容易做错、且错了会以**别的测试失败**的形式表现出来的一点：

1. **`logging.lastResort` 在 pytest 里是死的**。`_pytest` 的 logging 插件（默认内建）会在
   每个用例的 setup/call/teardown 往 **root** 上加两个 `LogCaptureHandler`
   （`_pytest/logging.py` 的 `catching_logs`）。于是 `Logger.callHandlers` 里
   `found == 2`，`if found == 0: lastResort.handle(record)` 这个分支**永远进不去** ——
   未配置时那条「warning 自动打 stderr」的后路在 pytest 下**不存在**。
   实测：`logger.warning("警告: ...")` 在用例里只出现在 pytest 的
   `--- Captured log call ---` 段，`capsys.readouterr().err` 是空串。
   （在 pytest **之外**它确实是逐字节等价于 `print(..., file=sys.stderr)` 的，见
   `tests/test_logging.py` 那条用例——所以 CLI 的真实行为不受影响。）
2. **挂在 root 上会污染 CLI 的字节锚点**。pytest 按文件名顺序收集，
   `test_api.py` 跑在 `test_cli_output.py` **之前**，于是轮到冻结基线那个用例时 root 已经被
   我们配过了。root 一旦是 INFO + 有 stdout handler，`httpx`（每次请求一条 INFO）、
   `langchain_core` 等库的记录就会落到 stdout 上，而那个用例是**整段精确比对**。
3. 反过来，自己的 handler 只挂在 `codepilot.*` 上，第三方噪音（httpx / langchain /
   langgraph / psycopg_pool）**根本流不进我们的 handler**，日志文件里不会被框架闲聊淹掉。
   同时 `propagate` **保持默认的 True** —— 记录继续上传到 root，pytest 的 caplog 才抓得到
   （`tests/test_postgres.py` 等靠 `caplog` 断言降级告警）。

## 流的分流沿用仓库既有约定（**uvicorn 的日志因此换了一边**）

服务侧输出里 31/36 是 stderr：`提示：`/裸 info → stdout，`警告:`/`错误：` → stderr。
这里用两个 handler + 级别过滤器**原样保留**这个分流，所以谁在重定向 stdout/stderr，
看到的服务侧输出与改动前一致。

⚠️ **一个刻意接受的例外：uvicorn 的 error/启动类日志会从 stderr 挪到 stdout。**
uvicorn 0.53 的 `LOGGING_CONFIG` 里 `default`（写 stderr）挂在**父 logger `uvicorn`** 上，
`access`（写 stdout）挂在 `uvicorn.access` 上。接进来之后本模块只剩一条规则（按级别分流），
于是它那些 **INFO** 级记录（`Uvicorn running on ...`、`Application startup complete.`）
走 stdout；ERROR（如 `Application startup failed`）仍走 stderr。要维持「uvicorn 一律
stderr」就得给它单独挂一对 handler，规则从一条变两条 —— 换来的只是「重定向的人少意外一次」。
真机起服务时两股都直接进同一个终端，看不出区别。**改这条之前先想清楚**：
`api/app.py` 的启动横幅本来就是 stdout，把 uvicorn 的启动行也放 stdout 反而让
「启动那一刻发生了什么」按时间顺序落在同一股流里。

⚠️ **uvicorn 的 handler 挂在父 logger 上，清的时候别只清子 logger。** 三个子 logger
（`uvicorn.error`/`asgi`/`access`）清空并上冒之后，记录会落到**父 logger**，而 uvicorn 自己
那个 `default` handler 恰恰就在那儿 —— 漏清的症状是**每条记录写两遍**：一遍我们的格式
（stdout）、一遍 uvicorn 的 `INFO:     ` 格式（stderr）。真机起服务实测踩过，见 `_adopt_uvicorn`。

## 其它三条实现约束

- **handler 的 stream 必须动态取**。`StreamHandler.__init__` 会把 `stream=None` 解析成
  **构造那一刻**的 `sys.stderr` 并存成实例属性、此后绑死；pytest 每个用例换一次流，
  绑死了就写进作废的缓冲。所以 `stream` 做成 property（照抄 `logging._StderrHandler`，
  它也因此只调 `Handler.__init__` 而**不**调 `StreamHandler.__init__`）。
- **幂等**。一个 pytest 进程里 `create_app` 会被真实实例化 ~59 次，且多个用例用同一个
  `tmp_path` 连起 2–4 次。重复 `addHandler` 会让同一条告警写 N 遍。
- **绝不抛异常**。沿用 `persistence/event_store.py` 的 `record()` 契约（关键坑 #23
  「可观测性不该杀死正在跑的任务」）与 `sweep_orphan_containers` 的「绝不挡启动」。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Callable, IO

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注，避免与 config.settings 循环 import
    from config.settings import Settings

#: 本项目的 logger 命名空间。**只往它上面挂 handler** —— 理由见模块 docstring。
LOG_NAMESPACE = "codepilot"

#: 给本模块装的 handler 打的标记。`setup_logging` 靠它认出「上次是我装的」并摘掉（幂等）。
_MARK = "_codepilot_owned"

#: 可按名解析的级别（顺序即是给用户看的可选列表）。
_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

#: 文件日志的轮转参数：单文件 2 MiB、留 5 份备份（最多约 12 MiB）。
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 5
LOG_FILENAME = "codepilot.log"

_CONSOLE_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
#: 文件里带上日期（终端看得出「刚刚」，翻文件时需要知道「哪天」）。
_FILE_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """取一个本项目命名空间下的 logger。

    各模块统一写 `logger = get_logger(__name__)`，于是 `config.settings` 会变成
    `codepilot.config.settings` —— 全部落在 `codepilot.*` 里，handler 才管得住。
    """
    if not name:
        return _namespace()
    return logging.getLogger(f"{LOG_NAMESPACE}.{name}")


def _namespace() -> logging.Logger:
    """命名空间的根（handler 挂在它上面）。

    ⛔ **必须是 `logging.getLogger("codepilot")`，不能拼成 `f"{LOG_NAMESPACE}.{name}"`
    再传空串** —— 那样得到的是 `"codepilot."`，名字末尾多一个点。它虽然是 `codepilot`
    的子 logger，却**不在** `codepilot.config.settings` 的祖先链上（链是按点切分的：
    settings → `codepilot.config` → `codepilot` → root）→ 挂上去的 handler **永远不会被调用**，
    表现为「日志一条都不出、也不报错」。
    """
    return logging.getLogger(LOG_NAMESPACE)


class _DynamicStreamHandler(logging.StreamHandler):
    """每次 emit 现取流的 handler —— `sys.stdout`/`sys.stderr` 被换掉也跟得上。

    关键在**不**调 `StreamHandler.__init__`：它会执行 `self.stream = sys.stderr`，
    而这里 `stream` 是只读 property（无 setter）→ `AttributeError`。
    `logging._StderrHandler` 同理只调 `Handler.__init__`。
    `StreamHandler.emit` 里是 `stream = self.stream` 局部取值，所以每次 emit 都会
    经过 property、拿到**当前**的流。
    """

    def __init__(self, resolve: Callable[[], IO[str]], level: int = logging.NOTSET) -> None:
        logging.Handler.__init__(self, level)
        self._resolve = resolve

    @property
    def stream(self) -> IO[str]:  # type: ignore[override]
        return self._resolve()

    def setStream(self, stream: IO[str]) -> IO[str] | None:
        """no-op —— 基类实现会 `self.stream = stream`，在只读 property 上直接 `AttributeError`。

        标准库与 pytest 都不会调它，但 `logging.config` 一类的路径会。
        """
        return None


class _SafeRotatingFileHandler(RotatingFileHandler):
    """轮转失败时跳过本轮，而不是从此每条记录都报错。

    Windows 上 `os.rename` 只要目标被**任何**别的句柄占着（上一次泄漏的 handler、
    `--workers` 的另一个进程、甚至编辑器打开着日志）就抛 `PermissionError [WinError 32]`。
    不接住的话 `Handler.handleError` 会在**此后每一条记录**上往 stderr 打一段
    `--- Logging error ---` + traceback，而且永远轮转不了。

    接住之后 `self.stream` 停在 None，`FileHandler.emit` 下次会自己重新 `_open()`
    （append 模式），所以日志照常写、只是这一轮不轮转 —— 正是想要的降级。
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError:
            pass


class _BelowWarning(logging.Filter):
    """info/debug → stdout（沿用「提示：走 stdout」的既有约定）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class _AtLeastWarning(logging.Filter):
    """warning 及以上 → stderr（沿用「警告:/错误：走 stderr」的既有约定）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


class _AccessErrorsOnly(logging.Filter):
    """uvicorn 的 access 记录只留 4xx/5xx。

    前端每 2–10 秒轮询一次 `GET /sessions`（有会话在跑时是 2 秒，见
    `web/src/model/session.ts` 的 `pollDelayMs`），全量 access 会把日志刷成健康检查
    流水账，真正的信息反而被淹掉。而 404（失效的 `?session=` 深链）与 500
    （DELETE 删库失败）恰恰是最该看见的。

    状态码在 `record.args` 第 5 位（uvicorn 的 access 模板是
    `'%s - "%s %s HTTP/%s" %d'`）。**形状对不上就放行** —— 宁可多记不可漏记。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        status = args[4]
        if not isinstance(status, int):
            return True
        return status >= 400


def setup_logging(settings: "Settings") -> None:
    """装日志。**绝不抛异常** —— 日志配不起来也不能挡住服务启动。

    幂等：重复调用只会摘掉自己上次装的再重装，不会叠加 handler。
    """
    try:
        _setup(settings)
    except Exception as e:  # noqa: BLE001  配置失败退回「未配置」状态，服务照跑
        print(f"警告: 日志系统初始化失败，本次退回默认输出: {e}", file=sys.stderr)


def teardown_logging() -> None:
    """摘掉并关闭**本模块装的** handler，复原本模块改过的开关。幂等。

    由 `create_app` 的 lifespan 在 `finally` 里调。

    **必须 close 而不只是摘掉**：Windows 上文件句柄不释放，`tmp_path` 拆除会报
    `PermissionError`，`RotatingFileHandler` 轮转时的 rename 也会被自己占住。

    两个 logger 都要收 —— handler 同时挂在命名空间和 `uvicorn` 上，是**同一批对象**
    （`_release` 先全摘完再按 `id()` 去重 close）。

    `uvicorn` 那几个 logger 的 `propagate` 与过滤器也要复原：`_adopt_uvicorn` 把它们
    改成「上冒到父 logger」并加了 access 过滤器，不还原的话，一个 app 起来又停掉之后
    这些改动会**留在进程里**（下一个 app 实例、或别的测试都受影响）。

    刻意**不**试图恢复 uvicorn 原先的 handler：那几个是被 setup `clear()` 掉的，恢复
    不了。生产里 uvicorn 只在进程启动时配一次日志、进程退出即结束，不需要恢复；
    这也是为什么 `_setup` 必须是「对 uvicorn 日志的最后一句话」。
    """
    ns = _namespace()
    parent = _uvicorn_parent()
    _release(ns, parent)
    parent.propagate = False  # uvicorn 的 dictConfig 就是这个值
    for name in ("uvicorn.error", "uvicorn.asgi", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.propagate = False  # 回到 uvicorn dictConfig 的形状（不再上冒）
        lg.filters = [f for f in lg.filters if not isinstance(f, _AccessErrorsOnly)]
    # 级别复位成 NOTSET —— 那正是「未配置」时的语义（继承 root 的 WARNING）。
    # 不复位的话，setup 之后残留的 INFO 会让 `lastResort` 的告警行为与 setup 之前不一致。
    ns.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------- 内部


def _setup(settings: "Settings") -> None:
    _reconfigure_streams()  # 最先做：下面几条告警本身也可能带中文
    level = _resolve_level(getattr(settings, "log_level", "INFO"))
    ns = _namespace()
    # 幂等：先把自己上次装的从**两个** logger 上摘干净（uvicorn 那边挂的是同一批对象）
    _release(ns, _uvicorn_parent())
    ns.setLevel(level)
    # propagate 保持默认 True：记录继续上传到 root，pytest 的 caplog 才抓得到。

    fmt = logging.Formatter(_CONSOLE_FMT, _CONSOLE_DATEFMT)
    stdout_handler = _DynamicStreamHandler(lambda: sys.stdout, level=level)
    stderr_handler = _DynamicStreamHandler(lambda: sys.stderr, level=level)
    stdout_handler.addFilter(_BelowWarning())
    stderr_handler.addFilter(_AtLeastWarning())
    handlers: list[logging.Handler] = [stdout_handler, stderr_handler]
    for handler in handlers:
        handler.setFormatter(fmt)

    log_dir = getattr(settings, "log_dir", None)
    if log_dir is not None:
        file_handler = _build_file_handler(Path(log_dir), level)
        if file_handler is not None:
            handlers.append(file_handler)

    for handler in handlers:
        _attach(ns, handler)

    # 文件 handler 也一并交给 uvicorn：**关掉终端之后，uvicorn 自己的启动行/错误/非 2xx
    # access 才是「当时到底发生了什么」的关键证据**，只落 codepilot.* 的话它们全丢。
    _adopt_uvicorn(handlers)


def _attach(logger: logging.Logger, handler: logging.Handler) -> None:
    setattr(handler, _MARK, True)
    logger.addHandler(handler)


def _uvicorn_parent() -> logging.Logger:
    """uvicorn 的父 logger —— 它自己的 `default` handler 就挂在这里（见 `_adopt_uvicorn`）。"""
    return logging.getLogger("uvicorn")


def _release(*loggers: logging.Logger) -> None:
    """把本模块装的 handler 从**所有这些** logger 上摘掉，再逐个 close。

    必须「先全摘完、再统一 close」，不能边摘边 close：同一批对象同时挂在 `codepilot` 与
    `uvicorn` 两个 logger 上，边摘边 close 会让后一个 logger 短暂持有一个**已 close 的文件
    handler** —— 那期间只要有一条记录经过它，`FileHandler.emit` 就会自己 `_open()` 把文件
    **重新打开**，收尾就白做了（Windows 上句柄就是这么漏的，`tmp_path` 拆除随即报
    `PermissionError`）。按 `id()` 去重，同一个 handler 只 close 一次。
    """
    owned: dict[int, logging.Handler] = {}
    for logger in loggers:
        for handler in [h for h in logger.handlers if getattr(h, _MARK, False)]:
            logger.removeHandler(handler)
            owned[id(handler)] = handler
    for handler in owned.values():
        try:
            handler.close()
        except Exception:  # noqa: BLE001  收尾路径上的失败不该再冒泡
            pass


def _build_file_handler(log_dir: Path, level: int) -> logging.Handler | None:
    """造文件 handler。建目录或开文件失败**只降级到控制台**（返回 None），不挡启动。

    `mkdir` 必须自己做 —— `FileHandler` **不会**建父目录，缺目录时它抛
    `FileNotFoundError`，被兜住之后就变成「文件日志静默不工作」，那正是要避免的失败模式。
    """
    target = log_dir / LOG_FILENAME
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = _SafeRotatingFileHandler(
            target,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",  # 不写就是 locale 编码 → 这台机器上是 GBK，中文第一条就炸
            errors="backslashreplace",  # 路径里的落单代理项等怪字符不该杀死整条记录
            delay=True,  # 不写日志就不建空文件、不占句柄
        )
    except Exception as e:  # noqa: BLE001
        print(f"警告: 日志文件 {target} 打不开，本次只输出到控制台: {e}", file=sys.stderr)
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FMT, _FILE_DATEFMT))
    return handler


def _adopt_uvicorn(handlers: list[logging.Handler]) -> None:
    """把 uvicorn 自己的 logger 并进同一套格式（我们的 handler 也挂到它上面）。

    时序已核实：uvicorn 在 `Config.__init__` 里就调了 `configure_logging()`
    （`.venv/.../uvicorn/config.py`），而 app 模块要到 `load()` 里的 `load_app()` 才被
    import —— uvicorn 配日志**早于**我们这一步，所以「清掉它的 handler」一定赢。
    （lifespan 本身在 `uvicorn/server.py` 的 `startup()` 里更晚才进。）

    ## ⛔ 必须清**父** logger 上的 handler，只清子 logger 是不够的

    uvicorn 0.53 的 `LOGGING_CONFIG` 是：

    ```
    "uvicorn":       {"handlers": ["default"], "level": "INFO", "propagate": false}
    "uvicorn.error": {"level": "INFO"}          # 无 handler，靠上冒
    "uvicorn.access":{"handlers": ["access"], "level": "INFO", "propagate": false}
    ```

    `default`（写 stderr）挂在**父** logger 上。我们给三个子 logger 清空 handler 并
    `propagate=True` 之后，记录正好落到父 logger —— 那里同时有 uvicorn 的 `default`
    和我们的 handler，于是**每条记录写两遍**：一遍我们的格式（stdout）、一遍
    `INFO:     ` 格式（stderr）。真机起服务实测踩到（漏清 `parent.handlers`）。

    做法：父 logger 上清掉所有**非本模块装的** handler，再挂上我们这**一批**（含文件
    handler —— uvicorn 的启动行/错误/非 2xx access 也该落文件，那正是「关掉终端之后
    还能回头看」的关键证据），然后 `propagate=False` 不再往 root 冒。

    子 logger 一律清空 handler + `propagate=True` —— 它们自己留 handler 的话，会既走
    自己又走父级，同样写两遍。

    **刻意不 `setLevel(NOTSET)`**：uvicorn 的 `--log-level` 是在 `dictConfig` 之后
    对子 logger 调 `setLevel` 生效的，抹掉它会让 `--log-level warning` 反而打出更多
    （级别越低越啰嗦）。保留原级别，让 uvicorn 自己的开关仍然有效；真正决定
    「控制台上出现什么」的是我们 handler 上的 `LOG_LEVEL`。
    """
    parent = _uvicorn_parent()
    for handler in [h for h in parent.handlers if not getattr(h, _MARK, False)]:
        parent.removeHandler(handler)  # uvicorn 的 `default`（stderr）
    for handler in handlers:
        _attach(parent, handler)
    parent.propagate = False

    for name in ("uvicorn.error", "uvicorn.asgi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True  # 冒到 `uvicorn` 上，那里有我们的 handler

    access = logging.getLogger("uvicorn.access")
    if not access.handlers and not access.propagate:
        # `--no-access-log` 的签名（这一版 uvicorn 是 `handlers = []` + `propagate = False`，
        # 已核对 `Config.configure_logging` 源码）：别人明确关了，我们不能借折叠之名
        # 把它打开 —— 清掉 handler 后上冒，等于把 access 又打开了。
        return
    access.handlers.clear()
    access.propagate = True
    access.filters = [f for f in access.filters if not isinstance(f, _AccessErrorsOnly)]
    access.addFilter(_AccessErrorsOnly())


def _reconfigure_streams() -> None:
    """把两个控制台流调成 UTF-8（照 `main.py:26-28` 的做法）。

    不加这一步，GBK 控制台上一条中文日志会在 logging 内部抛 `UnicodeEncodeError`,
    而 logging 把它吞成一句「--- Logging error ---」—— 比乱码难查得多。
    pytest 的捕获对象没有 `reconfigure`，所以整段包 try/except。
    （对 pytest 无实际作用：它每个用例都会换掉这两个流。这是给真机用的。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001  非 TextIOWrapper / 已被替换：忽略
            pass


def _resolve_level(raw: str) -> int:
    """解析级别名；非法值**告警回退 INFO**（沿用 `_env_sandbox_mode` 的不静默做法）。"""
    name = (raw or "INFO").strip().upper()
    if name not in _LEVELS:
        print(
            f"警告: LOG_LEVEL={raw!r} 非法，回退 INFO（可选: {' | '.join(_LEVELS)}）",
            file=sys.stderr,
        )
        return logging.INFO
    return _LEVELS[name]
