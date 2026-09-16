"""P9 增量：服务端日志（`config/logging_setup.py`）。

## 这个文件在守什么

日志是**横切**设施：写错了不会让某个功能挂掉，而是让**别的**测试以看不懂的方式失败。
所以这里的用例几乎都在守「副作用边界」，而不是「格式好不好看」：

- **幂等**：`tests/` 里 `create_app` 会被真实实例化 ~59 次。每次 `addHandler` 而不清理，
  同一条告警就会被写 N 遍 —— 直接撞死 `tests/test_postgres.py` 那句
  `assert err.count(...) == 1`。用例 `test_setup_is_idempotent_*` 就是那个钉的上游。
- **流是动态取的**：`capsys` 每个用例换一次 `sys.stderr`，handler 若在构造时把流
  绑死，写进的就是作废的缓冲、`capsys` 什么都读不到。
- **handler 挂在 `codepilot.*` 而不是 root**：挂 root 会让 `httpx`（每次请求一条 INFO）
  之类第三方记录涌进日志文件，而且会污染 `tests/test_cli_output.py` 的
  **36 行冻结基线**（pytest 按文件名收集，`test_api.py` 跑在它前面）。
- **收尾要真的关掉文件**：Windows 上句柄不 close，`tmp_path` 拆除与轮转 rename 都会
  撞 `PermissionError`。

## 刻意不写「未配置时字节与 print 相同」那条用例

`logging.lastResort` 在 pytest 里**不可达**（`_pytest.logging` 会往 root 挂两个
`LogCaptureHandler`，`callHandlers` 的 `found` 永远 > 0），所以那条性质**没法在 pytest
里断言**。它是真的（在裸进程里实测逐字节相同），但把它写成用例只会写出一条假绿。
详情见 `config/logging_setup.py` 的模块 docstring 与 CLAUDE.md 关键坑 #65。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from config import logging_setup
from config.logging_setup import (
    LOG_FILENAME,
    _AccessErrorsOnly,
    get_logger,
    setup_logging,
    teardown_logging,
)
from config.settings import Settings


def _settings(tmp_path, **kw) -> Settings:
    """直接构造 `Settings`（仓库惯例）—— 于是 `log_dir` 缺省是 `None`。"""
    base: dict = dict(
        llm_api_key="fake-key",
        llm_base_url="",
        llm_model="fake",
        workspace_root=Path(tmp_path) / "ws",
    )
    base.update(kw)
    return Settings(**base)


def _ns() -> logging.Logger:
    """handler 的落点：`codepilot` 命名空间的根。"""
    return logging.getLogger(logging_setup.LOG_NAMESPACE)


def _owned(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, logging_setup._MARK, False)]


@pytest.fixture(autouse=True)
def _clean_logging():
    """每个用例前后都把日志恢复成「未配置」。

    **必须的**：`setup_logging` 改的是**进程级全局状态**，泄漏到下一个用例就会让
    「这里为什么多一行 / 少一行」变成随机现象 —— 而且受害者往往是**别的文件**的测试。
    `teardown_logging` 幂等，重复调无害。

    级别也一并存还：用例里会故意 `setLevel(INFO)` 模拟 uvicorn 的 dictConfig
    （见 `test_access_filter_drops_2xx_end_to_end`），那是测试自己改的，得自己还回去。
    """
    watched = [
        logging.getLogger(name)
        for name in (
            logging_setup.LOG_NAMESPACE,
            "uvicorn",
            "uvicorn.error",
            "uvicorn.asgi",
            "uvicorn.access",
        )
    ]
    saved = {lg.name: lg.level for lg in watched}
    teardown_logging()
    yield
    teardown_logging()
    for lg in watched:
        if lg.level != saved[lg.name]:
            lg.setLevel(saved[lg.name])


# ---------------------------------------------------------------- 幂等


def test_setup_is_idempotent_handlers_do_not_accumulate(tmp_path):
    """连装 3 次，带标记的 handler **个数不涨**。"""
    settings = _settings(tmp_path, log_dir=Path(tmp_path) / "logs", log_level="INFO")
    setup_logging(settings)
    first = list(_owned(_ns()))
    assert first, "装完应该至少有一个带标记的 handler"

    setup_logging(settings)
    setup_logging(settings)
    assert len(_owned(_ns())) == len(first)


def test_setup_is_idempotent_one_record_is_written_once(tmp_path, capsys):
    """连装 3 次后，一条 warning 在 stderr 上**只出现一次**。

    这条是 `tests/test_postgres.py::test_record_warns_once_then_goes_quiet` 那句
    `assert err.count("事件持久化失败") == 1` 的上游保险：那里数的是 stderr 上的
    字节，这里数的是命中次数。
    """
    settings = _settings(tmp_path, log_dir=Path(tmp_path) / "logs", log_level="INFO")
    setup_logging(settings)
    setup_logging(settings)
    setup_logging(settings)

    get_logger("tests.probe").warning("警告: 只该出现一次")
    err = capsys.readouterr().err
    assert err.count("只该出现一次") == 1


# ---------------------------------------------------------------- 级别分流


def test_info_goes_to_stdout_and_warning_goes_to_stderr(tmp_path, capsys):
    """沿用仓库既有约定：info → stdout，warning 及以上 → stderr。

    服务侧的 36 处输出里 31 处走 stderr，重定向 stdout/stderr 的人不该看到输出换边。
    """
    setup_logging(_settings(tmp_path, log_level="INFO"))
    log = get_logger("tests.probe")
    log.info("普通信息")
    log.warning("告警信息")
    log.error("错误信息")

    out = capsys.readouterr()
    assert "普通信息" in out.out
    assert "普通信息" not in out.err
    assert "告警信息" in out.err
    assert "告警信息" not in out.out
    assert "错误信息" in out.err, "error 也走 stderr"


def test_level_below_threshold_is_dropped(tmp_path, capsys):
    """LOG_LEVEL=WARNING → info 一条都不出。"""
    setup_logging(_settings(tmp_path, log_level="WARNING"))
    log = get_logger("tests.probe")
    log.info("这条不该出现")
    log.warning("这条该出现")

    out = capsys.readouterr()
    assert "这条不该出现" not in out.out + out.err
    assert "这条该出现" in out.err


def test_invalid_level_falls_back_to_info_with_warning(tmp_path, capsys):
    """非法 LOG_LEVEL 不静默：告警 + 回退 INFO（沿用 `_env_sandbox_mode` 的做法）。"""
    setup_logging(_settings(tmp_path, log_level="banana"))
    err = capsys.readouterr().err
    assert "LOG_LEVEL" in err and "banana" in err
    assert _ns().level == logging.INFO


# ---------------------------------------------------------------- access 过滤


_UNSET = object()


def _access_record(status: int, args=_UNSET) -> logging.LogRecord:
    """照 uvicorn 的 access 形状造一条记录（模板 `'%s - "%s %s HTTP/%s" %d'`）。

    `args` 用哨兵而不是 `None` 当缺省：`None` 本身就是「形状不对」的一个用例，
    用 `None` 当缺省会让那个用例悄悄退化成「正常 5 元组」而假绿。
    """
    if args is _UNSET:
        args = ("127.0.0.1:1", "GET", "/sessions", "1.1", status)
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=args,
        exc_info=None,
    )


def test_access_filter_drops_2xx_keeps_4xx_5xx():
    """2xx 丢（前端每 2 秒轮询 GET /sessions，全量会把日志刷成健康检查流水账）；
    404/500 留（失效深链与删库失败是最该看见的两条）。"""
    f = _AccessErrorsOnly()
    assert f.filter(_access_record(200)) is False
    assert f.filter(_access_record(304)) is False
    assert f.filter(_access_record(404)) is True
    assert f.filter(_access_record(500)) is True
    assert f.filter(_access_record(499)) is True


def test_access_filter_passes_through_unexpected_shapes():
    """形状认不出来就**放行** —— 宁可多记不可漏记。"""
    f = _AccessErrorsOnly()
    assert f.filter(_access_record(200, args=("就当没这回事",))) is True
    assert f.filter(_access_record(200, args=None)) is True
    assert f.filter(
        _access_record(200, args=("a", "GET", "/", "1.1", "200"))
    ) is True, "状态码不是 int（uvicorn 传过字符串形状）→ 放行"


def test_access_filter_drops_2xx_end_to_end(tmp_path, capsys):
    """走完整链路：装完之后往 `uvicorn.access` 发一条 200 与一条 404，只有 404 留下。

    ⚠️ **两个容易看错的地方，都实测踩过**：

    1. 这里**必须先 `setLevel(INFO)`**。access 记录能不能被创建取决于该 logger 的
       生效级别，而那是 uvicorn 的 `dictConfig` 设的（真 uvicorn 下 `--log-level info`
       就是 INFO）。测试里没有 uvicorn，`uvicorn.access` 会继承 root 的 WARNING →
       `access.info(...)` **连记录都不产生** → 用例以「404 也没出现」的形式红，
       看着像过滤器坏了、其实是级别没开。`_adopt_uvicorn` 刻意不碰这个级别
       （抹成 NOTSET 会让 `--log-level warning` 反而打出更多），所以这里手工模拟。
    2. 留下来的那条走的是 **stdout**，不是 stderr。access 记录是 INFO 级，
       而本模块的分流规则是「info → stdout、warning 及以上 → stderr」——
       于是 uvicorn 自己的 access/error 日志**从 stderr 挪到了 stdout**
       （uvicorn 默认把两类都写 stderr）。这是刻意接受的一致规则，见模块 docstring。
    """
    setup_logging(_settings(tmp_path, log_level="INFO"))
    access = logging.getLogger("uvicorn.access")
    access.setLevel(logging.INFO)  # 模拟 uvicorn dictConfig 设的级别
    access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/sessions", "1.1", 200)
    access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/sessions/nope", "1.1", 404)

    out = capsys.readouterr()
    everything = out.out + out.err
    assert "/sessions/nope" in out.out, "404 必须留下（INFO 级 → stdout）"
    assert "GET /sessions HTTP" not in everything, "200 一条都不该留下"


def test_uvicorn_loggers_are_adopted_without_forcing_level(tmp_path):
    """handler 挂到 `uvicorn` 上、子 logger 往上冒；**不抹掉 uvicorn 自己的级别**。

    `setLevel(NOTSET)` 会让 `--log-level warning` 反而打出更多（级别越低越啰嗦），
    所以这里显式钉住「我们没动它」。
    """
    setup_logging(_settings(tmp_path, log_level="INFO"))
    parent = logging.getLogger("uvicorn")
    assert _owned(parent), "handler 应该挂在 uvicorn 父 logger 上"
    assert parent.propagate is False, "别再往 root 冒，否则 root 有 handler 时会写两遍"

    for name in ("uvicorn.error", "uvicorn.access"):
        child = logging.getLogger(name)
        assert child.handlers == [], f"{name} 自己的 handler 该被清掉（改走上冒）"
        assert child.propagate is True
        assert child.level == logging.NOTSET or child.level == logging.INFO, (
            f"{name} 的级别不该被我们改成别的值"
        )


def test_uvicorn_own_parent_handler_is_replaced_not_kept(tmp_path, capsys):
    """uvicorn 自己的 `default` handler 挂在**父** logger 上，必须被换掉。

    ⛔ 这条是**真机起服务实测踩出来的**（漏清 `parent.handlers`）：uvicorn 0.53 的
    `LOGGING_CONFIG` 是 `{"uvicorn": {"handlers": ["default"]}}` —— 写 stderr 的
    `default` 挂在**父** logger 上，而 `uvicorn.error` / `uvicorn.asgi` 自己无 handler、
    靠上冒。只清子 logger 的话，记录正好落到父 logger，那里同时有 uvicorn 的 `default`
    和我们的 handler → **每条记录写两遍**（我们的格式走 stdout、`INFO:     ` 格式走 stderr）。

    这里手工复刻「dictConfig 刚跑完那一刻」的形状（真的去调 `dictConfig` 会
    `_clearExistingHandlers()` 关掉全进程的 handler，包括 pytest 的捕获 handler —— 不值当）。
    """
    parent = logging.getLogger("uvicorn")
    theirs = logging.StreamHandler(sys.stderr)
    theirs.setFormatter(logging.Formatter("INFO:     %(message)s"))
    parent.addHandler(theirs)
    parent.propagate = False
    try:
        setup_logging(_settings(tmp_path, log_level="INFO"))
        assert theirs not in parent.handlers, "uvicorn 自己的 handler 该被摘掉"

        error_logger = logging.getLogger("uvicorn.error")
        error_logger.setLevel(logging.INFO)  # 级别门在「记录能不能被创建」上，同下条用例
        error_logger.info("启动完成")
        out = capsys.readouterr()
        assert out.out.count("启动完成") == 1, "我们的 handler 该写且只写一次"
        assert "启动完成" not in out.err, "旧格式那一路还在的话这里就会出现第二遍"
    finally:
        parent.removeHandler(theirs)


def test_access_filter_is_not_duplicated_across_setups(tmp_path):
    """反复装不会往 `uvicorn.access` 上堆叠同一个过滤器（堆了就是对每条记录多判几次）。"""
    settings = _settings(tmp_path, log_level="INFO")
    for _ in range(3):
        setup_logging(settings)
    filters = [f for f in logging.getLogger("uvicorn.access").filters
               if isinstance(f, _AccessErrorsOnly)]
    assert len(filters) == 1


# ---------------------------------------------------------------- 文件日志


def test_log_dir_none_writes_no_file(tmp_path):
    """`log_dir=None`（测试直接构造 `Settings` 时的缺省）= 完全不落文件。

    这个缺省是刻意的：~59 个 app 实例若都往仓库写文件，Windows 上的句柄泄漏会
    让 `tmp_path` 拆除报 `PermissionError`。
    """
    settings = _settings(tmp_path)  # 不传 log_dir
    assert settings.log_dir is None
    setup_logging(settings)
    get_logger("tests.probe").warning("只该在控制台")
    assert not list(Path(tmp_path).rglob(LOG_FILENAME))


def test_file_receives_records_in_utf8(tmp_path):
    """落文件的记录带中文也不炸，且日期格式带上「哪天」。"""
    log_dir = Path(tmp_path) / "logs"
    setup_logging(_settings(tmp_path, log_dir=log_dir, log_level="INFO"))
    get_logger("tests.probe").warning("警告: 中文也要能落盘")

    target = log_dir / LOG_FILENAME
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "中文也要能落盘" in text
    assert "WARNING" in text and "codepilot.tests.probe" in text


def test_file_dir_is_created_recursively(tmp_path):
    """`FileHandler` **不会**建父目录 —— 缺目录时必须我们自己 mkdir，否则文件日志静默失效。"""
    log_dir = Path(tmp_path) / "a" / "b" / "logs"
    setup_logging(_settings(tmp_path, log_dir=log_dir, log_level="INFO"))
    get_logger("tests.probe").warning("深目录")
    assert (log_dir / LOG_FILENAME).is_file()


def test_uvicorn_records_reach_the_file_too(tmp_path):
    """uvicorn 自己的记录也要落文件 —— 否则「关掉终端还能回头看」只覆盖一半。

    真机实测时正是这条先露的：文件里有启动横幅、**却没有** `Uvicorn running on ...`，
    也没有那条 404 的 access 行（文件 handler 当初只挂在 `codepilot` 上，而 uvicorn 的
    记录是挂在 `uvicorn` 那一支上的）。uvicorn 的启动行与 4xx/5xx access 恰恰是
    「当时到底发生了什么」最直接的两条证据。
    """
    log_dir = Path(tmp_path) / "logs"
    setup_logging(_settings(tmp_path, log_dir=log_dir, log_level="INFO"))

    error_logger = logging.getLogger("uvicorn.error")
    error_logger.setLevel(logging.INFO)  # 级别门在「记录能不能被创建」上
    error_logger.info("Application startup complete.")

    access = logging.getLogger("uvicorn.access")
    access.setLevel(logging.INFO)
    access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/sessions/nope", "1.1", 404)
    access.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/sessions", "1.1", 200)

    text = (log_dir / LOG_FILENAME).read_text(encoding="utf-8")
    assert "Application startup complete." in text
    assert "/sessions/nope" in text, "非 2xx access 要落文件（失效深链是最该查的两条之一）"
    assert "GET /sessions HTTP" not in text, "2xx 的过滤在文件这一路同样生效"


def test_log_dir_pointing_at_a_file_never_raises_and_keeps_console(tmp_path, capsys):
    """`log_dir` 是个**文件**（不是目录）→ 只降级到控制台，绝不抛。

    日志配不起来不能挡服务启动（沿用 `sweep_orphan_containers` 的「绝不挡启动」）。
    """
    blocker = Path(tmp_path) / "not-a-dir"
    blocker.write_text("我是文件", encoding="utf-8")

    setup_logging(_settings(tmp_path, log_dir=blocker, log_level="INFO"))  # 不该抛

    get_logger("tests.probe").warning("控制台仍然工作")
    out = capsys.readouterr()
    assert "控制台仍然工作" in out.err
    assert "打不开" in out.err, "降级本身要有声"


def test_teardown_releases_the_file_handle(tmp_path):
    """收尾之后文件句柄必须还回去 —— Windows 上不 close，rename 会 `PermissionError`。

    用「能不能改名」当句柄是否释放的探针（比 `psutil` 之类轻，且直接对应真实症状：
    `RotatingFileHandler` 轮转就是靠 rename）。
    """
    log_dir = Path(tmp_path) / "logs"
    settings = _settings(tmp_path, log_dir=log_dir, log_level="INFO")
    setup_logging(settings)
    get_logger("tests.probe").warning("先写一条，把文件真的打开")
    target = log_dir / LOG_FILENAME
    assert target.is_file()

    teardown_logging()

    moved = log_dir / "moved.log"
    target.rename(moved)  # 句柄没释放时这一步在 Windows 上抛 PermissionError
    assert moved.is_file()


def test_teardown_is_idempotent_and_removes_handlers(tmp_path, capsys):
    """收尾幂等，且把 handler 摘干净、级别复位（复位 = 回到「未配置」语义）。"""
    settings = _settings(tmp_path, log_dir=Path(tmp_path) / "logs", log_level="INFO")
    setup_logging(settings)
    assert _owned(_ns())

    teardown_logging()
    teardown_logging()
    assert _owned(_ns()) == []
    assert _owned(logging.getLogger("uvicorn")) == []
    assert _ns().level == logging.NOTSET

    get_logger("tests.probe").info("收尾后 info 不该进控制台")
    assert "收尾后 info 不该进控制台" not in capsys.readouterr().out


# ---------------------------------------------------------------- 命名空间


def test_handlers_live_on_codepilot_namespace_not_root(tmp_path):
    """**这条是整套设计的核心约束**：handler 只能挂 `codepilot.*`，绝不能挂 root。

    挂 root 的两个后果：① `httpx`（每次请求一条 INFO）之类第三方记录涌进日志文件；
    ② `tests/test_cli_output.py` 的 36 行冻结基线会被第三方 stdout 记录污染
    （pytest 按文件名收集，`test_api.py` 先跑）。
    """
    root = logging.getLogger()
    before = [h for h in root.handlers if getattr(h, logging_setup._MARK, False)]
    assert before == [], "前提：root 上不该有我们装的 handler"

    setup_logging(_settings(tmp_path, log_dir=Path(tmp_path) / "logs"))
    after = [h for h in root.handlers if getattr(h, logging_setup._MARK, False)]
    assert after == [], "装完之后 root 上仍然不能有我们装的 handler"
    assert _owned(_ns()), "handler 应该在 codepilot 命名空间上"


def test_third_party_records_do_not_reach_our_handlers(tmp_path, capsys):
    """第三方 logger 的记录流不进我们的 handler（它们的祖先链上没有 `codepilot`）。"""
    setup_logging(_settings(tmp_path, log_level="INFO"))
    logging.getLogger("httpx").info("HTTP Request: GET /sessions 200 OK")
    logging.getLogger("langchain_core.tracers").info("第三方闲聊")

    out = capsys.readouterr()
    assert "HTTP Request" not in out.out + out.err
    assert "第三方闲聊" not in out.out + out.err


def test_get_logger_prefixes_the_namespace():
    """各模块统一 `get_logger(__name__)`，于是全部落在 `codepilot.*` 里。"""
    assert get_logger("api.app").name == "codepilot.api.app"
    # 空串要**正好**是命名空间的根：拼成 `codepilot.` 就多一个点、成了一个不在
    # `codepilot.config.settings` 祖先链上的**子** logger，挂上去的 handler 永不触发。
    assert get_logger("").name == logging_setup.LOG_NAMESPACE


def test_setup_never_raises_even_with_a_hostile_settings_object():
    """`setup_logging` 对「不像 Settings 的东西」也必须不抛（它只 getattr 两个字段）。"""

    class Weird:
        log_level = "INFO"
        log_dir = None

    setup_logging(Weird())  # 不该抛
    assert _owned(_ns())
