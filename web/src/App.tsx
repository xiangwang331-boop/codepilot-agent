/**
 * 组装根。持有：选中会话（与 URL 同步）、事件流、会话列表、动作分发、主题。
 *
 * ## URL 深链为什么用 `?session=` 而不是路由
 *
 * 后端是 `StaticFiles(html=True)` 挂在 `/`（`api/app.py:113-114`）：未知路径直接
 * 404，没有 SPA 兜底。补一个 catch-all 就得改后端，而 P8 的硬约束是**后端一行不改**。
 * 查询参数天然被 `/` 接受，可刷新、可分享、零后端改动。
 * 用 `history.replaceState` 而不是 `pushState` —— 不制造历史条目，也就没有「后退」要处理。
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, createSession, deleteSession, sendApproval, sendMessage } from "./api/client";
import { ApprovalGate } from "./components/ApprovalGate";
import { AgentBadge, ConnBadge, StatusPill } from "./components/Badges";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { EventTimeline } from "./components/EventTimeline";
import { FilePanel } from "./components/FilePanel";
import { FilterBar } from "./components/FilterBar";
import { ResultPanel } from "./components/ResultPanel";
import { SessionMeta } from "./components/SessionMeta";
import { SessionRail } from "./components/SessionRail";
import { TokenPanel } from "./components/TokenPanel";
import { useSessionStream } from "./hooks/useSessionStream";
import { useSessions } from "./hooks/useSessions";
import { buildTimeline, collectAgents, DEFAULT_FILTERS, type FilterState } from "./model/events";
import { buildFileShadow } from "./model/files";
import { markReruns } from "./model/replay";
import { resultText } from "./model/session";
import { aggregateTokens } from "./model/tokens";

function readSessionParam(): string | null {
  if (typeof location === "undefined") return null;
  return new URLSearchParams(location.search).get("session");
}

function readTheme(): "dark" | "light" {
  try {
    return localStorage.getItem("cp-theme") === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export default function App(): React.JSX.Element {
  const [selected, setSelected] = useState<string | null>(readSessionParam);
  // 刚建出来的会话在轮询追上之前不在列表里 —— 用它顶一下，免得闪一下「会话不存在」。
  const [optimisticId, setOptimisticId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [theme, setTheme] = useState<"dark" | "light">(readTheme);

  const { envelopes, live, state } = useSessionStream(selected);
  const {
    sessions,
    error: listError,
    loaded,
    historyAvailable,
    refresh,
  } = useSessions(live);

  // 主题落到 <html data-theme>，CSS 只认这个属性（见 tokens.css）
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("cp-theme", theme);
    } catch {
      // 隐私模式下 localStorage 会抛 —— 忽略，主题退化成「不记忆」
    }
  }, [theme]);

  const chooseSession = useCallback((id: string): void => {
    setSelected(id);
    setOptimisticId(null);
    history.replaceState(null, "", `?session=${encodeURIComponent(id)}`);
  }, []);

  // 会话列表到位后自动选中：优先 URL 参数，其次列表第一个。
  // 只在「什么都没选」时兜底，不会把用户已选的会话顶掉。
  useEffect(() => {
    if (!loaded) return;
    if (selected === null) {
      const first = sessions[0];
      if (first) chooseSession(first.thread_id);
      return;
    }
    if (sessions.some((s) => s.thread_id === selected)) setOptimisticId(null);
  }, [loaded, sessions, selected, chooseSession]);

  const current = useMemo(
    () => sessions.find((s) => s.thread_id === selected) ?? null,
    [sessions, selected],
  );

  // 判重扫一遍就够 —— 折叠、token 聚合、文件重建三个消费者共用这一份。
  const timeline = useMemo(() => markReruns(envelopes), [envelopes]);
  const blocks = useMemo(() => buildTimeline(timeline), [timeline]);
  const agents = useMemo(() => collectAgents(blocks), [blocks]);
  const tokens = useMemo(() => aggregateTokens(timeline), [timeline]);
  const files = useMemo(() => buildFileShadow(timeline), [timeline]);

  const run = useCallback(
    async (fn: () => Promise<unknown>): Promise<void> => {
      setActionError(null);
      try {
        await fn();
        refresh();
      } catch (e) {
        setActionError(e instanceof ApiError ? e.message : String(e));
        // 409 忙：本地 status 是轮询来的（最旧 10 秒前），**服务端那句才是最新的** ——
        // 立刻拉一次，让「发送」按钮按真实状态禁用、批准卡按真实状态冒出来，
        // 而不是留一个点下去必然失败的按钮。`busyStatus` 只在对象 detail 上有值，
        // 所以 409 重名不会误触发（见 `ApiError.busyStatus`）。
        if (e instanceof ApiError && e.busyStatus !== null) refresh();
      }
    },
    [refresh],
  );

  const onCreate = useCallback((): void => {
    setCreating(true);
    void run(async () => {
      const info = await createSession();
      setOptimisticId(info.thread_id);
      chooseSession(info.thread_id);
    }).finally(() => setCreating(false));
  }, [run, chooseSession]);

  const onDelete = (id: string): void => {
    void run(async () => {
      await deleteSession(id);
      if (selected === id) {
        setSelected(null);
        setOptimisticId(null);
        history.replaceState(null, "", location.pathname);
      }
    });
  };

  const onSend = (task: string): void => {
    if (selected === null) return;
    void run(() => sendMessage(selected, task));
  };

  const onDecide = (approved: boolean): void => {
    if (selected === null) return;
    void run(() => sendApproval(selected, approved));
  };

  const clearParam = useCallback((): void => {
    setSelected(null);
    setOptimisticId(null);
    history.replaceState(null, "", location.pathname);
  }, []);

  // WS 的 4404 是「会话没了」的权威信号；轮询列表对不上只是它的兜底表现。
  const gone = state.kind === "fatal" && state.code === 4404;
  const missing = selected !== null && current === null && optimisticId !== selected;

  const status = current?.status ?? "idle";
  const approvals = current?.approval ?? [];

  const emptyProps = {
    onCreate,
    onPick: clearParam,
    hasSessions: sessions.length > 0,
    creating,
  };

  const mainArea = (): React.JSX.Element => {
    if (!loaded) return <EmptyState variant="loading" {...emptyProps} />;
    if (selected === null) return <EmptyState variant="no-session" {...emptyProps} />;
    if (gone || missing) {
      return (
        <EmptyState
          variant="missing-session"
          reason={state.kind === "fatal" ? state.reason : undefined}
          {...emptyProps}
        />
      );
    }
    return (
      <EventTimeline blocks={blocks} filters={filters} historyAvailable={historyAvailable} />
    );
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          CodePilot <small>multi-agent runtime</small>
        </div>
        {current !== null && <StatusPill status={status} />}
        {current !== null && <ConnBadge state={state} />}
        <span className="topbar__spacer" />
        {tokens.total.calls > 0 && (
          <span className="topbar__tokens" title={`${tokens.total.calls} 次 LLM 调用`}>
            输入 {tokens.total.prompt.toLocaleString()} / 输出{" "}
            {tokens.total.completion.toLocaleString()} = {tokens.total.total.toLocaleString()} tokens
          </span>
        )}
        <button
          className="iconbtn"
          title={theme === "dark" ? "切到浅色" : "切到深色"}
          onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? "☾" : "☀"}
        </button>
      </header>

      <SessionRail
        sessions={sessions}
        selectedId={selected}
        onSelect={chooseSession}
        onCreate={onCreate}
        onDelete={onDelete}
        creating={creating}
        historyAvailable={historyAvailable}
      />

      <main className="main">
        {current !== null && (
          <div className="main__head">
            <AgentBadge agent="Supervisor" />
            <FilterBar filters={filters} agents={agents} onChange={setFilters} />
          </div>
        )}

        {actionError !== null && (
          <div className="toast" style={{ margin: "8px 16px 0" }}>
            <span>{actionError}</span>
            <button onClick={() => setActionError(null)}>关闭</button>
          </div>
        )}

        {mainArea()}

        {current !== null && <Composer status={status} onSend={onSend} />}
      </main>

      <aside className="inspector">
        {approvals.length > 0 && (
          <div style={{ padding: "12px 12px 0" }}>
            <ApprovalGate approvals={approvals} busy={false} onDecide={onDecide} />
          </div>
        )}

        {current === null ? (
          <div className="panel__empty" style={{ padding: "12px" }}>
            {loaded ? "选择或新建一个会话。" : "正在读取会话列表…"}
          </div>
        ) : (
          <>
            <ResultPanel text={resultText(current.result)} />
            <FilePanel shadow={files} />
            <TokenPanel summary={tokens} />
            <SessionMeta session={current} />
          </>
        )}

        {listError !== null && (
          <div className="panel__empty" style={{ color: "var(--err)", padding: "12px" }}>
            会话列表拉取失败：{listError}
          </div>
        )}
      </aside>
    </div>
  );
}
