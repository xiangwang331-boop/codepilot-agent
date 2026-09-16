/**
 * 侧栏会话列表：自适应轮询 + 页面可见性联动。
 *
 * WS 只订阅**选中的那一个**会话，侧栏的其它会话状态只能靠 REST 轮询
 * （后端没有「订阅所有会话」的口子，P8 又不改后端）。所以轮询间隔是自适应的：
 * 有忙会话 → 2s，全闲 → 10s，页面不可见 → 停。
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, listSessions } from "../api/client";
import type { SessionInfo } from "../api/types";
import { pollDelayMs, withLiveStatus } from "../model/session";

export interface SessionsState {
  sessions: SessionInfo[];
  error: string | null;
  /** 首次拉取是否已完成。空列表和「还没拉到」是两回事，UI 得能区分。 */
  loaded: boolean;
  /** 立刻重取（新建/删除会话后调，不必等下一个轮询周期）。 */
  refresh: () => void;
}

/**
 * @param live 选中会话的 WS 实时状态。轮询结果里那一行会被它覆盖 —— 否则会出现
 *   「时间线已经在跑、侧栏还写着空闲」的割裂（轮询最长滞后 10 秒）。
 */
export function useSessions(live: SessionInfo | null): SessionsState {
  const [raw, setRaw] = useState<SessionInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [nonce, setNonce] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;

    const clear = (): void => {
      if (timer.current !== null) {
        clearTimeout(timer.current);
        timer.current = null;
      }
    };

    const tick = async (): Promise<void> => {
      if (document.visibilityState === "hidden") {
        // 页面不可见时不打服务端；等 visibilitychange 叫醒。
        return;
      }
      let next: SessionInfo[] | null = null;
      try {
        const res = await listSessions();
        next = res.sessions;
        if (!cancelled) setError(null);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : String(e));
        }
      }
      if (cancelled) return;
      if (next !== null) {
        setRaw(next);
        setLoaded(true);
      }
      // 用刚拿到的数据算间隔（不是 state 里的旧值，避免差一拍）
      const delay = pollDelayMs(next ?? []);
      clear();
      timer.current = setTimeout(() => void tick(), delay);
    };

    void tick();

    const onVisible = (): void => {
      if (document.visibilityState === "visible") {
        clear();
        void tick();
      }
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      cancelled = true;
      clear();
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [nonce]);

  return { sessions: withLiveStatus(raw, live), error, loaded, refresh };
}
