/**
 * `api/stream.ts` ↔ React 状态。
 *
 * ## StrictMode 的双挂载陷阱（React 19 开发态必然触发）
 *
 * `main.tsx` 用了 `<StrictMode>`，开发态它会**挂载 → 卸载 → 再挂载**。真正的危险不是
 * 「短暂有两个 socket」，而是**旧 socket 的回调在第二个已存在之后才触发**：
 * 旧 socket 的 `onclose` 会排一次重连 → 两个活着的 socket + 退避计数翻倍。
 *
 * 两道防线：
 * 1. effect 的依赖**只有 `sessionId`**（`lastSeq` 留在 `stream.ts` 的闭包里 ——
 *    放进依赖数组会导致每收到一条事件就重连一次）；
 * 2. `cancelled` 标志 + 每次回调都验一遍，卸载后旧回调一律不写 state。
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { openSessionStream, type Stream, type StreamState } from "../api/stream";
import type { EventEnvelope, SessionInfo } from "../api/types";

export interface SessionStream {
  /** 已收到的事件，按 seq 升序（WS 回填本身有序，去重在 stream.ts 里做过）。 */
  envelopes: EventEnvelope[];
  /** WS 推来的最新会话状态；比轮询新。 */
  live: SessionInfo | null;
  state: StreamState;
  /** 手动重连（页面重新可见时用）。 */
  reconnect: () => void;
}

export function useSessionStream(sessionId: string | null): SessionStream {
  const [envelopes, setEnvelopes] = useState<EventEnvelope[]>([]);
  const [live, setLive] = useState<SessionInfo | null>(null);
  const [state, setState] = useState<StreamState>({ kind: "connecting", attempt: 0 });
  const [epoch, setEpoch] = useState(0);
  const streamRef = useRef<Stream | null>(null);

  useEffect(() => {
    if (sessionId === null) {
      setEnvelopes([]);
      setLive(null);
      return;
    }

    let cancelled = false;
    // 换会话必须清空：上一个会话的事件流混进来会渲染出别人的历史。
    setEnvelopes([]);
    setLive(null);

    const stream = openSessionStream(sessionId, {
      onStatus: (session) => {
        if (!cancelled) setLive(session);
      },
      onEvent: (envelope) => {
        if (!cancelled) setEnvelopes((prev) => [...prev, envelope]);
      },
      onState: (next) => {
        if (!cancelled) setState(next);
      },
    });
    streamRef.current = stream;

    return () => {
      cancelled = true;
      stream.close();
      streamRef.current = null;
    };
  }, [sessionId, epoch]);

  const reconnect = useCallback(() => {
    streamRef.current?.reopen();
    // reopen 对已 stopped 的流是 no-op，所以顺带踢一次 effect 兜底。
    setEpoch((n) => n + 1);
  }, []);

  return { envelopes, live, state, reconnect };
}
