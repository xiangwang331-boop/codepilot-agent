import { useEffect, useMemo, useRef, useState } from "react";

import { BlockList } from "./BlockViews";
import { applyFilters, type Block, type FilterState } from "../model/events";

/** 判定「贴着底」的容差：小于这个距离就认为用户在跟读。 */
const BOTTOM_THRESHOLD = 32;

/**
 * 时间线。**接收已经折好的块**（不是原始事件）—— 折叠是 O(n) 的纯计算，
 * 在 `App` 里做一次就够（`collectAgents` 也要用同一棵树），这里只负责过滤与渲染。
 */
export function EventTimeline({
  blocks,
  filters,
}: {
  blocks: Block[];
  filters: FilterState;
}): React.JSX.Element {
  const scroller = useRef<HTMLDivElement | null>(null);
  const [atBottom, setAtBottom] = useState(true);

  const visible = useMemo(() => applyFilters(blocks, filters), [blocks, filters]);

  const onScroll = (): void => {
    const el = scroller.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAtBottom(distance <= BOTTOM_THRESHOLD);
  };

  // 贴底时才跟着新事件滚。用户往上翻就是在读历史，这时候把他拽回底部是最烦人的交互。
  useEffect(() => {
    const el = scroller.current;
    if (!el || !atBottom) return;
    // 直接改 scrollTop 而不是 scrollIntoView：后者每来一条事件就抖一次布局。
    el.scrollTop = el.scrollHeight;
  }, [blocks.length, atBottom]);

  const jump = (): void => {
    const el = scroller.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAtBottom(true);
  };

  if (blocks.length === 0) {
    return (
      <div className="timeline">
        <div className="empty">
          <h2>还没有事件</h2>
          <p>在下面输入开发需求，Supervisor 会把它拆给 specialist 执行，过程实时显示在这里。</p>
        </div>
      </div>
    );
  }

  return (
    <div className="timeline" ref={scroller} onScroll={onScroll}>
      <BlockList blocks={visible} />
      {!atBottom && (
        <button className="jump" onClick={jump}>
          有新事件 ↓
        </button>
      )}
    </div>
  );
}
