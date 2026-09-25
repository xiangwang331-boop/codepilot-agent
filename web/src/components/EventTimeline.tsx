import { useEffect, useMemo, useRef, useState } from "react";

import { BlockList } from "./BlockViews";
import { applyFilters, countBlocks, type Block, type FilterState } from "../model/events";

/** 判定「贴着底」的容差：小于这个距离就认为用户在跟读。 */
const BOTTOM_THRESHOLD = 32;

/**
 * 时间线。**接收已经折好的块**（不是原始事件）—— 折叠是 O(n) 的纯计算，
 * 在 `App` 里做一次就够（`collectAgents` 也要用同一棵树），这里只负责过滤与渲染。
 *
 * ## 跟随 / 不打扰，以及角标
 *
 * 事件是连续喷的，所以「自动滚到底」必须**有条件**：无脑贴底的话，用户往上翻去看
 * 第 3 步在干什么时，每来一条新事件就把他拽回底部——这是实时日志最难受的体验。
 * 这里由滚动位置决定：贴底 = 在跟读 → 跟着滚；翻上去 = 在读历史 → 停住不动，
 * 但给一个「N 条新事件 ↓」的角标和一键回底部的出口。
 *
 * 角标的数**必须**由「渲染行数的增量」算出来，不能只判「在不在底部」：
 * 判据若只有 `!atBottom`，那么在一个早就跑完、什么都不再发生的会话里往上滚一下，
 * 它也会冒出来并写着「有新事件」—— 把「你不在底部」说成了「有新东西」。
 */
export function EventTimeline({
  blocks,
  filters,
  historyAvailable,
}: {
  blocks: Block[];
  filters: FilterState;
  /** false（sqlite 后端）时空时间线要说清「不是没跑过，是没落库」。 */
  historyAvailable: boolean;
}): React.JSX.Element {
  const scroller = useRef<HTMLDivElement | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  /** 翻上去之后新来了多少行 —— 只增不减，回到/滚到底部时清零。 */
  const [pending, setPending] = useState(0);

  const visible = useMemo(() => applyFilters(blocks, filters), [blocks, filters]);
  // 计数走过滤后的列表（见 `countBlocks` 的 docstring），且用**数**而不是数组引用
  // 做依赖 —— `blocks` 每次重算都可能换引用，数才是稳定的信号。
  const shown = useMemo(() => countBlocks(visible), [visible]);

  /** 上一次「已经算见过」的行数 + 当时的过滤条件 —— 「新来了几行」由这两者一起算。 */
  const seen = useRef({ count: shown, filters });

  useEffect(() => {
    const prev = seen.current;
    seen.current = { count: shown, filters };
    // 换过滤条件改的是「看什么」，不是「来了什么」—— 这笔账不能记进角标，重设基线即可。
    // （`filters` 是 `App` 的 state，只在用户真改时换引用，故按引用比就是对的。）
    const grew = prev.filters === filters ? shown - prev.count : 0;

    if (atBottom) {
      setPending(0); // 贴底 = 全都看见了，没有「没看见的」
    } else if (grew > 0) {
      setPending((n) => n + grew);
    }
  }, [shown, filters, atBottom]);

  const onScroll = (): void => {
    const el = scroller.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAtBottom(distance <= BOTTOM_THRESHOLD);
  };

  // 贴底时才跟着新事件滚。用户往上翻就是在读历史，这时候把他拽回底部是最烦人的交互。
  // 依赖是**渲染行数**而不是顶层块数：委派的子事件在同一个块里长，顶层块数不变 ——
  // 用顶层块数的话，一次长委派从头到尾都不会触发跟随（页面看着像卡住了）。
  useEffect(() => {
    const el = scroller.current;
    if (!el || !atBottom) return;
    // 直接改 scrollTop 而不是 scrollIntoView：后者每来一条事件就抖一次布局。
    el.scrollTop = el.scrollHeight;
  }, [shown, atBottom]);

  const jump = (): void => {
    const el = scroller.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAtBottom(true); // 角标由上面那个 effect 统一清零，这里不再重复写一份
  };

  // **只有一个滚动容器**：空态也走这里。
  //
  // 原先空态是单独一支 `return`，那支的 `.timeline` 上没有 `ref`、没有 `onScroll` ——
  // 而「刚进入」那一刻恰恰经常就是空态（回填还在路上，`blocks` 先是 `[]`，随后才灌进来）。
  // 于是进入路径上存在一个 `scroller.current === null` 的窗口，两个 effect（跟随 / 角标）
  // 都靠它，命不中就静默什么都不做。React 会把同一个 `div.timeline` 复用掉，所以这不是
  // 「必崩」的 bug，但它让**进入**与**稳定后**成为两种 DOM 形态 —— 而用户报的正是
  // 「一进去不行、刷新才行」。把形态收成一个，这个差异就不存在了。
  return (
    <div className="timeline" ref={scroller} onScroll={onScroll}>
      {blocks.length === 0 ? (
        <div className="empty">
          <h2>还没有事件</h2>
          {/*
            两种「空」完全不同：一种是**还没跑过**（去输入需求就有），
            另一种是**跑过但读不回来**（sqlite 后端事件不落库，重启即丢）。
            后者的空时间线是既成事实，让用户以为是界面 bug 才是最坏的结果。
          */}
          {historyAvailable ? (
            <p>在下面输入开发需求，Supervisor 会把它拆给 specialist 执行，过程实时显示在这里。</p>
          ) : (
            <p>
              sqlite 后端的事件只存在进程内存里，<strong>服务重启即丢</strong>：
              重启前跑过的会话从 checkpoint 恢复后就是空的，新建的会话则是还没下发过指令。
              要留住事件流，用 <code>PERSISTENCE_BACKEND=postgres</code> 后端。
            </p>
          )}
        </div>
      ) : (
        <BlockList blocks={visible} />
      )}
      {/*
        两个条件都要：`pending > 0` 是「真的有新东西」（贴底时它恒为 0，见上面那个 effect），
        `!atBottom` 则挡掉 `jump()` 之后那一帧 —— effect 在 paint 之后才跑，
        只判 pending 的话点下去会闪一下还没归零的角标。`blocks.length > 0` 是空态下的闸门：
        角标说的是「有新事件」，一行都没有的时候它没有意义。
      */}
      {blocks.length > 0 && !atBottom && pending > 0 && (
        <button className="jump" onClick={jump} title="跳到最后一条事件">
          {pending} 条新事件 ↓
        </button>
      )}
    </div>
  );
}
