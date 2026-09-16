import { DelegationCard } from "./DelegationCard";
import { SimpleRow } from "./SimpleRow";
import { ToolRowView } from "./ToolRowView";
import type { Block } from "../model/events";

/**
 * 块的派发器。
 *
 * 委派卡片的子块通过 **React children** 传下去（而不是让 `DelegationCard` 反过来
 * import 本文件）—— 那样就成了 `BlockViews ⇄ DelegationCard` 的循环 import。
 * 组件在渲染期才调用，循环 import 多数情况下能跑，但一旦打包器把它变成
 * `undefined` 就是运行时崩溃，不值得为省这三行冒险。
 */
export function BlockView({ block }: { block: Block }): React.JSX.Element {
  switch (block.kind) {
    case "delegation":
      return (
        <DelegationCard block={block}>
          <BlockList blocks={block.blocks} />
        </DelegationCard>
      );
    case "tool":
      return <ToolRowView row={block} />;
    default:
      return <SimpleRow block={block} />;
  }
}

export function BlockList({ blocks }: { blocks: Block[] }): React.JSX.Element {
  return (
    <>
      {blocks.map((b) => (
        <BlockView key={b.id} block={b} />
      ))}
    </>
  );
}
