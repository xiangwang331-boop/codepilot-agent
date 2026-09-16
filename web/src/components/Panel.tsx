import { useState, type ReactNode } from "react";

/** 检查器里的可折叠小节。 */
export function Panel({
  title,
  count,
  defaultOpen = true,
  children,
}: {
  title: string;
  count?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}): React.JSX.Element {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="panel">
      <button className="panel__head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="panel__caret" data-open={open}>
          ▶
        </span>
        {title}
        {count !== undefined && <span className="panel__count">{count}</span>}
      </button>
      {open && <div className="panel__body">{children}</div>}
    </section>
  );
}
