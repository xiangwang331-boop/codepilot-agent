import { useState } from "react";

import { Panel } from "./Panel";
import { guessLanguage, type FileEntry, type FileShadow } from "../model/files";

/**
 * 产出文件面板 —— 从 `write_file` / `edit_file` 事件的 `detail.args` **在客户端**
 * 重建「本会话写过哪些文件」。用的是 `tools/filesystem.py:30-39` 的 schema：
 * `write_file` 带整份 content，`edit_file` 带 old_string/new_string。
 *
 * **后端为此一行没改。**
 *
 * 两条边界（界面上如实说明，不含糊过去）：
 * 1. 只能重建「写」，看不到 agent **读**过什么（读的结果不在事件流里）；
 * 2. 是**尽力重建**，不是磁盘真值 —— 文件工具把异常转成 `"ERROR: …"` 字符串返回，
 *    这种情况下照样发 `ToolCallCompleted`。
 */
export function FilePanel({ shadow }: { shadow: FileShadow }): React.JSX.Element {
  return (
    <Panel
      title="产出文件"
      count={shadow.files.length > 0 ? `${shadow.count} 个` : undefined}
      defaultOpen={shadow.files.length > 0}
    >
      {shadow.files.length === 0 ? (
        <div className="panel__empty" style={{ padding: 0 }}>
          本会话还没有写文件的操作。
        </div>
      ) : (
        <>
          {shadow.files.map((f) => (
            <FileItem key={f.path} file={f} />
          ))}
          <div className="delegation__note" style={{ padding: 0 }}>
            内容由事件参数重建，仅供参考；以 workspace 目录里的实际文件为准。
          </div>
        </>
      )}
    </Panel>
  );
}

function FileItem({ file }: { file: FileEntry }): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const lang = guessLanguage(file.path);

  const badge = [
    file.writes > 0 && `写 ${file.writes}`,
    file.edits > 0 && `改 ${file.edits}`,
    file.deletes > 0 && `删 ${file.deletes}`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="fileitem" data-deleted={file.deleted}>
      <button className="fileitem__head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="panel__caret" data-open={open}>
          ▶
        </span>
        <span className="fileitem__path" title={file.path}>
          {file.path}
        </span>
        <span className="fileitem__badge">{badge}</span>
      </button>

      {open && (
        <div className="fileitem__body">
          <ul className="fileitem__history">
            {file.history.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
          {file.content !== null ? (
            <div className="codeblock">
              {lang !== "" && <div className="codeblock__lang">{lang}</div>}
              <pre>{file.content}</pre>
            </div>
          ) : (
            <div className="delegation__note">没有全文基线（只发生过 edit，从未 write）。</div>
          )}
        </div>
      )}
    </div>
  );
}
