/**
 * 产出文件面板：从事件流**在客户端**重建「本会话写过的文件」。
 *
 * 之所以做得到，是因为 `ToolCallStarted.detail.args` 带着完整参数：
 *   - `write_file(path, content)` —— **content 是整份文件正文**（`tools/filesystem.py:30-32`）
 *   - `edit_file(path, old_string, new_string, replace_all)` （:35-39）
 *   - `delete_file(path)`（:42-44）
 *
 * **不需要后端改任何东西**。
 *
 * ## 两条已知边界（不绕过，如实标注）
 *
 * 1. **只能重建「写」，看不到「读」**：`read_file` 的结果被 `core.py:130-134` 丢掉了
 *    （`TOOL_CALL_COMPLETED` 不带 detail），所以 agent 读过什么、内容是什么，事件流里没有。
 * 2. **`ToolCallCompleted` ≠ 写入成功**：文件工具内部把异常转成 `"ERROR: …"` 字符串
 *    返回（`filesystem.py:58-59` 的 `_err`），这种情况下 `tool.invoke` 不抛，照样发
 *    `ToolCallCompleted`。所以「写完的内容」是**尽力而为的重建**，不是磁盘真值的保证。
 *    真正以磁盘为准的是 workspace 目录本身。
 */
import type { ToolCallStartedDetail } from "../api/types";
import type { TimelineEvent } from "./replay";

export interface FileEntry {
  path: string;
  writes: number;
  edits: number;
  deletes: number;
  deleted: boolean;
  /** 尽力重建的最新全文；没经历过 `write_file` 的话是 null（只有 edit 记录）。 */
  content: string | null;
  /** 人类可读的操作轨迹，最新在后。 */
  history: string[];
}

export interface FileShadow {
  files: FileEntry[];
  /** 文件数（不含已删除）。 */
  count: number;
}

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

/** 在已有全文上套一次替换。`replace_all` 对应后端的同名参数。 */
function applyEdit(content: string, oldStr: string, newStr: string, replaceAll: boolean): string {
  if (oldStr === "") return content;
  if (replaceAll) return content.split(oldStr).join(newStr);
  const i = content.indexOf(oldStr);
  if (i === -1) return content;
  return content.slice(0, i) + newStr + content.slice(i + oldStr.length);
}

/**
 * 从**已按 seq 排序**的事件流重建文件影子。文件按首次出现顺序排列。
 */
export function buildFileShadow(items: TimelineEvent[]): FileShadow {
  const byPath = new Map<string, FileEntry>();
  const order: string[] = [];

  const entryFor = (path: string): FileEntry => {
    let e = byPath.get(path);
    if (!e) {
      e = { path, writes: 0, edits: 0, deletes: 0, deleted: false, content: null, history: [] };
      byPath.set(path, e);
      order.push(path);
    }
    return e;
  };

  for (const { event } of items) {
    if (event.type !== "ToolCallStarted") continue;
    const args = (event.detail as ToolCallStartedDetail | null)?.args;
    if (!args) continue;

    const name = event.message;
    const path = str(args.path);
    if (!path) continue;

    if (name === "write_file") {
      const entry = entryFor(path);
      const content = str(args.content);
      entry.writes += 1;
      entry.deleted = false;
      entry.content = content;
      const lines = content === "" ? 0 : content.split("\n").length;
      entry.history.push(`写入全文（${lines} 行 / ${content.length} 字符）`);
    } else if (name === "edit_file") {
      const entry = entryFor(path);
      const oldStr = str(args.old_string);
      const newStr = str(args.new_string);
      const replaceAll = args.replace_all === true;
      entry.edits += 1;
      entry.deleted = false;
      if (entry.content !== null) {
        const matched = oldStr !== "" && entry.content.includes(oldStr);
        entry.content = applyEdit(entry.content, oldStr, newStr, replaceAll);
        const times = replaceAll ? "全部" : "首处";
        entry.history.push(matched ? `第 ${entry.edits} 次编辑（${times}替换）` : `第 ${entry.edits} 次编辑（原文未匹配，已跳过）`);
      } else {
        entry.history.push(`第 ${entry.edits} 次编辑（无全文基线，只能记次数）`);
      }
    } else if (name === "delete_file") {
      const entry = entryFor(path);
      entry.deletes += 1;
      entry.deleted = true;
      entry.history.push("删除");
    }
  }

  const files = order.map((p) => byPath.get(p)!).filter((e) => e.writes + e.edits + e.deletes > 0);
  return { files, count: files.filter((f) => !f.deleted).length };
}

/** 从路径猜一个语言标签，给代码块的 class 用（纯装饰，猜不出就给空）。 */
export function guessLanguage(path: string): string {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  const map: Record<string, string> = {
    py: "python",
    js: "javascript",
    jsx: "jsx",
    ts: "typescript",
    tsx: "tsx",
    json: "json",
    md: "markdown",
    css: "css",
    html: "html",
    yml: "yaml",
    yaml: "yaml",
    toml: "toml",
    sh: "bash",
    txt: "text",
  };
  return map[ext] ?? "";
}
