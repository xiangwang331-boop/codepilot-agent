/**
 * 产出文件面板的**客户端重建**。
 *
 * 依据只有一处：`ToolCallStarted.detail.args` 带完整参数。所以这一层的每一条断言都
 * 对应一个「后端真的这么发」的事实（`tools/filesystem.py` 的 schema），而不是我编的形状。
 *
 * ⚠️ 它是**尽力重建**，不是磁盘真值：文件工具内部把异常转成 `"ERROR: …"` 字符串返回，
 * 那种情况下 `tool.invoke` 不抛、照样发 `ToolCallCompleted`。真正的真值是 workspace 目录。
 */
import { describe, expect, it } from "vitest";

import type { AgentEvent, EventType } from "../api/types";
import { APPROVED, BATCH } from "./__fixtures__";
import { buildFileShadow, guessLanguage } from "./files";
import { markReruns } from "./replay";

function tool(
  name: string,
  args: Record<string, unknown>,
  type: EventType = "ToolCallStarted",
): AgentEvent {
  return {
    type,
    agent: "coder",
    message: name,
    detail: { args },
    timestamp: "2026-01-01T00:00:00+00:00",
    thread_id: "t",
    step: 2,
    node: "tools",
  };
}

function seq(...events: AgentEvent[]): ReturnType<typeof markReruns> {
  return markReruns(events.map((event, i) => ({ kind: "event" as const, seq: i, event })));
}

describe("buildFileShadow", () => {
  it("write_file 重建全文（content 就是整份正文）", () => {
    const s = buildFileShadow(seq(tool("write_file", { path: "main.py", content: "a\nb\nc\n" })));
    expect(s.count).toBe(1);
    expect(s.files[0]).toMatchObject({ path: "main.py", writes: 1, edits: 0, deleted: false });
    expect(s.files[0]!.content).toBe("a\nb\nc\n");
    expect(s.files[0]!.history).toEqual(["写入全文（4 行 / 6 字符）"]);
  });

  it("空内容算 0 行（不是 1 行）", () => {
    const s = buildFileShadow(seq(tool("write_file", { path: "empty.py", content: "" })));
    expect(s.files[0]!.history).toEqual(["写入全文（0 行 / 0 字符）"]);
  });

  it("同文件再写一次 → writes 累加、内容换成新的", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "old" }),
        tool("write_file", { path: "a.py", content: "new" }),
      ),
    );
    expect(s.files).toHaveLength(1);
    expect(s.files[0]).toMatchObject({ writes: 2, content: "new" });
  });

  it("edit_file 在已有全文上套替换（默认只换首处）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "x\nfoo\ny\nfoo\n" }),
        tool("edit_file", { path: "a.py", old_string: "foo", new_string: "bar" }),
      ),
    );
    expect(s.files[0]!.content).toBe("x\nbar\ny\nfoo\n");
    expect(s.files[0]!.edits).toBe(1);
    expect(s.files[0]!.history[1]).toBe("第 1 次编辑（首处替换）");
  });

  it("replace_all=true 换全部", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "foo foo foo" }),
        tool("edit_file", {
          path: "a.py",
          old_string: "foo",
          new_string: "bar",
          replace_all: true,
        }),
      ),
    );
    expect(s.files[0]!.content).toBe("bar bar bar");
    expect(s.files[0]!.history[1]).toBe("第 1 次编辑（全部替换）");
  });

  it("old_string 在全文里找不到 → 内容不动，但历史里说清楚（不假装改成功了）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "hello" }),
        tool("edit_file", { path: "a.py", old_string: "nope", new_string: "x" }),
      ),
    );
    expect(s.files[0]!.content).toBe("hello");
    expect(s.files[0]!.edits).toBe(1);
    expect(s.files[0]!.history[1]).toBe("第 1 次编辑（原文未匹配，已跳过）");
  });

  it("没有全文基线时只记次数（agent 编辑了一个它没写过的文件）", () => {
    const s = buildFileShadow(
      seq(tool("edit_file", { path: "legacy.py", old_string: "a", new_string: "b" })),
    );
    expect(s.files[0]).toMatchObject({ edits: 1, writes: 0, content: null });
    expect(s.files[0]!.history).toEqual(["第 1 次编辑（无全文基线，只能记次数）"]);
  });

  it("old_string 为空串 → 不改内容（避免 JS 的 split('') 把文件炸成字符数组）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "abc" }),
        tool("edit_file", { path: "a.py", old_string: "", new_string: "-", replace_all: true }),
      ),
    );
    expect(s.files[0]!.content).toBe("abc");
  });

  it("delete_file 标记删除，且**不计入 count**（但条目还在，能看见发生过什么）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "keep.py", content: "k" }),
        tool("write_file", { path: "gone.py", content: "g" }),
        tool("delete_file", { path: "gone.py" }),
      ),
    );
    expect(s.files.map((f) => f.path)).toEqual(["keep.py", "gone.py"]);
    expect(s.count).toBe(1);
    expect(s.files[1]).toMatchObject({ deleted: true, deletes: 1 });
  });

  it("删除后又写回来 → 不再是删除态", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "1" }),
        tool("delete_file", { path: "a.py" }),
        tool("write_file", { path: "a.py", content: "2" }),
      ),
    );
    expect(s.files[0]).toMatchObject({ deleted: false, deletes: 1, writes: 2, content: "2" });
    expect(s.count).toBe(1);
  });

  it("按**首次出现**的顺序排列（不是按最后一次操作）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "z.py", content: "1" }),
        tool("write_file", { path: "a.py", content: "1" }),
        tool("write_file", { path: "z.py", content: "2" }),
      ),
    );
    expect(s.files.map((f) => f.path)).toEqual(["z.py", "a.py"]);
  });

  it("读类工具与非文件工具一概不进面板（只重建「写」，看不到「读」）", () => {
    const s = buildFileShadow(
      seq(
        tool("read_file", { path: "a.py" }),
        tool("list_files", { path: "." }),
        tool("search_code", { pattern: "foo" }),
        tool("run_command", { command: "pytest" }),
        tool("delegate", { specialist: "coder", task: "写 a.py" }),
      ),
    );
    expect(s.files).toEqual([]);
    expect(s.count).toBe(0);
  });

  it("args 缺 path / 非字符串 path → 跳过，不造出名为 \"undefined\" 的假文件", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { content: "no path" }),
        tool("write_file", { path: 42, content: "x" }),
        tool("write_file", null as unknown as Record<string, unknown>),
      ),
    );
    expect(s.files).toEqual([]);
  });

  it("Completed / Failed 事件本身不参与重建（只有 Started 带 args）", () => {
    const s = buildFileShadow(
      seq(
        tool("write_file", { path: "a.py", content: "x" }, "ToolCallCompleted"),
        tool("write_file", { path: "b.py", content: "y" }, "ToolCallFailed"),
      ),
    );
    expect(s.files).toEqual([]);
  });
});

// ---------------------------------------------------------------- 真实夹具

describe("对真实夹具重建", () => {
  it("批准夹具：重建出 main.py 与快排正文", () => {
    const s = buildFileShadow(markReruns(APPROVED));
    expect(s.files.map((f) => f.path)).toEqual(["main.py"]);
    expect(s.files[0]!.content).toBe("def quicksort(arr):\n    return sorted(arr)\n");
    expect(s.files[0]!.writes).toBe(1);
  });

  it("一批两个委派的夹具：**两个文件各写一次**，不是三次", () => {
    // 裸事件流里 write_file 有 3 条（a.py 出现两次：首轮 + 重放），
    // 但按 seq 折叠后 rebuild 会把 a.py 的第二次覆盖成同一个结果 —— 面板显示
    // 「a.py 写过 1 次」还是「2 次」是**有意的选择**：这里如实给出 2 次（事件真的发了两次），
    // 内容取最后一次。断言它，免得以后有人以为这是 bug。
    const s = buildFileShadow(markReruns(BATCH));
    expect(s.files.map((f) => f.path)).toEqual(["a.py", "b.py"]);
    expect(s.files.map((f) => [f.writes, f.content])).toEqual([
      [2, "# a.py\n"],
      [1, "# b.py\n"],
    ]);
    expect(s.count).toBe(2);
  });
});

describe("guessLanguage", () => {
  it("常见扩展名映射", () => {
    expect(guessLanguage("main.py")).toBe("python");
    expect(guessLanguage("a/b/c.tsx")).toBe("tsx");
    expect(guessLanguage("docker-compose.yml")).toBe("yaml");
  });

  it("大小写不敏感", () => {
    expect(guessLanguage("MAIN.PY")).toBe("python");
  });

  it("猜不出就给空串（纯装饰，不值得为它编一个）", () => {
    expect(guessLanguage("Makefile")).toBe("");
    expect(guessLanguage("a.zzz")).toBe("");
    expect(guessLanguage(".gitignore")).toBe("");
  });
});
