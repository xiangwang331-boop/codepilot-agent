import { useRef, useState } from "react";

import { canSend } from "../model/session";
import type { SessionStatus } from "../api/types";

/**
 * 需求输入框。Enter 发送 / Shift+Enter 换行。
 *
 * 忙碌时禁用而不是隐藏：用户得看得见「现在不能发」，否则会以为界面卡了。
 * 提示语说明**为什么**不能发 —— 尤其是挂起态，它其实在等人去点批准。
 */
export function Composer({
  status,
  onSend,
}: {
  status: SessionStatus;
  onSend: (task: string) => void;
}): React.JSX.Element {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement | null>(null);

  const enabled = canSend(status);
  const trimmed = text.trim();

  const submit = (): void => {
    if (!enabled || trimmed === "") return;
    onSend(trimmed);
    setText("");
    // 高度复位（auto-grow 是下面 onInput 干的）
    if (area.current) area.current.style.height = "auto";
  };

  const hint = (() => {
    switch (status) {
      case "running":
        return "任务执行中，等它跑完或停下来再下发新指令。";
      case "awaiting_approval":
        return "有委派在等批准 —— 先在右侧点「批准执行」或「拒绝」，会话才能继续。";
      case "closed":
        return "会话已关闭，新建一个才能继续。";
      default:
        return "Enter 发送 / Shift+Enter 换行。会话有历史时会作为追加指令下发。";
    }
  })();

  return (
    <>
      <div className="composer">
        <textarea
          ref={area}
          className="composer__input"
          rows={1}
          value={text}
          placeholder={enabled ? "描述开发需求，例如：实现一个快排并补上单元测试" : "当前不可下发指令"}
          disabled={!enabled}
          onChange={(e) => setText(e.target.value)}
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = "auto";
            el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button className="composer__send" disabled={!enabled || trimmed === ""} onClick={submit}>
          发送
        </button>
      </div>
      <div className="composer__hint">{hint}</div>
    </>
  );
}
