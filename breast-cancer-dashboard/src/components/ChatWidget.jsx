import { useEffect, useRef, useState } from "react";
import { MessageCircle, X, Send } from "lucide-react";
import { chat } from "../api"; // adjust if your api file path is different

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState([
    { role: "assistant", content: "Hi! Ask me anything about the dashboard or predictions." },
  ]);

  const listRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [open, msgs]);
const send = async () => {
  const text = input.trim();
  if (!text || busy) return;

  const next = [...msgs, { role: "user", content: text }];
  setMsgs(next);
  setInput("");
  setBusy(true);

  try {
    // ✅ send FULL messages array
    const reply = await chat(next);

    setMsgs((prev) => [
      ...prev,
      { role: "assistant", content: reply },
    ]);
  } catch (e) {
    setMsgs((prev) => [
      ...prev,
      { role: "assistant", content: `⚠️ ${e.message}` },
    ]);
  } finally {
    setBusy(false);
  }
};


  return (
    <>
      {/* floating toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-6 right-6 z-[9999] rounded-2xl bg-gradient-to-br from-pink-500 to-violet-500 p-4 text-white shadow-xl hover:shadow-2xl transition"
        title="Chat"
      >
        {open ? <X size={20} /> : <MessageCircle size={20} />}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-[9999] w-[360px] max-w-[90vw] overflow-hidden rounded-3xl border border-slate-200 bg-white/90 shadow-2xl backdrop-blur">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
            <div>
              <div className="text-sm font-extrabold text-slate-900">Dashboard Assistant</div>
              <div className="text-xs text-slate-500">{busy ? "Thinking..." : "Online"}</div>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="rounded-xl px-2 py-1 text-slate-600 hover:bg-slate-100"
              title="Close"
            >
              <X size={18} />
            </button>
          </div>

          <div ref={listRef} className="max-h-[420px] overflow-auto px-3 py-3 space-y-2">
            {msgs.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
                    m.role === "user" ? "bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100" : "bg-slate-100 text-slate-900"
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-slate-200 p-3">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                disabled={busy}
className="
  w-full rounded-2xl
  border border-white/10
  bg-slate-900/90
  px-3 py-2 text-sm
  text-slate-100
  placeholder:text-slate-400
  outline-none
  focus:border-pink-400/60
  focus:ring-2 focus:ring-pink-400/30
  disabled:opacity-60
"
                placeholder="Type a message..."
              />
              <button
                onClick={send}
                disabled={busy || !input.trim()}
                className="rounded-2xl bg-gradient-to-br from-pink-500 to-violet-500 px-4 py-2 text-white shadow disabled:opacity-50"
                title="Send"
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
