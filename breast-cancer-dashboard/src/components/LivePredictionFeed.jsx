import { Activity, HeartPulse } from "lucide-react";

export default function LivePredictionFeed({ items = [] }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-extrabold text-slate-100">
            Live prediction feed
          </div>
          <div className="mt-1 text-xs text-slate-300">
            Most recent model inferences
          </div>
        </div>
        <Activity className="text-cyan-300" size={18} />
      </div>

      <div className="mt-4 space-y-2 max-h-[260px] overflow-auto pr-1">
        {items.length === 0 && (
          <div className="text-xs text-slate-400">
            No predictions yet.
          </div>
        )}

        {items.map((p, i) => {
          const malignant = p.prediction === 1;
          return (
            <div
              key={i}
              className={`flex items-center justify-between rounded-xl border px-3 py-2 text-xs
                ${
                  malignant
                    ? "border-rose-400/30 bg-rose-500/10"
                    : "border-emerald-400/30 bg-emerald-500/10"
                }`}
            >
              <div className="flex items-center gap-2">
                <HeartPulse
                  size={14}
                  className={malignant ? "text-rose-300" : "text-emerald-300"}
                />
                <span className="font-bold">
                  {malignant ? "Malignant" : "Benign"}
                </span>
              </div>

              <div className="text-right">
                <div className="font-mono text-slate-200">
                  {p.confidence != null
                    ? `${(p.confidence * 100).toFixed(1)}%`
                    : "—"}
                </div>
                <div className="text-[10px] text-slate-400">
                  {p.latency_ms} ms
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
