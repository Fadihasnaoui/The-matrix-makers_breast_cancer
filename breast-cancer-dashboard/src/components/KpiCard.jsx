export default function KpiCard({
  title,
  value,
  hint,
  right,
  badge,
  tone = "neutral", // neutral | good | warn | bad
}) {
  const toneMap = {
    neutral: "border-slate-200",
    good: "border-emerald-200",
    warn: "border-amber-200",
    bad: "border-rose-200",
  };

  const glowMap = {
    neutral: "shadow-[0_10px_22px_rgba(148,163,184,0.22)]",
    good: "shadow-[0_12px_28px_rgba(16,185,129,0.18)]",
    warn: "shadow-[0_12px_28px_rgba(245,158,11,0.18)]",
    bad: "shadow-[0_12px_28px_rgba(244,63,94,0.18)]",
  };

  return (
    <div
      className={[
        "rounded-3xl border bg-white/90 p-4 backdrop-blur transition",
        "hover:-translate-y-[1px] hover:shadow-[0_18px_40px_rgba(15,23,42,0.12)]",
        toneMap[tone],
        glowMap[tone],
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="text-xs font-extrabold text-slate-500 uppercase tracking-wide">
              {title}
            </div>
            {badge ? (
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-600 border border-slate-200">
                {badge}
              </span>
            ) : null}
          </div>

          <div className="mt-1 text-3xl font-extrabold text-slate-900 leading-none">
            {value}
          </div>

          {hint ? (
            <div className="mt-2 text-xs text-slate-600 leading-snug">{hint}</div>
          ) : null}
        </div>

        {right ? <div className="shrink-0">{right}</div> : null}
      </div>
    </div>
  );
}
