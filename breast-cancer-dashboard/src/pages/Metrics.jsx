import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { Crown, ShieldAlert, AlertTriangle, BarChart3 } from "lucide-react";
import { api, getApiBase } from "../api"; // ✅ use shared axios instance

const fmtPct = (v) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
const fmtNum = (v, d = 3) =>
  v === null || v === undefined ? "—" : Number(v).toFixed(d);

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

/**
 * UI label overrides (does NOT touch backend JSON).
 */
const MODEL_NAME_OVERRIDE = {
  "k-nn_(k=5)": "k-NN (k=5)",
  "k-nn_(optimized)": "k-NN (Optimized)",
};

function displayModelName(row) {
  if (!row) return "—";
  return MODEL_NAME_OVERRIDE[row.model_id] ?? row.model_name ?? row.model_id;
}

/** ✅ Heatmap-style Confusion Matrix (2x2) */
function ConfusionMatrix({ row }) {
  if (!row) return null;
  const { tn = 0, fp = 0, fn = 0, tp = 0 } = row;

  // intensity is based on COUNT magnitude (true heatmap)
  const max = Math.max(tn, fp, fn, tp, 1);
  const cellAlpha = (v) => 0.08 + 0.42 * (v / max);

  const rowBenign = tn + fp;
  const rowMalignant = fn + tp;
  const colBenign = tn + fn;
  const colMalignant = fp + tp;
  const total = rowBenign + rowMalignant;

  const toneColor = (tone) => {
    if (tone === "good") return "16,185,129"; // emerald-500
    if (tone === "warn") return "245,158,11"; // amber-500
    if (tone === "bad") return "244,63,94"; // rose-500
    return "148,163,184"; // slate
  };

  const borderFromTone = (tone) => {
    if (tone === "good") return "border-emerald-400/30";
    if (tone === "warn") return "border-amber-400/30";
    if (tone === "bad") return "border-rose-400/30";
    return "border-white/10";
  };

  const Cell = ({ title, value, subtitle, tone }) => {
    const rgb = toneColor(tone);
    const bg = `rgba(${rgb},${cellAlpha(value)})`;
    const border = borderFromTone(tone);

    return (
      <div
        className={`rounded-2xl border ${border} p-5 shadow-[0_0_40px_rgba(0,0,0,0.15)]`}
        style={{ background: bg }}
      >
        <div className="text-[11px] font-extrabold uppercase text-slate-200">
          {title}
        </div>
        <div className="mt-3 text-4xl font-extrabold text-white">{value}</div>
        <div className="mt-2 text-sm text-slate-300">{subtitle}</div>
      </div>
    );
  };

  // Legend chips that actually match the matrix colors + low/med/high intensities
  const LegendChip = ({ label, tone, a }) => {
    const rgb = toneColor(tone);
    return (
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "h-3 w-6 rounded border",
            borderFromTone(tone),
            "border-white/10"
          )}
          style={{ background: `rgba(${rgb},${a})` }}
        />
        <span>{label}</span>
      </div>
    );
  };

  // Use fixed alphas for the legend so it’s stable and easy to read
  const A_LOW = 0.14;
  const A_MED = 0.30;
  const A_HIGH = 0.50;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
      <div className="mb-5">
        <div className="text-base font-extrabold text-slate-100">
          Confusion Matrix
        </div>
        <div className="mt-1 text-sm text-slate-300">
          Holdout test • Negative = Benign • Positive = Malignant
        </div>
      </div>

      {/* MATRIX */}
      <div className="grid grid-cols-12 gap-4">
        {/* Y axis */}
        <div className="col-span-12 sm:col-span-1 flex sm:flex-col items-center justify-center text-xs font-bold text-slate-400 uppercase">
          Actual
        </div>

        <div className="col-span-12 sm:col-span-11">
          <div className="grid grid-cols-3 gap-4">
            <div />
            <div className="text-center text-xs font-bold text-slate-400">
              Predicted Benign
            </div>
            <div className="text-center text-xs font-bold text-slate-400">
              Predicted Malignant
            </div>

            {/* Actual Benign */}
            <div className="flex items-center text-xs font-bold text-slate-400">
              Actual Benign
            </div>
            <Cell title="TN" value={tn} subtitle="Benign → Benign" tone="good" />
            <Cell title="FP" value={fp} subtitle="Benign → Malignant" tone="warn" />

            {/* Row total */}
            <div />
            <div className="col-span-2 text-right text-xs text-slate-400">
              Row total: <b>{rowBenign}</b>
            </div>

            {/* Actual Malignant */}
            <div className="flex items-center text-xs font-bold text-rose-300">
              Actual Malignant
            </div>
            <Cell
              title="FN"
              value={fn}
              subtitle="Malignant → Benign (CRITICAL)"
              tone="bad"
            />
            <Cell
              title="TP"
              value={tp}
              subtitle="Malignant → Malignant"
              tone="good"
            />

            {/* Row total */}
            <div />
            <div className="col-span-2 text-right text-xs text-slate-400">
              Row total: <b>{rowMalignant}</b>
            </div>

            {/* Column totals */}
            <div />
            <div className="text-center text-xs text-slate-400">
              Col total: <b>{colBenign}</b>
            </div>
            <div className="text-center text-xs text-slate-400">
              Col total: <b>{colMalignant}</b>
            </div>
          </div>
        </div>
      </div>

      {/* ✅ INTENSITY LEGEND */}
      <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4">
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <span className="font-bold text-slate-300">
            Intensity = relative count magnitude (vs max cell)
          </span>
          <span className="ml-auto">
            Total samples: <b className="text-slate-200">{total}</b>
          </span>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-white/10 bg-slate-950/40 p-3">
            <div className="text-xs font-bold text-slate-300 mb-2">
              Low / Medium / High intensity
            </div>
            <div className="flex flex-wrap gap-4 text-xs text-slate-400">
              <LegendChip label="Low" tone="good" a={A_LOW} />
              <LegendChip label="Medium" tone="good" a={A_MED} />
              <LegendChip label="High" tone="good" a={A_HIGH} />
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-slate-950/40 p-3">
            <div className="text-xs font-bold text-slate-300 mb-2">Outcome colors</div>
            <div className="flex flex-wrap gap-4 text-xs text-slate-400">
              <LegendChip label="Correct (TN/TP)" tone="good" a={A_MED} />
              <LegendChip label="False Positive (FP)" tone="warn" a={A_MED} />
              <LegendChip label="False Negative (FN)" tone="bad" a={A_MED} />
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-slate-950/40 p-3">
            <div className="text-xs font-bold text-slate-300 mb-2">Clinical read</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              FN is clinically critical. A small FN count can still be high risk—always
              interpret alongside FNR and the confusion totals.
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-slate-400">
        Clinical priority: minimizing false negatives (missed malignancies).
      </div>
    </div>
  );
}

function RocOverlay({ rows, roc, selectedId, rocMode, setRocMode }) {
  const series = useMemo(() => {
    return Object.entries(roc || {}).map(([model_id, v]) => {
      const pts = (v.fpr || []).map((x, i) => ({
        fpr: x,
        tpr: v.tpr?.[i] ?? null,
      }));
      return { model_id, auc: v.auc, pts };
    });
  }, [roc]);

  const COLORS = ["#22d3ee", "#a78bfa", "#fb923c", "#34d399", "#f472b6", "#fbbf24"];

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-base font-extrabold text-slate-100">ROC Curves</div>
          <div className="mt-1 text-sm text-slate-300">
            Receiver Operating Characteristic • Toggle focus on selected model
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setRocMode("all")}
            className={cn(
              "rounded-xl border px-3 py-2 text-xs font-bold transition",
              rocMode === "all"
                ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-200"
                : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
            )}
          >
            Show all
          </button>
          <button
            onClick={() => setRocMode("focus")}
            className={cn(
              "rounded-xl border px-3 py-2 text-xs font-bold transition",
              rocMode === "focus"
                ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-200"
                : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
            )}
          >
            Focus selected
          </button>
        </div>
      </div>

      <div className="h-[420px] rounded-2xl border border-white/10 bg-slate-950/50 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              type="number"
              dataKey="fpr"
              domain={[0, 1]}
              label={{
                value: "False Positive Rate",
                position: "insideBottom",
                offset: -5,
                fill: "#94a3b8",
              }}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="tpr"
              domain={[0, 1]}
              label={{
                value: "True Positive Rate",
                angle: -90,
                position: "insideLeft",
                fill: "#94a3b8",
              }}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12,
                color: "#e2e8f0",
              }}
            />
            <Legend wrapperStyle={{ color: "#e2e8f0" }} />

            {series.map((s, idx) => {
              const row = rows.find((r) => r.model_id === s.model_id);
              const name = displayModelName(row) ?? s.model_id;

              const isSel = s.model_id === selectedId;
              const focus = rocMode === "focus";

              return (
                <Line
                  key={s.model_id}
                  data={s.pts}
                  type="monotone"
                  dataKey="tpr"
                  dot={false}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={isSel ? 4 : 2}
                  opacity={focus ? (isSel ? 1 : 0.08) : 1}
                  name={`${name} (AUC ${fmtNum(s.auc, 3)})`}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-slate-400">
        Clinical note: ROC can look great even if specificity is weak. Always review confusion matrix + FNR.
      </div>
    </div>
  );
}

export default function Metrics() {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [report, setReport] = useState(null);
  const [selectedId, setSelectedId] = useState("");
  const [rocMode, setRocMode] = useState("all");

  useEffect(() => {
    // ✅ axios cancellation (works like AbortController)
    const controller = new AbortController();

    (async () => {
      try {
        setLoading(true);
        setErr(null);

        const res = await api.get("/evaluation-report", { signal: controller.signal });
        const json = res.data;

        setReport(json);
        setSelectedId(json?.recommended_model_id ?? json?.rows?.[0]?.model_id ?? "");
      } catch (e) {
        if (e?.name === "CanceledError") return;
        setErr(e?.message ?? String(e));
      } finally {
        setLoading(false);
      }
    })();

    return () => controller.abort();
  }, []);

  const rows = report?.rows ?? [];
  const roc = report?.roc ?? {};
  const nTest = report?.n_test ?? null;
  const posRate = report?.positive_rate_test ?? null;

  const selectedRow = useMemo(
    () => rows.find((r) => r.model_id === selectedId) ?? null,
    [rows, selectedId]
  );

  const recommendedRow = useMemo(
    () =>
      rows.find((r) => r.model_id === report?.recommended_model_id) ??
      rows[0] ??
      null,
    [rows, report?.recommended_model_id]
  );

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
        <div className="text-2xl font-extrabold">Loading evaluation report...</div>
        <div className="text-sm text-slate-400">API: {getApiBase()}</div>
      </div>
    );
  }

  if (err) {
    return (
      <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
        <div className="rounded-3xl border border-rose-400/30 bg-rose-500/10 p-6">
          <div className="flex items-center gap-2 text-lg font-extrabold text-rose-200">
            <AlertTriangle size={20} />
            Failed to load evaluation report
          </div>
          <div className="mt-2 text-sm text-rose-300">{err}</div>
          <div className="mt-3 text-xs text-rose-400">
            Check <code>{getApiBase()}/evaluation-report</code>
          </div>
        </div>
      </div>
    );
  }

  if (!rows.length) {
    return (
      <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div className="text-slate-300">No evaluation rows found.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
      {/* HERO */}
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl shadow-[0_0_60px_rgba(59,130,246,0.15)]">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/10 via-indigo-500/10 to-violet-500/10" />
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-violet-400/10 blur-3xl" />

        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-extrabold text-slate-200">
                <BarChart3 size={14} className="text-cyan-300" /> Model Evaluation
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-xs font-bold text-amber-200">
                <ShieldAlert size={14} /> Safety-first
              </span>
            </div>

            <h1 className="mt-3 text-3xl font-extrabold leading-tight">
              Clinical Performance & Risk Profile
            </h1>

            <p className="mt-2 max-w-2xl text-sm text-slate-300">
              Compare candidate models using holdout test metrics with emphasis on{" "}
              <b className="text-white">false negatives</b> (missed malignancies). Includes ROC overlays,
              confusion matrices, and executive recommendation.
            </p>

            <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-400">
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                Source: {report?.source}
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                Test samples: {nTest}
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                Positive rate: {fmtPct(posRate)}
              </span>
            </div>
          </div>

          {/* Model Selector */}
          <div className="w-full max-w-xs">
            <div className="mb-2 text-xs font-bold text-slate-300">Select model to inspect</div>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-100 outline-none backdrop-blur focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20"
            >
              {rows.map((r) => (
                <option key={r.model_id} value={r.model_id} className="bg-slate-900">
                  {displayModelName(r)}{" "}
                  {r.model_id === report?.recommended_model_id ? "• Recommended" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* KPI GRID */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-6">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
          <div className="text-xs font-bold text-slate-300">AUC</div>
          <div className="mt-2 text-2xl font-extrabold text-cyan-300">{fmtNum(selectedRow?.auc, 3)}</div>
          <div className="mt-1 text-xs text-slate-400">Class separation</div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
          <div className="text-xs font-bold text-slate-300">Accuracy</div>
          <div className="mt-2 text-2xl font-extrabold text-emerald-300">{fmtPct(selectedRow?.accuracy)}</div>
          <div className="mt-1 text-xs text-slate-400">Overall correct</div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
          <div className="text-xs font-bold text-slate-300">Recall</div>
          <div className="mt-2 text-2xl font-extrabold text-indigo-300">{fmtPct(selectedRow?.recall)}</div>
          <div className="mt-1 text-xs text-slate-400">Sensitivity (TPR)</div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
          <div className="text-xs font-bold text-slate-300">Precision</div>
          <div className="mt-2 text-2xl font-extrabold text-violet-300">{fmtPct(selectedRow?.precision)}</div>
          <div className="mt-1 text-xs text-slate-400">Positive pred value</div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
          <div className="text-xs font-bold text-slate-300">Specificity</div>
          <div className="mt-2 text-2xl font-extrabold text-slate-100">{fmtPct(selectedRow?.specificity)}</div>
          <div className="mt-1 text-xs text-slate-400">TNR (Selectivity)</div>
        </div>

        <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 backdrop-blur shadow-[0_0_30px_rgba(244,63,94,0.12)]">
          <div className="flex items-center gap-1 text-xs font-bold text-rose-300">
            <AlertTriangle size={12} /> FNR
          </div>
          <div className="mt-2 text-2xl font-extrabold text-rose-300">{fmtPct(selectedRow?.fnr)}</div>
          <div className="mt-1 text-xs text-rose-400">Miss rate (CRITICAL)</div>
        </div>
      </div>

      {/* Deployment Rationale */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <div className="mb-5">
          <div className="text-base font-extrabold text-slate-100">Deployment Rationale</div>
          <div className="mt-1 text-sm text-slate-300">
            Models ranked by lowest FNR first, then highest AUC
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 p-5">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-300 uppercase">
              <Crown size={14} /> Recommended
            </div>
            <div className="mt-3 text-xl font-extrabold text-white">{displayModelName(recommendedRow)}</div>
            <div className="mt-3 flex gap-4 text-sm">
              <div>
                <span className="text-slate-400">FNR:</span>{" "}
                <span className="font-bold text-emerald-300">{fmtPct(recommendedRow?.fnr)}</span>
              </div>
              <div>
                <span className="text-slate-400">AUC:</span>{" "}
                <span className="font-bold text-emerald-300">{fmtNum(recommendedRow?.auc, 3)}</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="text-xs font-bold text-slate-300 uppercase">Selected Model</div>
            <div className="mt-3 text-xl font-extrabold text-white">{displayModelName(selectedRow)}</div>
            <div className="mt-3 flex gap-4 text-sm">
              <div>
                <span className="text-slate-400">FNR:</span>{" "}
                <span className="font-bold text-cyan-300">{fmtPct(selectedRow?.fnr)}</span>
              </div>
              <div>
                <span className="text-slate-400">AUC:</span>{" "}
                <span className="font-bold text-cyan-300">{fmtNum(selectedRow?.auc, 3)}</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-amber-400/30 bg-amber-500/10 p-5">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-300 uppercase">
              <ShieldAlert size={14} /> Clinical Guardrail
            </div>
            <div className="mt-3 text-sm leading-relaxed text-amber-200">
              In screening, minimizing missed malignancies is prioritized. False positives are managed via
              confirmatory testing.
            </div>
          </div>
        </div>
      </div>

      {/* Leaderboard + ROC */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur lg:col-span-4">
          <div className="mb-5">
            <div className="text-base font-extrabold text-slate-100">Model Leaderboard</div>
            <div className="mt-1 text-sm text-slate-300">
              Click to inspect • Ranked by FNR → AUC
            </div>
          </div>

          <div className="space-y-3 max-h-[520px] overflow-auto pr-2">
            {rows.map((r) => {
              const active = r.model_id === selectedId;
              const isRec = r.model_id === report?.recommended_model_id;

              return (
                <button
                  key={r.model_id}
                  onClick={() => setSelectedId(r.model_id)}
                  className={cn(
                    "w-full rounded-2xl border p-4 text-left transition-all",
                    active
                      ? "border-cyan-400/50 bg-cyan-500/10 shadow-[0_0_30px_rgba(34,211,238,0.15)]"
                      : "border-white/10 bg-white/5 hover:bg-white/10"
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <div className="font-bold text-slate-100">{displayModelName(r)}</div>
                        {isRec && (
                          <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-300">
                            <Crown size={10} /> REC
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-xs text-slate-400">{r.model_id}</div>
                    </div>

                    <div className="text-right text-xs">
                      <div className="text-slate-400">
                        AUC: <span className="font-bold text-cyan-300">{fmtNum(r.auc, 3)}</span>
                      </div>
                      <div className="mt-1 text-slate-400">
                        FNR: <span className="font-bold text-rose-300">{fmtPct(r.fnr)}</span>
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="lg:col-span-8">
          <RocOverlay rows={rows} roc={roc} selectedId={selectedId} rocMode={rocMode} setRocMode={setRocMode} />
        </div>
      </div>

      {/* Confusion Matrix */}
      <ConfusionMatrix row={selectedRow} />

      <div className="text-xs text-slate-400 text-center">
        Demo dashboard for academic purposes. Not a medical diagnosis.
      </div>
    </div>
  );
}
