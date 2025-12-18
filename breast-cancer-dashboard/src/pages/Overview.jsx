import { useEffect, useMemo, useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import {
  Crown,
  ShieldAlert,
  Sparkles,
  Database,
  BadgeCheck,
  BarChart3,
  Layers,
  Activity,
  Timer,
  Siren,
} from "lucide-react";

import KpiCard from "../components/KpiCard";
import LivePredictionFeed from "../components/LivePredictionFeed";
import {
  fetchDatasetInfo,
  fetchMetrics,
  fetchMonitoringSummary,
  fetchLivePredictions,
} from "../api";

const PIE_COLORS = ["#22c55e", "#ef4444"]; // benign green, malignant red

// ✅ ONLY TWO DEPLOYED MODELS + their display names
const DEPLOYED_MODELS = {
  "k-nn_(optimized)": "L2NN (Euclidean)",
  random_forest: "Random Forest",
};

function pickBestModel(metricsObj) {
  // ✅ keep ONLY deployed models
  const entries = Object.entries(metricsObj || {}).filter(
    ([modelId, m]) => DEPLOYED_MODELS[modelId] && m && !m.error
  );

  if (!entries.length) return null;

  // ✅ your original logic (AUC then Accuracy) — just filtered
  entries.sort((a, b) => {
    const ma = a[1],
      mb = b[1];
    const aucA = ma.auc ?? -1;
    const aucB = mb.auc ?? -1;
    if (aucB !== aucA) return aucB - aucA;
    const accA = ma.accuracy ?? -1;
    const accB = mb.accuracy ?? -1;
    return accB - accA;
  });

  const [bestId, best] = entries[0];

  // ✅ force displayed name
  return { id: bestId, name: DEPLOYED_MODELS[bestId], ...best };
}

export default function Overview() {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [monitoring, setMonitoring] = useState(null);
  const [liveFeed, setLiveFeed] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  // Load "static-ish" overview stuff once
  useEffect(() => {
    (async () => {
      try {
        const [ds, met, mon] = await Promise.all([
          fetchDatasetInfo(),
          fetchMetrics(),
          fetchMonitoringSummary().catch(() => null),
        ]);

        setDatasetInfo(ds || null);
        setMetrics(met || {});
        setMonitoring(mon);
        setErrorMsg("");
      } catch (e) {
        setErrorMsg(e?.message || "Failed to load overview data.");
      }
    })();
  }, []);

  const [isNarrow, setIsNarrow] = useState(false);

  useEffect(() => {
    const onResize = () => setIsNarrow(window.innerWidth < 420);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Poll live feed so page feels "alive"
  useEffect(() => {
    let alive = true;

    const tick = async () => {
      try {
        const data = await fetchLivePredictions(12);
        if (alive) setLiveFeed(Array.isArray(data) ? data : []);
      } catch {
        // don't crash page if endpoint isn't ready
      }
    };

    tick();
    const id = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const best = useMemo(() => pickBestModel(metrics), [metrics]);

  // Derived: Recall (Sensitivity) and Specificity (Selectivity)
  const recall = useMemo(() => {
    if (!best) return null;
    const tp = best.tp ?? null;
    const fn = best.fn ?? null;
    if (tp == null || fn == null) return null;
    const denom = tp + fn;
    if (!denom) return null;
    return tp / denom;
  }, [best]);

  const specificity = useMemo(() => {
    if (!best) return null;
    const tn = best.tn ?? null;
    const fp = best.fp ?? null;
    if (tn == null || fp == null) return null;
    const denom = tn + fp;
    if (!denom) return null;
    return tn / denom;
  }, [best]);

  const classPie = useMemo(() => {
    const counts = datasetInfo?.class_counts;
    if (!counts) return [];
    const benign = Number(counts["0"] ?? 0);
    const malignant = Number(counts["1"] ?? 0);
    return [
      { name: "Benign", value: benign },
      { name: "Malignant", value: malignant },
    ];
  }, [datasetInfo]);

  const total = useMemo(() => classPie.reduce((s, x) => s + (x.value || 0), 0), [classPie]);

  const fnRate = useMemo(() => {
    if (!best) return null;
    const fn = best.fn ?? null;
    const tp = best.tp ?? null;
    if (fn == null || tp == null) return null;
    const denom = fn + tp;
    if (!denom) return null;
    return fn / denom;
  }, [best]);

  const hasBackend = !errorMsg;

  const fnSeverity =
    fnRate == null ? "unknown" : fnRate === 0 ? "safe" : fnRate <= 0.02 ? "warn" : "bad";

  const fnBannerClass =
    fnSeverity === "safe"
      ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-200"
      : fnSeverity === "warn"
      ? "border-amber-400/30 bg-amber-500/10 text-amber-200"
      : fnSeverity === "bad"
      ? "border-rose-400/30 bg-rose-500/10 text-rose-200"
      : "border-white/10 bg-white/5 text-slate-200";

  const glow =
    fnSeverity === "safe"
      ? "shadow-[0_0_60px_rgba(34,197,94,0.15)]"
      : fnSeverity === "warn"
      ? "shadow-[0_0_60px_rgba(245,158,11,0.15)]"
      : fnSeverity === "bad"
      ? "shadow-[0_0_60px_rgba(244,63,94,0.18)]"
      : "shadow-[0_0_60px_rgba(59,130,246,0.15)]";

  return (
    <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
      {/* HERO */}
      <div
        className={`relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl ${glow}`}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/10 via-indigo-500/10 to-violet-500/10" />
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-violet-400/10 blur-3xl" />

        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-extrabold text-slate-200">
              <Sparkles size={14} className="text-cyan-300" /> Executive Overview
            </div>

            <h1 className="mt-3 text-3xl font-extrabold leading-tight">
              Breast Cancer Detection — Multi-Model ML Dashboard
            </h1>

            <p className="mt-2 max-w-2xl text-sm text-slate-300">
              Deployment-ready monitoring + clinical-grade model comparison + patient-level prediction.
              Built with FastAPI + React.
            </p>

            {/* Badges row */}
            <div className="mt-4 flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <BadgeCheck size={14} className={hasBackend ? "text-emerald-300" : "text-rose-300"} />
                Status: {hasBackend ? "Live" : "Offline"}
              </span>

              {/* ✅ Models count forced to 2 */}
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <Layers size={14} className="text-indigo-300" />
                Models: {Object.keys(DEPLOYED_MODELS).length}
              </span>

              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <Database size={14} className="text-cyan-300" />
                Dataset: {datasetInfo?.dataset_name || "—"}
              </span>
            </div>

            {/* Safety banner */}
            <div className={`mt-4 inline-flex items-center gap-2 rounded-2xl border px-4 py-2 text-sm ${fnBannerClass}`}>
              <ShieldAlert size={16} />
              <span className="font-bold">False Negative Rate:</span>
              <span className="font-extrabold">
                {fnRate != null ? (fnRate * 100).toFixed(2) + "%" : "—"}
              </span>
              {fnSeverity !== "safe" && fnRate != null ? (
                <span className="ml-2 inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs font-bold">
                  <Siren size={14} />
                  Attention
                </span>
              ) : null}
            </div>
          </div>

          {/* Recommended model */}
          <div className="w-full max-w-md rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur">
            <div className="text-xs font-bold text-slate-300">Recommended model</div>

            <div className="mt-1 flex items-center gap-2 text-lg font-extrabold">
              <Crown size={18} className="text-yellow-300" />
              {/* ✅ shows only the display name of the 2 models */}
              {best?.name || "—"}
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="text-slate-300">AUC</div>
                <div className="text-base font-extrabold text-cyan-300">
                  {best?.auc != null ? (best.auc * 100).toFixed(2) + "%" : "—"}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="text-slate-300">Accuracy</div>
                <div className="text-base font-extrabold text-emerald-300">
                  {best?.accuracy != null ? (best.accuracy * 100).toFixed(2) + "%" : "—"}
                </div>
              </div>
            </div>

            {/* NEW: Recall + Specificity (Selectivity) */}
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="text-slate-300">Recall (Sensitivity)</div>
                <div className="text-base font-extrabold text-indigo-200">
                  {recall != null ? (recall * 100).toFixed(2) + "%" : "—"}
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="text-slate-300">Specificity</div>
                <div className="text-base font-extrabold text-slate-100">
                  {specificity != null ? (specificity * 100).toFixed(2) + "%" : "—"}
                </div>
              </div>
            </div>

            <div className="mt-3 text-xs text-slate-300">
              Ranked by <b className="text-slate-100">Accuracy</b> then <b className="text-slate-100">AUC</b>.
            </div>

            {/* Monitoring mini row */}
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="flex items-center gap-2 text-slate-300">
                  <Activity size={14} className="text-indigo-300" /> Predictions today
                </div>
                <div className="mt-1 text-base font-extrabold text-slate-100">
                  {monitoring?.predictions_today ?? "—"}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <div className="flex items-center gap-2 text-slate-300">
                  <Timer size={14} className="text-cyan-300" /> Avg latency
                </div>
                <div className="mt-1 text-base font-extrabold text-slate-100">
                  {monitoring?.avg_latency_ms != null ? `${monitoring.avg_latency_ms} ms` : "—"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {errorMsg ? (
        <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-200">
          {errorMsg}
        </div>
      ) : null}

      {/* KPI GRID */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Samples"
          value={datasetInfo?.n_samples ?? "—"}
          hint="Total rows in dataset"
          badge="DATA"
          right={<Database className="text-cyan-300" size={20} />}
        />
        <KpiCard
          title="Features"
          value={datasetInfo?.n_features ?? "—"}
          hint="Model input dimensions"
          badge="DATA"
          right={<Layers className="text-indigo-300" size={20} />}
        />
        <KpiCard
          title="Best AUC"
          value={best?.auc != null ? (best.auc * 100).toFixed(2) + "%" : "—"}
          hint="Separation power (higher is better)"
          badge="MODEL"
          tone="good"
          right={<BarChart3 className="text-cyan-300" size={20} />}
        />
        <KpiCard
          title="Best Accuracy"
          value={best?.accuracy != null ? (best.accuracy * 100).toFixed(2) + "%" : "—"}
          hint={best?.name ? `Model: ${best.name}` : "—"}
          badge="MODEL"
          tone="good"
          right={<BadgeCheck className="text-emerald-300" size={20} />}
        />
      </div>

      {/* STORY GRID: Donut + Live Feed + Executive Summary */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Class Balance */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur lg:col-span-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-extrabold text-slate-100">Class balance</div>
              <div className="mt-1 text-xs text-slate-300">
                Distribution of benign vs malignant samples.
              </div>
            </div>
            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[11px] font-bold text-slate-200">
              {total ? `${total} total` : "—"}
            </span>
          </div>

          <div className="mt-4 h-[280px] -mx-4">
            {classPie.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={classPie}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    innerRadius={50}
                    paddingAngle={2}
                    label={({ name, value }) =>
                      total ? `${name}: ${((value / total) * 100).toFixed(0)}%` : name
                    }
                    labelLine={true}
                  >
                    {classPie.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-sm text-slate-300">
                Dataset info not available. Ensure backend returns <code>/dataset-info</code>.
              </div>
            )}
          </div>

          {classPie.length ? (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {classPie.map((x, i) => (
                <span
                  key={x.name}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-2 py-1 text-slate-200"
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: PIE_COLORS[i % PIE_COLORS.length] }}
                  />
                  {x.name}: <b className="text-white">{x.value}</b>
                </span>
              ))}
            </div>
          ) : null}
        </div>

        {/* Live Prediction Feed */}
        <div className="lg:col-span-4">
          <LivePredictionFeed items={liveFeed} />
        </div>

        {/* Executive Summary */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur lg:col-span-5">
          <div className="text-sm font-extrabold text-slate-100">Executive summary</div>

          <div className="mt-3 text-sm text-slate-300 leading-relaxed">
            {best?.name ? (
              <>
                Recommended deployment target:{" "}
                <b className="text-white">{best.name}</b>. Selected because it maximizes{" "}
                <b className="text-white">AUC</b> (class separation) and then{" "}
                <b className="text-white">Accuracy</b>. Operational KPIs (latency + live feed) are
                surfaced to support clinical readiness and production monitoring.
              </>
            ) : (
              <>
                Metrics are not loaded yet. Confirm backend endpoints{" "}
                <code>/metrics</code> and <code>/dataset-info</code>.
              </>
            )}
          </div>

          <div className="mt-5 grid grid-cols-3 gap-3">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[10px] font-bold text-slate-300 uppercase mb-2">Recommendation</div>
              <div className="text-xs font-extrabold text-white leading-tight">
                Deploy<br />{best?.name || "—"}
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[10px] font-bold text-slate-300 uppercase mb-2">Primary objective</div>
              <div className="text-xs font-extrabold text-white leading-tight">
                Minimize<br />missed<br />cancers
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[10px] font-bold text-slate-300 uppercase mb-2">Next step</div>
              <div className="text-xs font-extrabold text-white leading-tight">
                Metrics +<br />Explain-<br />ability
              </div>
            </div>
          </div>

          <div className="mt-4 text-xs text-slate-400">
            *For deployment: wire prediction logs to a database and enable role-based access (clinician/admin).
          </div>
        </div>
      </div>
    </div>
  );
}
