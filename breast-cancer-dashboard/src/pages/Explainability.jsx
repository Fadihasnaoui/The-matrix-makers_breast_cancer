import { useEffect, useState, useMemo } from "react";
import { api } from "../api"; // ✅ use axios instance everywhere

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Sparkles,
  Brain,
  Target,
  Info,
  AlertCircle,
  TrendingUp,
  Layers,
  Crown,
  Eye,
  Zap,
} from "lucide-react";

// ✅ Only deployed models
const ALLOWED_MODELS = ["k-nn_(optimized)", "random_forest"];

// ✅ Per-model capabilities
const modelCapabilities = {
  "k-nn_(optimized)": {
    shap: true,
    lime: true,
    featureImportance: false,
  },
  random_forest: {
    shap: true,
    lime: true,
    featureImportance: true,
  },
};

function ExplainabilityPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [models, setModels] = useState({});
  const [selectedModelId, setSelectedModelId] = useState(null);

  const [shapSummary, setShapSummary] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [limeExplanation, setLimeExplanation] = useState(null);

  const [sampleIndex, setSampleIndex] = useState(0);

  // Load models on mount
  useEffect(() => {
    async function loadModels() {
      try {
        setLoading(true);
        setError(null);

        // ✅ use api instance (baseURL + auth header)
        const res = await api.get("/models");
        const data = res.data;

        // ✅ keep only allowed models from backend registry
        const filtered = Object.fromEntries(
          Object.entries(data).filter(([id]) => ALLOWED_MODELS.includes(id))
        );

        setModels(filtered);

        const ids = Object.keys(filtered);
        if (ids.length > 0) setSelectedModelId(ids[0]);
      } catch (err) {
        setError(err?.response?.data?.detail || err?.message || String(err));
      } finally {
        setLoading(false);
      }
    }
    loadModels();
  }, []);

  const modelIds = useMemo(() => Object.keys(models), [models]);

  const selectedModelName = useMemo(() => {
    if (!selectedModelId) return "—";
    return models[selectedModelId]?.name || selectedModelId;
  }, [models, selectedModelId]);

  const caps = selectedModelId ? modelCapabilities[selectedModelId] : null;

  // Load explainability data when model changes
  useEffect(() => {
    if (!selectedModelId) return;

    async function loadExplainability() {
      try {
        setLoading(true);
        setError(null);

        // reset old data
        setShapSummary(null);
        setFeatureImportance(null);
        setLimeExplanation(null);

        // SHAP
        if (caps?.shap) {
          const shapRes = await api.get(`/shap-summary/${selectedModelId}`);
          setShapSummary(shapRes.data);
        }

        // Feature importance (RF only)
        if (caps?.featureImportance) {
          const impRes = await api.get(`/feature-importance/${selectedModelId}`);
          setFeatureImportance(impRes.data);
        }

        // LIME
        if (caps?.lime) {
          const limeRes = await api.get(`/lime-explanation/${selectedModelId}`, {
            params: { sample_index: sampleIndex },
          });
          setLimeExplanation(limeRes.data);
        }
      } catch (err) {
        setError(err?.response?.data?.detail || err?.message || String(err));
      } finally {
        setLoading(false);
      }
    }

    loadExplainability();
  }, [selectedModelId, sampleIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Prepare feature importance data for chart
  const featureImportanceData = useMemo(() => {
    if (!featureImportance?.features?.length || !featureImportance?.importances?.length) return [];
    return featureImportance.features
      .map((feature, idx) => ({
        feature: feature.length > 20 ? feature.substring(0, 20) + "..." : feature,
        importance: (featureImportance.importances[idx] ?? 0) * 100,
        fullName: feature,
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 15);
  }, [featureImportance]);

  // Prepare LIME data for chart
  const limeData = useMemo(() => {
    if (!limeExplanation?.feature_weights) return [];
    return Object.entries(limeExplanation.feature_weights)
      .map(([feature, weight]) => ({
        feature: feature.length > 20 ? feature.substring(0, 20) + "..." : feature,
        weight: Number(weight),
        fullName: feature,
      }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 15);
  }, [limeExplanation]);

  const hasLime = caps?.lime && limeData.length > 0;
  const hasShap =
    caps?.shap && (shapSummary?.plot_url || shapSummary?.plot_file || shapSummary?.message);
  const showFeatureImportance = caps?.featureImportance; // RF only

  if (loading && !selectedModelId) {
    return (
      <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
        <div className="text-2xl font-extrabold">Loading explainability data...</div>
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
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-extrabold text-slate-200">
                <Sparkles size={14} className="text-cyan-300" /> Model Explainability
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-violet-400/30 bg-violet-500/10 px-3 py-1 text-xs font-bold text-violet-200">
                <Brain size={14} /> Interpretability
              </span>
            </div>

            <h1 className="mt-3 text-3xl font-extrabold leading-tight">
              Model Interpretation & Feature Analysis
            </h1>

            <p className="mt-2 max-w-2xl text-sm text-slate-300">
              Understand model decisions through <b className="text-white">SHAP</b> (global feature impact),{" "}
              <b className="text-white">Feature Importance</b> (only for models that support it), and{" "}
              <b className="text-white">LIME</b> (local instance explanations).
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <Eye size={14} className="text-cyan-300" />
                Current model: {selectedModelName}
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <Layers size={14} className="text-indigo-300" />
                {modelIds.length} models deployed
              </span>
            </div>
          </div>

          {/* Model Selector */}
          <div className="w-full max-w-xs">
            <div className="text-xs font-bold text-slate-300 mb-2">Select model to explain</div>
            <select
              value={selectedModelId || ""}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-100 outline-none backdrop-blur focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20"
            >
              {modelIds.map((id) => (
                <option key={id} value={id} className="bg-slate-900">
                  {models[id]?.name || id}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-200">
          <div className="flex items-center gap-2 font-bold">
            <AlertCircle size={16} />
            Error loading explainability data
          </div>
          <div className="mt-1">{error}</div>
        </div>
      )}

      {/* Explainability Method Cards */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-3xl border border-cyan-400/30 bg-cyan-500/10 p-6 backdrop-blur">
          <div className="flex items-center gap-3 mb-3">
            <div className="rounded-xl bg-cyan-500/20 p-2">
              <Target size={24} className="text-cyan-300" />
            </div>
            <div>
              <div className="text-base font-extrabold text-cyan-100">SHAP Values</div>
              <div className="text-xs text-cyan-300">Global explanation</div>
            </div>
          </div>
          <p className="text-sm text-cyan-200 leading-relaxed">
            SHAP shows how features contribute to model predictions across many samples.
          </p>
        </div>

        <div className="rounded-3xl border border-violet-400/30 bg-violet-500/10 p-6 backdrop-blur">
          <div className="flex items-center gap-3 mb-3">
            <div className="rounded-xl bg-violet-500/20 p-2">
              <TrendingUp size={24} className="text-violet-300" />
            </div>
            <div>
              <div className="text-base font-extrabold text-violet-100">Feature Importance</div>
              <div className="text-xs text-violet-300">Model-specific weights</div>
            </div>
          </div>
          <p className="text-sm text-violet-200 leading-relaxed">
            Available only for models that expose intrinsic importance (e.g. Random Forest).
          </p>
        </div>

        <div className="rounded-3xl border border-emerald-400/30 bg-emerald-500/10 p-6 backdrop-blur">
          <div className="flex items-center gap-3 mb-3">
            <div className="rounded-xl bg-emerald-500/20 p-2">
              <Zap size={24} className="text-emerald-300" />
            </div>
            <div>
              <div className="text-base font-extrabold text-emerald-100">LIME</div>
              <div className="text-xs text-emerald-300">Local instance explanation</div>
            </div>
          </div>
          <p className="text-sm text-emerald-200 leading-relaxed">
            LIME explains a single prediction by showing which features push it one way or the other.
          </p>
        </div>
      </div>

      {/* ✅ Feature Importance Chart (ONLY for Random Forest) */}
      {showFeatureImportance && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
          <div className="mb-6">
            <div className="flex items-center gap-2">
              <Crown size={20} className="text-violet-300" />
              <div className="text-base font-extrabold text-slate-100">Feature Importance</div>
            </div>
            <div className="mt-1 text-sm text-slate-300">
              Top 15 features ranked by model-specific importance weights
            </div>
          </div>

          {featureImportanceData.length > 0 ? (
            <div className="h-[500px] rounded-2xl border border-white/10 bg-slate-950/50 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportanceData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    type="number"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    label={{
                      value: "Importance (%)",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#94a3b8",
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    width={150}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(15, 23, 42, 0.95)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      color: "#e2e8f0",
                    }}
                    formatter={(value, name, props) => [
                      `${Number(value).toFixed(2)}%`,
                      props?.payload?.fullName ?? "—",
                    ]}
                  />
                  <Bar dataKey="importance" fill="#a78bfa" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-12 text-center">
              <TrendingUp size={48} className="mx-auto mb-4 text-slate-600" />
              <div className="text-sm font-medium text-slate-400">No feature importance available</div>
              <div className="mt-2 text-xs text-slate-500">
                Check /feature-importance/{selectedModelId}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SHAP Summary Plot */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <div className="mb-6">
          <div className="flex items-center gap-2">
            <Target size={20} className="text-cyan-300" />
            <div className="text-base font-extrabold text-slate-100">SHAP Summary Plot</div>
          </div>
          <div className="mt-1 text-sm text-slate-300">
            Global feature impact across predictions • Red = high feature value, Blue = low
          </div>
        </div>

        {hasShap && shapSummary?.plot_url ? (
          <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-6 overflow-hidden">
            <img
              src={shapSummary.plot_url}
              alt="SHAP Summary Plot"
              className="w-full h-auto rounded-xl"
            />
          </div>
        ) : shapSummary?.message ? (
          <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-12 text-center">
            <Info size={48} className="mx-auto mb-4 text-slate-600" />
            <div className="text-sm font-medium text-slate-400">{shapSummary.message}</div>
          </div>
        ) : (
          <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-12 text-center">
            <Brain size={48} className="mx-auto mb-4 text-slate-600" />
            <div className="text-sm font-medium text-slate-400">No SHAP data available</div>
            <div className="mt-2 text-xs text-slate-500">
              Check /shap-summary/{selectedModelId}
            </div>
          </div>
        )}

        {shapSummary?.description && (
          <div className="mt-4 rounded-xl border border-cyan-400/30 bg-cyan-500/10 p-4">
            <div className="text-xs font-bold text-cyan-300 uppercase mb-2">Interpretation Guide</div>
            <p className="text-sm text-cyan-200 leading-relaxed">{shapSummary.description}</p>
          </div>
        )}
      </div>

      {/* LIME Explanation */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Zap size={20} className="text-emerald-300" />
              <div className="text-base font-extrabold text-slate-100">LIME Local Explanation</div>
            </div>
            <div className="mt-1 text-sm text-slate-300">
              Feature contributions for individual prediction • Sample #{sampleIndex}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setSampleIndex(Math.max(0, sampleIndex - 1))}
              disabled={sampleIndex === 0}
              className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-100 transition-all hover:bg-white/10 disabled:opacity-50"
            >
              ← Prev
            </button>
            <input
              type="number"
              value={sampleIndex}
              onChange={(e) => setSampleIndex(parseInt(e.target.value, 10) || 0)}
              className="w-20 rounded-lg border border-white/10 bg-slate-900/50 px-2 py-1 text-center text-xs text-slate-100 outline-none focus:border-cyan-400/50"
            />
            <button
              onClick={() => setSampleIndex(sampleIndex + 1)}
              className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-100 transition-all hover:bg-white/10"
            >
              Next →
            </button>
          </div>
        </div>

        {hasLime ? (
          <>
            {limeExplanation?.prediction !== undefined && (
              <div className="mb-6 grid grid-cols-2 gap-4">
                <div
                  className={`rounded-2xl border p-4 ${
                    limeExplanation.prediction === 1
                      ? "border-rose-400/30 bg-rose-500/10"
                      : "border-emerald-400/30 bg-emerald-500/10"
                  }`}
                >
                  <div className="text-xs text-slate-300">Predicted Class</div>
                  <div className="mt-2 text-2xl font-extrabold">
                    {limeExplanation.prediction === 1 ? (
                      <span className="text-rose-300">Malignant (1)</span>
                    ) : (
                      <span className="text-emerald-300">Benign (0)</span>
                    )}
                  </div>
                </div>

                {limeExplanation?.probability !== undefined && (
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <div className="text-xs text-slate-300">Prediction Confidence</div>
                    <div className="mt-2 text-2xl font-extrabold text-cyan-300">
                      {(Number(limeExplanation.probability) * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="h-[500px] rounded-2xl border border-white/10 bg-slate-950/50 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={limeData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    type="number"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    label={{
                      value: "Feature Weight",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#94a3b8",
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    width={150}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(15, 23, 42, 0.95)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      color: "#e2e8f0",
                    }}
                    formatter={(value, name, props) => [
                      Number(value).toFixed(4),
                      props?.payload?.fullName ?? "—",
                    ]}
                  />

                  {/* ✅ GREEN/RED bars */}
                  <Bar
                    dataKey="weight"
                    radius={[0, 8, 8, 0]}
                    shape={(props) => {
                      const { x, y, width, height, value } = props;
                      const barWidth = Math.abs(width);
                      const barX = value >= 0 ? x : x - barWidth;
                      const fillColor = value >= 0 ? "#22c55e" : "#ef4444";

                      return (
                        <rect
                          x={barX}
                          y={y}
                          width={barWidth}
                          height={height}
                          fill={fillColor}
                          rx={8}
                        />
                      );
                    }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-4">
              <div className="text-xs font-bold text-emerald-300 uppercase mb-2">How to read this</div>
              <p className="text-sm text-emerald-200 leading-relaxed">
                <span className="font-bold text-emerald-300">Green bars</span> push the prediction toward{" "}
                <b>Malignant (1)</b>.{" "}
                <span className="font-bold text-rose-300">Red bars</span> push toward{" "}
                <b>Benign (0)</b>. Longer bars indicate stronger influence on this specific prediction.
              </p>
            </div>
          </>
        ) : (
          <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-12 text-center">
            <Zap size={48} className="mx-auto mb-4 text-slate-600" />
            <div className="text-sm font-medium text-slate-400">No LIME explanation available</div>
            <div className="mt-2 text-xs text-slate-500">
              Check /lime-explanation/{selectedModelId}?sample_index={sampleIndex}
            </div>
          </div>
        )}
      </div>

      {/* Clinical Interpretation Guide */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <div className="mb-4">
          <div className="flex items-center gap-2">
            <Info size={20} className="text-indigo-300" />
            <div className="text-base font-extrabold text-slate-100">Clinical Interpretation Guide</div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="text-xs font-bold text-slate-300 uppercase mb-2">Global vs Local</div>
            <p className="text-sm text-slate-400 leading-relaxed">
              SHAP shows global behavior. LIME explains individual cases.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="text-xs font-bold text-slate-300 uppercase mb-2">Trust & Validation</div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Use explainability to validate models focus on clinically meaningful features.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="text-xs font-bold text-slate-300 uppercase mb-2">Regulatory Compliance</div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Explainable AI helps justify predictions, but it doesn’t replace clinical judgment.
            </p>
          </div>
        </div>
      </div>

      <div className="text-xs text-slate-400 text-center">
        Demo dashboard for academic purposes. Explainability helps understand model behavior but doesn’t replace diagnosis.
      </div>
    </div>
  );
}

export default ExplainabilityPage;
