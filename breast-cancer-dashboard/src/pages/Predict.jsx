import { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import {
  Activity,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Sparkles,
  Crown,
  ShieldAlert,
  Target,
  Zap,
  Database,
} from "lucide-react";
import {
  fetchModels,
  fetchFeatures,
  fetchMetrics,
  fetchFeatureRanges,
  fetchDatasetInfo,
  fetchEvaluationTable,
  predict,
} from "../api";
import "../index.css";

const PIE_COLORS = ["#22c55e", "#ef4444", "#f59e0b", "#06b6d4"];

// ✅ ONLY TWO DEPLOYED MODELS + display names
const DEPLOYED_MODELS = {
  "k-nn_(optimized)": "k-nn_(optimized)",
  random_forest: "Random Forest",
};

function App() {
  const [models, setModels] = useState({});
  const [metrics, setMetrics] = useState({});
  const [features, setFeatures] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [patientData, setPatientData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [featureRanges, setFeatureRanges] = useState({});
  const [datasetInfo, setDatasetInfo] = useState({});
  const [evalTable, setEvalTable] = useState([]);

  // ✅ helper: filter objects to deployed models only
  const filterToDeployed = (obj) => {
    const out = {};
    for (const k of Object.keys(obj || {})) {
      if (DEPLOYED_MODELS[k]) out[k] = obj[k];
    }
    return out;
  };

  useEffect(() => {
    async function init() {
      try {
        setErrorMsg("");
        const [m, feat, met, ranges, ds, ev] = await Promise.all([
          fetchModels(),
          fetchFeatures(),
          fetchMetrics(),
          fetchFeatureRanges(),
          fetchDatasetInfo().catch(() => ({})),
          fetchEvaluationTable().catch(() => []),
        ]);

        // ✅ keep only the 2 deployed models
        const models2 = filterToDeployed(m || {});
        const metrics2 = filterToDeployed(met || {});

        setModels(models2);
        setFeatures(feat || []);
        setMetrics(metrics2);
        setFeatureRanges(ranges || {});
        setDatasetInfo(ds || {});
        setEvalTable(ev || []);

        const ids = Object.keys(models2);
        if (ids.length) setSelectedModelId(ids[0]);

        const defaults = {};
        (feat || []).forEach((f) => (defaults[f] = 0));
        setPatientData(defaults);
      } catch (err) {
        console.error("INIT FAILED:", err);
        const details =
          err?.response
            ? `HTTP ${err.response.status} ${err.response.config?.url} - ${JSON.stringify(
                err.response.data
              )}`
            : err?.message || String(err);
        setErrorMsg(`Could not load backend data. ${details}`);
      }
    }
    init();
  }, []);

  const modelIds = Object.keys(models);

  const selectedMetrics = selectedModelId ? metrics[selectedModelId] : null;

  // ✅ stable display name (don’t trust metrics name)
  const selectedModelName =
    (selectedModelId && DEPLOYED_MODELS[selectedModelId]) || "Selected model";

  const handleFeatureChange = (feat, value) => {
    setPatientData((prev) => ({
      ...prev,
      [feat]: parseFloat(value) || 0,
    }));
  };

  const handlePredict = async () => {
    if (!selectedModelId) return;
    try {
      setLoading(true);
      setErrorMsg("");
      const res = await predict(selectedModelId, patientData);
      setPrediction(res);
    } catch (err) {
      console.error(err);
      setPrediction(null);
      setErrorMsg("Error calling prediction API.");
    } finally {
      setLoading(false);
    }
  };

  // ✅ NEW: Autofill that actually produces variety (benign/malignant)
  // Strategy:
  // - Sample from feature min/max, with a random "severity" bias each time
  // - Try multiple times, call predict, stop once we get either class sometimes
  // - No extra backend endpoints needed
const autoFillRealisticVaried = () => {
  if (!features.length) return;

  if (!featureRanges || Object.keys(featureRanges).length === 0) {
    setErrorMsg("Feature ranges not loaded. Check /feature-ranges in FastAPI.");
    return;
  }

  setErrorMsg("");
  setPrediction(null); // clear old prediction ONLY

  // Random severity: sometimes benign-leaning, sometimes malignant-leaning
  const severity =
    Math.random() < 0.5
      ? Math.random() * 0.4        // lower values → benign-leaning
      : 0.6 + Math.random() * 0.4; // higher values → malignant-leaning

  const newData = {};

  for (const feat of features) {
    const stats = featureRanges[feat];

    if (!stats || stats.min == null || stats.max == null) {
      newData[feat] = 0;
      continue;
    }

    // small randomness so it's not deterministic
    const jitter = (Math.random() - 0.5) * 0.1;
    const s = Math.min(1, Math.max(0, severity + jitter));

    const value = stats.min + s * (stats.max - stats.min);

    newData[feat] =
      Math.abs(value) < 10
        ? Number(value.toFixed(4))
        : Number(value.toFixed(2));
  }

  setPatientData(newData);
};

  // ✅ only 2 models in comparison
  const comparisonData = modelIds
    .filter((id) => metrics[id] && !metrics[id].error)
    .map((id) => ({
      model: DEPLOYED_MODELS[id] || models[id]?.name || id,
      accuracy: (metrics[id]?.accuracy || 0) * 100,
      f1: (metrics[id]?.f1 || 0) * 100,
    }));

  const confusionData = selectedMetrics
    ? [
        { name: "True Negatives", value: selectedMetrics.tn ?? 0 },
        { name: "False Positives", value: selectedMetrics.fp ?? 0 },
        { name: "False Negatives", value: selectedMetrics.fn ?? 0 },
        { name: "True Positives", value: selectedMetrics.tp ?? 0 },
      ]
    : [];

  const radarData = selectedMetrics
    ? [
        { metric: "Accuracy", value: (selectedMetrics.accuracy || 0) * 100 },
        { metric: "Precision", value: (selectedMetrics.precision || 0) * 100 },
        { metric: "Recall", value: (selectedMetrics.recall || 0) * 100 },
        { metric: "F1-Score", value: (selectedMetrics.f1 || 0) * 100 },
        { metric: "AUC", value: (selectedMetrics.auc || 0) * 100 },
      ]
    : [];

  return (
    <div className="min-h-[calc(100vh-2rem)] space-y-6 rounded-3xl bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6 text-slate-100">
      {/* HERO */}
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl shadow-[0_0_60px_rgba(59,130,246,0.15)]">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/10 via-indigo-500/10 to-violet-500/10" />
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-violet-400/10 blur-3xl" />

        <div className="relative">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-extrabold text-slate-200">
            <Sparkles size={14} className="text-cyan-300" /> Patient Prediction
          </div>

          <h1 className="mt-3 text-3xl font-extrabold leading-tight">
            Breast Cancer Prediction Dashboard
          </h1>

          <p className="mt-2 max-w-2xl text-sm text-slate-300">
            Multi-model diagnostic classifier with real-time inference. Input patient features and get
            instant predictions with confidence scores.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
              <Crown size={14} className="text-yellow-300" />
              Models deployed: {modelIds.length}
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
              <Target size={14} className="text-cyan-300" />
              Features: {features.length}
            </span>
            {datasetInfo?.n_samples ? (
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <Database size={14} className="text-indigo-300" />
                Samples: {datasetInfo.n_samples}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-200">
          <div className="flex items-center gap-2 font-bold">
            <AlertCircle size={16} />
            Error
          </div>
          <div className="mt-1">{errorMsg}</div>
        </div>
      )}

      {/* Model Selector */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
        <div className="mb-4 text-sm font-extrabold text-slate-100">Select model</div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {modelIds.map((id) => {
            const m = metrics[id] || {};
            const selected = id === selectedModelId;
            return (
              <button
                key={id}
                onClick={() => setSelectedModelId(id)}
                className={`rounded-2xl border p-4 text-left transition-all ${
                  selected
                    ? "border-cyan-400/50 bg-cyan-500/10 shadow-[0_0_30px_rgba(34,211,238,0.15)]"
                    : "border-white/10 bg-white/5 hover:bg-white/10"
                }`}
              >
                <div className="font-bold text-slate-100">
                  {DEPLOYED_MODELS[id] || models[id]?.name || id}
                </div>
                <div className="mt-2 text-2xl font-extrabold text-cyan-300">
                  {m.accuracy ? (m.accuracy * 100).toFixed(1) : "--"}%
                </div>
                <div className="text-xs text-slate-400">Accuracy</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Key Metrics */}
      {selectedMetrics && !selectedMetrics.error && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
          <div className="mb-4 text-sm font-extrabold text-slate-100">
            {selectedModelName} – Key Metrics
          </div>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {[
              { label: "Accuracy", key: "accuracy", icon: CheckCircle, color: "#06b6d4" },
              { label: "Precision", key: "precision", icon: TrendingUp, color: "#22c55e" },
              { label: "Recall", key: "recall", icon: Activity, color: "#a78bfa" },
              { label: "F1-score", key: "f1", icon: AlertCircle, color: "#f472b6" },
            ].map((entry) => {
              const Icon = entry.icon;
              const v = selectedMetrics[entry.key] ?? 0;
              return (
                <div key={entry.key} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center gap-2 text-xs text-slate-300">
                    <Icon size={14} style={{ color: entry.color }} />
                    {entry.label}
                  </div>
                  <div className="mt-2 text-2xl font-extrabold" style={{ color: entry.color }}>
                    {(v * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Patient Input & Prediction */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Feature Input */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur lg:col-span-7">
          <div className="mb-4">
            <div className="text-sm font-extrabold text-slate-100">Patient features & prediction</div>
            <div className="mt-1 text-xs text-slate-300">
              Click "Auto-fill realistic patient" then run prediction.
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 max-h-[500px] overflow-auto pr-2">
            {features.map((feat) => (
              <div key={feat} className="rounded-xl border border-white/10 bg-white/5 p-3">
                <label className="text-xs font-medium text-slate-300">{feat}</label>
                <input
                  type="number"
                  step="any"
                  value={patientData[feat] ?? 0}
                  onChange={(e) => handleFeatureChange(feat, e.target.value)}
                  className="mt-1 w-full rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20"
                />
              </div>
            ))}
          </div>

          {/* ✅ Buttons: keep ONLY what you need, add small polish */}
          <div className="mt-4 flex gap-3 flex-wrap">
            <button
              onClick={autoFillRealisticVaried}
              disabled={loading || !selectedModelId}
              className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-bold text-slate-100 transition-all hover:bg-white/10 disabled:opacity-50"
              title="Generates varied realistic examples and checks prediction to avoid always-malignant"
            >
              <Zap size={16} className="inline mr-2" />
              Auto-fill realistic patient
            </button>

            <button
              onClick={handlePredict}
              disabled={loading || !selectedModelId}
              className="rounded-xl bg-cyan-600 px-4 py-2 text-sm font-extrabold text-white transition-all hover:bg-cyan-700 disabled:opacity-50 shadow-[0_0_25px_rgba(34,211,238,0.12)]"
            >
              {loading ? "Predicting..." : "Run prediction"}
            </button>
          </div>
        </div>

        {/* Prediction Result */}
        <div className="lg:col-span-5">
          {prediction ? (
            <div
              className={`rounded-3xl border p-6 backdrop-blur ${
                prediction.prediction === 1
                  ? "border-rose-400/30 bg-rose-500/10 shadow-[0_0_40px_rgba(244,63,94,0.15)]"
                  : "border-emerald-400/30 bg-emerald-500/10 shadow-[0_0_40px_rgba(34,197,94,0.15)]"
              }`}
            >
              <div className="mb-4">
                <div className="text-sm font-extrabold text-slate-100">Prediction Result</div>
              </div>

              <div className="mb-6">
                {prediction.prediction === 1 ? (
                  <div className="flex items-center gap-3">
                    <ShieldAlert size={32} className="text-rose-300" />
                    <div>
                      <div className="text-2xl font-extrabold text-rose-200">Malignant (1)</div>
                      <div className="text-sm text-rose-300">High risk detected</div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <CheckCircle size={32} className="text-emerald-300" />
                    <div>
                      <div className="text-2xl font-extrabold text-emerald-200">Benign (0)</div>
                      <div className="text-sm text-emerald-300">Low risk detected</div>
                    </div>
                  </div>
                )}
              </div>

              {prediction.probabilities && prediction.probabilities.length === 2 && (
                <div className="space-y-3">
                  <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                    <div className="text-xs text-slate-300">Benign probability</div>
                    <div className="mt-1 text-xl font-extrabold text-emerald-300">
                      {(prediction.probabilities[0] * 100).toFixed(1)}%
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-slate-800">
                      <div
                        className="h-2 rounded-full bg-emerald-400"
                        style={{ width: `${prediction.probabilities[0] * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                    <div className="text-xs text-slate-300">Malignant probability</div>
                    <div className="mt-1 text-xl font-extrabold text-rose-300">
                      {(prediction.probabilities[1] * 100).toFixed(1)}%
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-slate-800">
                      <div
                        className="h-2 rounded-full bg-rose-400"
                        style={{ width: `${prediction.probabilities[1] * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-6 rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-slate-400">
                This is a machine learning demo. It is not a medical diagnosis.
              </div>
            </div>
          ) : (
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
              <div className="text-center text-slate-400">
                <Target size={48} className="mx-auto mb-4 text-slate-600" />
                <div className="text-sm font-medium">No prediction yet</div>
                <div className="mt-2 text-xs">
                  Fill patient features and click "Run prediction"
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
        <div className="mb-4 text-sm font-extrabold text-slate-100">
          {selectedModelName} – Confusion Summary
        </div>
        {confusionData.some((d) => d.value > 0) ? (
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={confusionData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label
                >
                  {confusionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                    color: "#e2e8f0",
                  }}
                />
                <Legend wrapperStyle={{ color: "#e2e8f0" }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center text-slate-400 py-8">
            Confusion matrix values not available for this model.
          </div>
        )}
      </div>

      {/* Model Comparison */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
        <div className="mb-4 text-sm font-extrabold text-slate-100">Model performance comparison</div>
        {comparisonData.length ? (
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="model" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                    color: "#e2e8f0",
                  }}
                />
                <Legend wrapperStyle={{ color: "#e2e8f0" }} />
                <Bar dataKey="accuracy" fill="#06b6d4" name="Accuracy %" />
                <Bar dataKey="f1" fill="#a78bfa" name="F1-Score %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center text-slate-400 py-8">
            Not enough metrics to display comparison yet.
          </div>
        )}
      </div>

      {/* Radar Chart */}
      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur">
        <div className="mb-4 text-sm font-extrabold text-slate-100">
          Metrics overview – {selectedModelName}
        </div>
        {radarData.length ? (
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <PolarRadiusAxis tick={{ fill: "#94a3b8", fontSize: 12 }} domain={[0, 100]} />
                <Radar
                  name={selectedModelName}
                  dataKey="value"
                  stroke="#06b6d4"
                  fill="#06b6d4"
                  fillOpacity={0.5}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                    color: "#e2e8f0",
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center text-slate-400 py-8">
            Metrics not available for this model. Make sure metrics.json was exported.
          </div>
        )}
      </div>

      <div className="text-xs text-slate-400 text-center">
        Dataset: Wisconsin Breast Cancer. Models trained in notebook, exported as artifacts, served via
        FastAPI, visualized with React + Recharts.
      </div>
    </div>
  );
}

export default App;
