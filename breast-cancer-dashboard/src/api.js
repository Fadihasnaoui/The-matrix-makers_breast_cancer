// src/api.js
import axios from "axios";

/**
 * IMPORTANT:
 * - In production (Vercel), set VITE_API_BASE_URL to your Render URL:
 *   https://the-matrix-makers-breast-cancer.onrender.com
 * - Locally it falls back to http://127.0.0.1:8000
 */
const API_BASE =
  (import.meta.env.VITE_API_BASE_URL || "").trim() || "http://127.0.0.1:8000";

// ✅ Use the SAME token key as AuthContext.jsx
const TOKEN_KEY = "token";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 20000,
});

// ✅ Attach JWT automatically
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// -------------------- Core endpoints --------------------
export async function fetchModels() {
  const res = await api.get("/models");
  return res.data;
}

export async function fetchFeatures() {
  const res = await api.get("/features");
  return res.data.features;
}

export async function fetchMetrics() {
  const res = await api.get("/metrics");
  return res.data;
}

export async function fetchFeatureRanges() {
  const res = await api.get("/feature-ranges");
  return res.data;
}

export async function predict(modelId, data) {
  const res = await api.post("/predict", { data }, { params: { model_id: modelId } });
  return res.data;
}

// -------------------- Artifacts endpoints --------------------
export async function fetchDatasetInfo() {
  const res = await api.get("/dataset-info");
  return res.data;
}

export async function fetchEvaluationTable() {
  const res = await api.get("/evaluation-table");
  return res.data;
}

// -------------------- Monitoring endpoints --------------------
export async function fetchMonitoringSummary() {
  const res = await api.get("/monitoring/summary");
  return res.data;
}

export async function fetchMonitoringPing() {
  const res = await api.get("/monitoring/ping");
  return res.data;
}

export async function fetchLivePredictions(limit = 10) {
  // ✅ uses axios baseURL + auth header automatically
  const res = await api.get("/monitoring/live", { params: { limit } });
  return res.data;
}

// -------------------- Chat --------------------
export async function chat(messages) {
  const res = await api.post("/chat", {
    model: "gpt-4o-mini",
    messages,
  });
  return res.data.reply;
}

// -------------------- Auth --------------------
export async function signup(email, password, role) {
  const res = await api.post("/auth/signup", { email, password, role });
  return res.data;
}

export async function login(email, password) {
  const res = await api.post("/auth/login", { email, password });
  return res.data;
}

// Helpers (optional)
export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}
export function getApiBase() {
  return API_BASE;
}
