import React, { useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { login as apiLogin } from "../api";
import { useAuth } from "../auth/AuthContext";
import { Activity, ShieldCheck, Sparkles } from "lucide-react";

export default function SignIn() {
  const nav = useNavigate();
  const loc = useLocation();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const pills = useMemo(
    () => [
      { icon: <ShieldCheck size={14} />, text: "Role-based access" },
      { icon: <Activity size={14} />, text: "Live monitoring" },
      { icon: <Sparkles size={14} />, text: "Explainability ready" },
    ],
    []
  );

  async function onSubmit(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      const data = await apiLogin(email, password);
      login({ token: data.access_token, role: data.role, email });
      const go = loc.state?.from || "/overview";
      nav(go, { replace: true });
    } catch (e) {
      setErr(e.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen text-slate-100 bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950">
      <AuthFX />

      <div className="relative mx-auto max-w-[1100px] px-4 py-10">
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          {/* Hero */}
          <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-cyan-400/12 via-transparent to-fuchsia-500/10" />
            <div className="pointer-events-none absolute -inset-1 rounded-3xl blur-2xl opacity-40 ring-flow" />

            <div className="relative p-7 sm:p-10">
              <div className="flex items-center gap-3">
                <div className="grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-cyan-400 to-violet-500 text-white shadow-[0_0_30px_rgba(34,211,238,0.18)]">
                  <RibbonIcon />
                </div>
                <div>
                  <div className="text-sm font-extrabold tracking-tight text-white">
                    Breast Cancer ML Dashboard
                  </div>
                  <div className="text-xs text-slate-300">
                    Patient-level prediction • SHAP/LIME explainability • Monitoring
                  </div>
                </div>
              </div>

              <h1 className="mt-8 text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
                Sign in to your clinical AI workspace
              </h1>
              <p className="mt-3 max-w-xl text-sm sm:text-base text-slate-300">
                Built for trustworthy inference and clear explanations — without turning
                your workflow into a science fair.
              </p>

              <div className="mt-6 flex flex-wrap gap-2">
                {pills.map((p) => (
                  <span
                    key={p.text}
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-bold text-slate-200"
                  >
                    <span className="text-cyan-200/90">{p.icon}</span>
                    {p.text}
                  </span>
                ))}
              </div>

              {/* Decorative “cell / network” art */}
              <div className="mt-10 grid gap-3 sm:grid-cols-3">
                <MiniCard
                  title="Risk Signal"
                  value="Benign/Malignant"
                  sub="Model output + confidence"
                />
                <MiniCard title="Why this?" value="SHAP + LIME" sub="Feature-level reasoning" />
                <MiniCard title="Ops" value="Live feed" sub="Latency + request logs" />
              </div>

              <div className="pointer-events-none absolute right-[-120px] top-[-120px] opacity-70">
                <CancerCellArt />
              </div>
            </div>
          </section>

          {/* Form */}
          <aside className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/8 via-transparent to-transparent" />
            <div className="relative p-6 sm:p-7">
              <div className="text-xl font-extrabold tracking-tight text-white">
                Welcome back
              </div>
              <div className="mt-1 text-sm text-slate-300">
                Use your account to access the dashboard
              </div>

              <form onSubmit={onSubmit} className="mt-5 space-y-3">
                <Field label="Email">
                  <input
                    className="mt-1 w-full rounded-xl bg-slate-950/60 border border-white/10 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40 focus:ring-2 focus:ring-cyan-400/10"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    type="email"
                    required
                    placeholder="you@lab.com"
                  />
                </Field>

                <Field label="Password">
                  <input
                    className="mt-1 w-full rounded-xl bg-slate-950/60 border border-white/10 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40 focus:ring-2 focus:ring-cyan-400/10"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    type="password"
                    required
                    placeholder="••••••••"
                  />
                </Field>

                {err && (
                  <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                    {err}
                  </div>
                )}

                <button
                  disabled={loading}
                  className="group relative w-full overflow-hidden rounded-xl border border-cyan-400/20 bg-white/10 px-3 py-2 text-sm font-extrabold text-white hover:bg-white/15 disabled:opacity-60"
                >
                  <span className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-r from-cyan-400/10 via-white/5 to-violet-500/10" />
                  <span className="relative">
                    {loading ? "Signing in..." : "Sign in"}
                  </span>
                </button>

                <div className="text-xs text-slate-300">
                  No account?{" "}
                  <Link className="text-cyan-200 hover:text-cyan-100" to="/signup">
                    Create one
                  </Link>
                </div>

                <div className="mt-3 rounded-2xl border border-white/10 bg-white/6 p-3 text-[11px] text-slate-300">
                  Demo only — not a medical diagnosis.
                </div>
              </form>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <label className="text-xs font-bold text-slate-300">{label}</label>
      {children}
    </div>
  );
}

function MiniCard({ title, value, sub }) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/6 p-4">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-cyan-400/10 via-transparent to-transparent" />
      <div className="relative">
        <div className="text-xs font-bold text-slate-300">{title}</div>
        <div className="mt-2 text-base font-extrabold text-white">{value}</div>
        <div className="mt-1 text-xs text-slate-300">{sub}</div>
      </div>
    </div>
  );
}

function AuthFX() {
  return (
    <>
      <style>{`
        @keyframes drift {
          0% { transform: translate3d(0,0,0) scale(1); }
          50% { transform: translate3d(16px,-10px,0) scale(1.02); }
          100% { transform: translate3d(0,0,0) scale(1); }
        }
        @keyframes borderFlow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .ring-flow {
          background: linear-gradient(90deg,
            rgba(34,211,238,0.35),
            rgba(168,85,247,0.35),
            rgba(236,72,153,0.25),
            rgba(34,211,238,0.35)
          );
          background-size: 300% 300%;
          animation: borderFlow 7s ease-in-out infinite;
        }
        .blob {
          filter: blur(55px);
          animation: drift 10s ease-in-out infinite;
          opacity: 0.35;
        }
      `}</style>

      <div className="pointer-events-none fixed inset-0">
        <div className="absolute -top-24 -left-24 h-80 w-80 rounded-full bg-cyan-400/30 blob" />
        <div className="absolute top-40 -right-28 h-96 w-96 rounded-full bg-fuchsia-500/25 blob" />
        <div className="absolute -bottom-28 left-1/3 h-96 w-96 rounded-full bg-violet-500/25 blob" />
        <div className="absolute inset-0 opacity-[0.04] [background-image:radial-gradient(rgba(255,255,255,0.9)_1px,transparent_1px)] [background-size:18px_18px]" />
      </div>
    </>
  );
}

function RibbonIcon() {
  // simple “awareness ribbon” vibe (SVG) – no external image needed
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3c-3 2.2-5 5.2-5 8.7C7 16 10 21 12 21s5-5 5-9.3C17 8.2 15 5.2 12 3Z"
        stroke="white"
        strokeWidth="1.8"
        opacity="0.95"
      />
      <path
        d="M10.1 10.7 5.4 19.6M13.9 10.7l4.7 8.9"
        stroke="white"
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity="0.9"
      />
    </svg>
  );
}

function CancerCellArt() {
  return (
    <svg width="360" height="360" viewBox="0 0 360 360" fill="none" aria-hidden="true">
      <defs>
        <radialGradient id="g1" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(180 180) rotate(90) scale(170)">
          <stop stopColor="rgba(34,211,238,0.35)" />
          <stop offset="0.6" stopColor="rgba(168,85,247,0.18)" />
          <stop offset="1" stopColor="rgba(236,72,153,0.0)" />
        </radialGradient>
      </defs>
      <circle cx="180" cy="180" r="170" fill="url(#g1)" />
      {Array.from({ length: 18 }).map((_, i) => {
        const a = (i / 18) * Math.PI * 2;
        const x = 180 + Math.cos(a) * 120;
        const y = 180 + Math.sin(a) * 120;
        return (
          <g key={i} opacity="0.55">
            <circle cx={x} cy={y} r="6" fill="rgba(255,255,255,0.22)" />
            <path
              d={`M180 180 L ${x} ${y}`}
              stroke="rgba(255,255,255,0.10)"
              strokeWidth="1.5"
            />
          </g>
        );
      })}
      <circle cx="180" cy="180" r="38" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.12)" />
      <circle cx="180" cy="180" r="16" fill="rgba(34,211,238,0.18)" />
    </svg>
  );
}
