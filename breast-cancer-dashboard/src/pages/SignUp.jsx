import React, { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { signup as apiSignup } from "../api";
import { useAuth } from "../auth/AuthContext";
import { Brain, Stethoscope, BarChart3, ShieldCheck } from "lucide-react";

export default function SignUp() {
  const nav = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("scientist");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const rolePreview = useMemo(() => {
    const common = [
      { icon: <BarChart3 size={16} />, label: "Overview" },
      { icon: <Stethoscope size={16} />, label: "Predict" },
      { icon: <Brain size={16} />, label: "Explainability" },
    ];
    const dsOnly = [{ icon: <ShieldCheck size={16} />, label: "Metrics (DS only)" }];

    return role === "data_scientist" ? [...common, ...dsOnly] : common;
  }, [role]);

  async function onSubmit(e) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      const data = await apiSignup(email, password, role);
      login({ token: data.access_token, role: data.role, email });
      nav("/overview", { replace: true });
    } catch (e) {
      setErr(e.message || "Signup failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen text-slate-100 bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950">
      <AuthFX />

      <div className="relative mx-auto max-w-[1100px] px-4 py-10">
        <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
          {/* Form */}
          <aside className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/8 via-transparent to-transparent" />
            <div className="relative p-6 sm:p-7">
              <div className="text-xl font-extrabold tracking-tight text-white">
                Create account
              </div>
              <div className="mt-1 text-sm text-slate-300">
                Pick a role — we’ll automatically unlock the right pages.
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
                    placeholder="Minimum 6 characters"
                  />
                  <div className="mt-1 text-[11px] text-slate-400">
                    Tip: keep it under 72 bytes (bcrypt limit).
                  </div>
                </Field>

                <Field label="Role">
                  <select
                    className="mt-1 w-full rounded-xl bg-slate-950/60 border border-white/10 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40 focus:ring-2 focus:ring-cyan-400/10"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                  >
                    <option value="scientist">Scientist</option>
                    <option value="data_scientist">Data Scientist</option>
                  </select>
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
                    {loading ? "Creating..." : "Sign up"}
                  </span>
                </button>

                <div className="text-xs text-slate-300">
                  Already have an account?{" "}
                  <Link className="text-cyan-200 hover:text-cyan-100" to="/signin">
                    Sign in
                  </Link>
                </div>
              </form>
            </div>
          </aside>

          {/* Hero */}
          <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-fuchsia-500/10 via-transparent to-cyan-400/12" />
            <div className="pointer-events-none absolute -inset-1 rounded-3xl blur-2xl opacity-40 ring-flow" />

            <div className="relative p-7 sm:p-10">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-extrabold text-slate-200">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_14px_rgba(52,211,153,0.35)]" />
                Secure sign-in with roles
              </div>

              <h1 className="mt-6 text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
                Build confidence in every prediction
              </h1>
              <p className="mt-3 max-w-xl text-sm sm:text-base text-slate-300">
                This dashboard highlights risk signals, explains them with SHAP/LIME,
                and keeps performance insights close — without clutter.
              </p>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <FeatureCard
                  title="Explainability"
                  desc="Understand what features drive each case."
                  icon={<Brain size={18} />}
                />
                <FeatureCard
                  title="Clinical workflow"
                  desc="Predict page tailored for patient-level inference."
                  icon={<Stethoscope size={18} />}
                />
              </div>

              <div className="mt-8 rounded-3xl border border-white/10 bg-white/6 p-5">
                <div className="text-sm font-extrabold text-white">
                  Pages you’ll unlock
                </div>
                <div className="mt-1 text-xs text-slate-300">
                  Based on your selected role:{" "}
                  <span className="text-cyan-200 font-bold">
                    {role === "data_scientist" ? "Data Scientist" : "Scientist"}
                  </span>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {rolePreview.map((p) => (
                    <span
                      key={p.label}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-950/35 px-3 py-1 text-xs font-bold text-slate-200"
                    >
                      <span className="text-cyan-200/90">{p.icon}</span>
                      {p.label}
                    </span>
                  ))}
                </div>
              </div>

              <div className="pointer-events-none absolute -left-16 -bottom-16 opacity-80">
                <CancerCellArt />
              </div>
            </div>
          </section>
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

function FeatureCard({ title, desc, icon }) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/6 p-4">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-cyan-400/10 via-transparent to-transparent" />
      <div className="relative flex items-start gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 text-white">
          {icon}
        </div>
        <div>
          <div className="text-sm font-extrabold text-white">{title}</div>
          <div className="mt-1 text-xs text-slate-300">{desc}</div>
        </div>
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
        <div className="absolute -top-24 -left-24 h-80 w-80 rounded-full bg-fuchsia-500/25 blob" />
        <div className="absolute top-40 -right-28 h-96 w-96 rounded-full bg-cyan-400/25 blob" />
        <div className="absolute -bottom-28 left-1/3 h-96 w-96 rounded-full bg-violet-500/25 blob" />
        <div className="absolute inset-0 opacity-[0.04] [background-image:radial-gradient(rgba(255,255,255,0.9)_1px,transparent_1px)] [background-size:18px_18px]" />
      </div>
    </>
  );
}

function CancerCellArt() {
  return (
    <svg width="360" height="360" viewBox="0 0 360 360" fill="none" aria-hidden="true">
      <defs>
        <radialGradient id="g1" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(180 180) rotate(90) scale(170)">
          <stop stopColor="rgba(236,72,153,0.30)" />
          <stop offset="0.6" stopColor="rgba(168,85,247,0.18)" />
          <stop offset="1" stopColor="rgba(34,211,238,0.0)" />
        </radialGradient>
      </defs>
      <circle cx="180" cy="180" r="170" fill="url(#g1)" />
      {Array.from({ length: 16 }).map((_, i) => {
        const a = (i / 16) * Math.PI * 2;
        const x = 180 + Math.cos(a) * 115;
        const y = 180 + Math.sin(a) * 115;
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
      <circle cx="180" cy="180" r="40" fill="rgba(255,255,255,0.07)" stroke="rgba(255,255,255,0.12)" />
      <circle cx="180" cy="180" r="16" fill="rgba(236,72,153,0.16)" />
    </svg>
  );
}
