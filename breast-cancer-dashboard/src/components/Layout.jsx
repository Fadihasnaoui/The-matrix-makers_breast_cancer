import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Activity, BarChart3, Brain, Stethoscope } from "lucide-react";
import ChatWidget from "./ChatWidget";

// ✅ NEW
import { useAuth, ROLE_ACCESS } from "../auth/AuthContext";

const base =
  "group relative flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition-all duration-200";
const idle = "text-slate-200/80 hover:text-white hover:bg-white/8";
const active =
  "text-white bg-white/10 border border-cyan-400/30 shadow-[0_0_30px_rgba(34,211,238,0.18)]";

const ROUTE_TITLES = {
  "/overview": { title: "Overview", subtitle: "Executive summary & monitoring" },
  "/metrics": { title: "Metrics", subtitle: "Model performance & comparisons" },
  "/explainability": {
    title: "Explainability",
    subtitle: "SHAP / feature insights",
  },
  "/predict": { title: "Predict", subtitle: "Patient-level inference & confidence" },
};

export default function Layout() {
  const location = useLocation();

  // ✅ NEW
  const { role, isAuthed, logout } = useAuth();
  const allowedRoutes = ROLE_ACCESS[role] || [];

  const meta = ROUTE_TITLES[location.pathname] || {
    title: "Dashboard",
    subtitle: "Multi-model breast cancer classifier",
  };

  return (
    <div className="min-h-screen text-slate-100 bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950">
      {/* Local CSS for shimmer / glow / motion */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-120%) skewX(-12deg); opacity: 0; }
          20% { opacity: 1; }
          100% { transform: translateX(220%) skewX(-12deg); opacity: 0; }
        }
        @keyframes drift {
          0% { transform: translate3d(0,0,0) scale(1); }
          50% { transform: translate3d(12px,-10px,0) scale(1.02); }
          100% { transform: translate3d(0,0,0) scale(1); }
        }
        @keyframes pulseGlow {
          0% { filter: drop-shadow(0 0 0 rgba(34,211,238,0.0)); }
          50% { filter: drop-shadow(0 0 18px rgba(34,211,238,0.22)); }
          100% { filter: drop-shadow(0 0 0 rgba(34,211,238,0.0)); }
        }
        @keyframes borderFlow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes pageIn {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0px); }
        }
        .wow-bg {
          background: radial-gradient(circle at top left, rgba(34,211,238,0.12), transparent 45%),
                      radial-gradient(circle at top right, rgba(168,85,247,0.14), transparent 45%),
                      radial-gradient(circle at bottom, rgba(236,72,153,0.10), transparent 55%);
          animation: drift 10s ease-in-out infinite;
        }
        .noise {
          background-image: radial-gradient(rgba(255,255,255,0.9) 1px, transparent 1px);
          background-size: 18px 18px;
          opacity: 0.04;
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
        .page-enter {
          animation: pageIn 260ms ease-out;
        }
      `}</style>

      {/* Ambient layer */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute inset-0 wow-bg" />
        <div className="absolute inset-0 noise" />
      </div>

      <div className="relative mx-auto max-w-[1200px] px-4 py-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_1fr]">
          {/* Sidebar */}
          <aside className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 p-4 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent" />
            <div className="pointer-events-none absolute -inset-1 rounded-3xl bg-gradient-to-br from-cyan-400/10 via-violet-500/10 to-transparent blur-2xl" />

            <div className="relative flex items-center gap-3">
              <div
                className="relative grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-cyan-400 to-violet-500 text-white"
                style={{ animation: "pulseGlow 2.8s ease-in-out infinite" }}
              >
                <Activity size={20} />
                <div className="pointer-events-none absolute -inset-8 rounded-full bg-cyan-400/12 blur-2xl" />
              </div>

              <div>
                <div className="text-sm font-extrabold tracking-tight text-white">
                  Breast Cancer ML
                </div>
                <div className="text-xs text-slate-300">Multi-model dashboard</div>
              </div>
            </div>

            {/* ✅ role-based nav */}
            <nav className="relative mt-5 space-y-1.5">
              {allowedRoutes.includes("/overview") && (
                <NavItem to="/overview" icon={<BarChart3 size={16} />} label="Overview" />
              )}

              {allowedRoutes.includes("/metrics") && (
                <NavItem to="/metrics" icon={<Activity size={16} />} label="Metrics" />
              )}

              {allowedRoutes.includes("/explainability") && (
                <NavItem
                  to="/explainability"
                  icon={<Brain size={16} />}
                  label="Explainability"
                />
              )}

              {allowedRoutes.includes("/predict") && (
                <NavItem to="/predict" icon={<Stethoscope size={16} />} label="Predict" />
              )}
            </nav>

            <div className="relative mt-5 rounded-2xl border border-white/10 bg-white/6 p-3 text-xs text-slate-300">
              Demo only — not a medical diagnosis.
            </div>

            {/* ✅ sign out */}
            {isAuthed && (
              <button
                onClick={logout}
                className="relative mt-3 w-full rounded-2xl border border-white/10 bg-white/6 px-3 py-2 text-xs font-bold text-slate-200 hover:bg-white/10"
              >
                Sign out
              </button>
            )}
          </aside>

          {/* Main */}
          <main className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/6 p-6 shadow-2xl backdrop-blur-xl">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-cyan-400/10 via-transparent to-violet-500/10" />
            <div className="pointer-events-none absolute -inset-1 rounded-3xl blur-2xl ring-flow opacity-25" />

            {/* Page header (auto matches route) */}
            <div className="relative mb-5 flex items-start justify-between gap-3">
              <div>
                <div className="text-xl font-extrabold tracking-tight text-white">
                  {meta.title}
                </div>
                <div className="mt-1 text-sm text-slate-300">{meta.subtitle}</div>
              </div>

              {/* Tiny status pill */}
              <div className="hidden sm:flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold text-slate-200">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_14px_rgba(52,211,153,0.35)]" />
                Live UI
              </div>
            </div>

            {/* Smooth page transition */}
            <div key={location.pathname} className="relative page-enter">
              <Outlet />
            </div>
          </main>
        </div>
      </div>

      {/* ✅ always visible on all pages (fixed position inside ChatWidget) */}
      <ChatWidget />
    </div>
  );
}

function NavItem({ to, icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `${base} ${isActive ? `${active} active` : idle}`}
    >
      <span className="pointer-events-none absolute inset-0 rounded-xl opacity-0 transition-opacity duration-200 group-hover:opacity-100 bg-gradient-to-r from-cyan-400/10 via-white/5 to-violet-500/10" />

      <span className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl">
        <span
          className="absolute -left-1/2 top-0 h-full w-1/2 opacity-0 group-[.active]:opacity-100"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent)",
            animation: "shimmer 2.4s ease-in-out infinite",
          }}
        />
      </span>

      <span className="pointer-events-none absolute left-1 top-1 bottom-1 w-[3px] rounded-full bg-cyan-300/0 transition-all duration-200 group-[.active]:bg-cyan-300/70" />

      <span className="relative flex items-center gap-2">
        <span className="text-cyan-200/90 group-hover:text-cyan-200">{icon}</span>
        <span>{label}</span>
      </span>
    </NavLink>
  );
}
