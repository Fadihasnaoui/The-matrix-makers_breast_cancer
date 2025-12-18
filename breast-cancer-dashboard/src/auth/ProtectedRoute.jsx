import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth, ROLE_ACCESS } from "./AuthContext";

export default function ProtectedRoute({ children }) {
  const { isAuthed, role } = useAuth();
  const loc = useLocation();

  // not logged in -> go sign in
  if (!isAuthed) {
    return <Navigate to="/signin" replace state={{ from: loc.pathname }} />;
  }

  // logged in but role can't access this page -> kick to overview
  const allowed = ROLE_ACCESS[role] || [];
  if (!allowed.includes(loc.pathname)) {
    return <Navigate to="/overview" replace />;
  }

  return children;
}
