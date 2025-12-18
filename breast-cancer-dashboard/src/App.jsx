import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";

import Overview from "./pages/Overview";
import Metrics from "./pages/Metrics";
import Explainability from "./pages/Explainability";
import Predict from "./pages/Predict";

import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";

import { AuthProvider } from "./auth/AuthContext";
import ProtectedRoute from "./auth/ProtectedRoute";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public auth pages */}
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} />

          {/* Protected dashboard layout */}
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/overview" replace />} />

            <Route
              path="/overview"
              element={
                <ProtectedRoute>
                  <Overview />
                </ProtectedRoute>
              }
            />

            <Route
              path="/metrics"
              element={
                <ProtectedRoute>
                  <Metrics />
                </ProtectedRoute>
              }
            />

            <Route
              path="/explainability"
              element={
                <ProtectedRoute>
                  <Explainability />
                </ProtectedRoute>
              }
            />

            <Route
              path="/predict"
              element={
                <ProtectedRoute>
                  <Predict />
                </ProtectedRoute>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
