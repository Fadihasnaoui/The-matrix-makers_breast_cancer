import React, { createContext, useContext, useMemo, useState } from "react";

const AuthCtx = createContext(null);

export const ROLE_ACCESS = {
  scientist: ["/overview", "/predict", "/explainability"],
  data_scientist: ["/overview", "/predict", "/explainability", "/metrics"],
};

function readStored() {
  return {
    token: localStorage.getItem("token"),
    role: localStorage.getItem("role"),
    email: localStorage.getItem("email"),
  };
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(readStored());

  const value = useMemo(() => {
    return {
      token: auth.token,
      role: auth.role,
      email: auth.email,
      isAuthed: Boolean(auth.token),

      login: ({ token, role, email }) => {
        localStorage.setItem("token", token);
        localStorage.setItem("role", role);
        localStorage.setItem("email", email);
        setAuth({ token, role, email });
      },

      logout: () => {
        localStorage.removeItem("token");
        localStorage.removeItem("role");
        localStorage.removeItem("email");
        setAuth({ token: null, role: null, email: null });
      },
    };
  }, [auth]);

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  return useContext(AuthCtx);
}
