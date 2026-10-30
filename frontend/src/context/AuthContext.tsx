"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { User, Session, AuthError } from "@supabase/supabase-js";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

interface AuthContextType {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  isConfigured: boolean;
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>;
  signUp: (email: string, password: string, fullName?: string) => Promise<{ error: AuthError | null; user: User | null }>;
  signOut: () => Promise<void>;
  resetPassword: (email: string) => Promise<{ error: AuthError | null }>;
  updatePassword: (password: string) => Promise<{ error: AuthError | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const configured = isSupabaseConfigured();

  useEffect(() => {
    if (configured) {
      supabase.auth.getSession().then(({ data: { session } }) => {
        setSession(session);
        setUser(session?.user ?? null);
        setIsLoading(false);
      });

      const {
        data: { subscription },
      } = supabase.auth.onAuthStateChange((_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setIsLoading(false);
      });

      return () => {
        subscription.unsubscribe();
      };
    } else {
      // Local exploration session restore
      if (typeof window !== "undefined") {
        try {
          const saved = localStorage.getItem("kneeai_local_session");
          if (saved) {
            const parsed = JSON.parse(saved);
            setUser(parsed);
            setSession({
              access_token: "demo-token",
              token_type: "bearer",
              expires_in: 86400,
              refresh_token: "demo-refresh",
              user: parsed,
            });
          }
        } catch {
          // ignore
        }
      }
      setIsLoading(false);
    }
  }, [configured]);

  const signIn = async (email: string, password: string): Promise<{ error: AuthError | null }> => {
    if (!configured) {
      const mockUser: User = {
        id: "demo-researcher-id",
        app_metadata: { provider: "email" },
        user_metadata: { full_name: email.split("@")[0] || "Researcher" },
        aud: "authenticated",
        created_at: new Date().toISOString(),
        email: email || "researcher@kneeai.local",
        phone: "",
        role: "authenticated",
        updated_at: new Date().toISOString(),
      };
      setUser(mockUser);
      setSession({
        access_token: "demo-token",
        token_type: "bearer",
        expires_in: 86400,
        refresh_token: "demo-refresh",
        user: mockUser,
      });
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("kneeai_local_session", JSON.stringify(mockUser));
        } catch {
          // ignore
        }
      }
      return { error: null };
    }

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    return { error };
  };

  const signUp = async (
    email: string,
    password: string,
    fullName?: string
  ): Promise<{ error: AuthError | null; user: User | null }> => {
    if (!configured) {
      const mockUser: User = {
        id: "demo-researcher-id",
        app_metadata: { provider: "email" },
        user_metadata: { full_name: fullName || email.split("@")[0] || "Researcher" },
        aud: "authenticated",
        created_at: new Date().toISOString(),
        email: email || "researcher@kneeai.local",
        phone: "",
        role: "authenticated",
        updated_at: new Date().toISOString(),
      };
      setUser(mockUser);
      setSession({
        access_token: "demo-token",
        token_type: "bearer",
        expires_in: 86400,
        refresh_token: "demo-refresh",
        user: mockUser,
      });
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("kneeai_local_session", JSON.stringify(mockUser));
        } catch {
          // ignore
        }
      }
      return { error: null, user: mockUser };
    }

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName || "",
        },
        emailRedirectTo: `${typeof window !== "undefined" ? window.location.origin : ""}/login`,
      },
    });
    return { error, user: data?.user ?? null };
  };

  const signOut = async (): Promise<void> => {
    if (configured) {
      await supabase.auth.signOut();
    } else {
      if (typeof window !== "undefined") {
        try {
          localStorage.removeItem("kneeai_local_session");
        } catch {
          // ignore
        }
      }
    }
    setUser(null);
    setSession(null);
  };

  const resetPassword = async (email: string): Promise<{ error: AuthError | null }> => {
    if (!configured) {
      const configErr = {
        name: "ConfigurationError",
        message:
          "Supabase Authentication is not configured. Please define NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in frontend/.env.local.",
      } as AuthError;
      return { error: configErr };
    }

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${typeof window !== "undefined" ? window.location.origin : ""}/reset-password`,
    });
    return { error };
  };

  const updatePassword = async (password: string): Promise<{ error: AuthError | null }> => {
    if (!configured) {
      const configErr = {
        name: "ConfigurationError",
        message: "Supabase Authentication is not configured.",
      } as AuthError;
      return { error: configErr };
    }

    const { error } = await supabase.auth.updateUser({ password });
    return { error };
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        isLoading,
        isConfigured: configured,
        signIn,
        signUp,
        signOut,
        resetPassword,
        updatePassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
