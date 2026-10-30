"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  Mail,
  Lock,
  Eye,
  EyeOff,
  AlertCircle,
  ArrowRight,
  ShieldCheck,
  Loader2,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export default function InitialAuthPage() {
  const router = useRouter();
  const { user, signIn, isLoading: authLoading, isConfigured } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // If already authenticated, redirect directly to dashboard
  useEffect(() => {
    if (!authLoading && user) {
      router.replace("/dashboard");
    }
  }, [user, authLoading, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedEmail = email.trim();
    if (!trimmedEmail) {
      setError("Please enter your email address.");
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(trimmedEmail)) {
      setError("Please enter a valid email address format.");
      return;
    }

    if (!password) {
      setError("Please enter your password.");
      return;
    }

    setIsSubmitting(true);

    try {
      const { error: authError } = await signIn(trimmedEmail, password);

      if (authError) {
        setError(authError.message || "Failed to authenticate. Please check your credentials.");
      } else {
        router.push("/dashboard");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-softBg flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Centered Auth Card Container */}
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center gap-2.5 mb-3">
            <div className="w-11 h-11 rounded-xl bg-softBg border border-lightBorder flex items-center justify-center text-sageAccent shadow-subtle">
              <Activity className="w-6 h-6 stroke-[2.5]" />
            </div>
            <span className="font-extrabold text-3xl tracking-tight text-darkGreen">
              Knee<span className="text-sageAccent">AI</span>
            </span>
          </div>
          <p className="text-xs uppercase tracking-wider font-bold text-darkGreen">
            Deep Learning Radiograph Analysis
          </p>
        </div>

        {/* Minimal Mode Switcher: Login | Create Account */}
        <div className="flex bg-lightSage/40 border border-lightBorder p-1 rounded-xl mb-6 shadow-inner text-xs font-semibold">
          <div className="flex-1 py-2 text-center rounded-lg bg-white text-darkGreen border border-lightBorder/60 shadow-subtle transition">
            Login
          </div>
          <Link
            href="/register"
            className="flex-1 py-2 text-center rounded-lg text-darkGreen hover:text-black font-semibold transition"
          >
            Create Account
          </Link>
        </div>

        {/* Main Login Card */}
        <Card className="p-8 shadow-subtle border-lightBorder bg-white hover:border-sageAccent hover:shadow-card transition-all duration-200">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-darkGreen">Welcome Back</h2>
            <p className="text-xs text-darkGreen font-medium mt-1">
              Sign in to continue to your account.
            </p>
          </div>

          {/* Development Notice / Quick Access */}
          {!isConfigured && (
            <div className="mb-6 p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 flex items-start gap-3">
              <ShieldCheck className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
              <div className="w-full">
                <p className="font-semibold text-emerald-950">Local Exploration Mode Active</p>
                <p className="mt-1 text-emerald-800 leading-relaxed">
                  Supabase is optional. You can sign in with any email and password, or enter instantly using the demo access below.
                </p>
                <button
                  type="button"
                  onClick={async () => {
                    setIsSubmitting(true);
                    await signIn("researcher@kneeai.local", "demo123");
                    router.push("/dashboard");
                  }}
                  className="mt-2.5 inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-700 hover:bg-emerald-800 text-white rounded-md text-xs font-semibold shadow-sm transition"
                >
                  Quick Demo Access <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-5 p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
              <span>{error}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="email"
                className="block text-xs font-semibold text-darkGreen mb-1.5"
              >
                Email Address
              </label>
              <div className="relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-darkGreen">
                  <Mail className="h-4 w-4" />
                </div>
                <input
                  id="email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@institution.edu"
                  className="block w-full pl-9 pr-3 py-2.5 text-sm border border-lightBorder rounded-lg focus:ring-2 focus:ring-sageAccent focus:border-sageAccent bg-white placeholder-darkGreen/50 text-darkGreen transition font-medium"
                  required
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  htmlFor="password"
                  className="block text-xs font-semibold text-darkGreen"
                >
                  Password
                </label>
                <Link
                  href="/forgot-password"
                  className="text-xs font-semibold text-sageAccent hover:underline transition"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-darkGreen">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="block w-full pl-9 pr-10 py-2.5 text-sm border border-lightBorder rounded-lg focus:ring-2 focus:ring-sageAccent focus:border-sageAccent bg-white placeholder-darkGreen/50 text-darkGreen transition font-medium"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center text-darkGreen hover:text-black focus:outline-none"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            {/* Remember Me Checkbox */}
            <div className="flex items-center pt-1">
              <input
                id="remember-me"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="h-4 w-4 rounded border-lightBorder text-sageAccent focus:ring-sageAccent cursor-pointer"
              />
              <label
                htmlFor="remember-me"
                className="ml-2 block text-xs text-darkGreen font-semibold cursor-pointer select-none"
              >
                Remember me
              </label>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              variant="primary"
              size="lg"
              disabled={isSubmitting}
              className="w-full justify-center mt-2 shadow-sm font-semibold"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Signing In...
                </>
              ) : (
                <>
                  Sign In <ArrowRight className="w-4 h-4 ml-1.5" />
                </>
              )}
            </Button>
          </form>

          {/* Footer Switch to Sign Up */}
          <div className="mt-6 pt-5 border-t border-lightBorder text-center text-xs text-darkGreen font-medium">
            Don&apos;t have an account?{" "}
            <Link
              href="/register"
              className="font-bold text-sageAccent hover:underline"
            >
              Create one
            </Link>
          </div>
        </Card>

        {/* System Disclaimer */}
        <p className="mt-6 text-center text-xs text-darkGreen font-medium max-w-sm mx-auto leading-relaxed">
          KneeAI Automated Radiograph Analysis Platform • Kellgren-Lawrence Grading
        </p>
      </div>
    </div>
  );
}
