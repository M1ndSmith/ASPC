"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, login } from "@/lib/api";
import { ErrorBanner, PrimaryButton, TextInput } from "@/components/ui";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await login(username, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message || "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-aspc-bg px-4">
      <div className="w-full max-w-md rounded-card border border-aspc-border bg-aspc-panel p-8 shadow-card">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-aspc-accent text-aspc-bg shadow-glow">
            <svg viewBox="0 0 24 24" className="h-7 w-7" fill="currentColor">
              <path d="M12 2a10 10 0 1 0 10 10h-4a6 6 0 1 1-6-6V2z" />
            </svg>
          </div>
          <div className="text-2xl font-semibold tracking-wide text-aspc-accent">ASPC</div>
          <h1 className="mt-3 text-xl font-semibold text-aspc-text">Sign in</h1>
          <p className="mt-1 text-sm text-aspc-muted">Operator console · JWT against the ASPC API</p>
        </div>

        {error && <ErrorBanner message={error} />}

        <form onSubmit={onSubmit} className="space-y-4">
          <TextInput
            id="username"
            label="Username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <TextInput
            id="password"
            label="Password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <PrimaryButton type="submit" disabled={busy} className="w-full">
            {busy ? "Signing in…" : "Sign in"}
          </PrimaryButton>
        </form>
      </div>
    </div>
  );
}
