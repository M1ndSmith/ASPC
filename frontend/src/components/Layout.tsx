"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { getToken, logout } from "@/lib/api";

const NAV = [
  {
    href: "/",
    label: "Overview",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
  {
    href: "/onboarding",
    label: "Onboarding",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M12 3v18M5 10l7-7 7 7" />
      </svg>
    ),
  },
  {
    href: "/live",
    label: "Live",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M4 14c2-4 4-6 8-6s6 2 8 6" />
        <circle cx="12" cy="17" r="1.5" fill="currentColor" stroke="none" />
      </svg>
    ),
  },
  {
    href: "/analyze",
    label: "Analyze",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M4 19V5M4 19h16" />
        <path d="M8 15v-4M12 15V8M16 15v-6" />
      </svg>
    ),
  },
  {
    href: "/capability",
    label: "Capability",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
  {
    href: "/msa",
    label: "MSA",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M7 7h10v10H7z" />
        <path d="M10 4v3M14 4v3M10 17v3M14 17v3M4 10h3M4 14h3M17 10h3M17 14h3" />
      </svg>
    ),
  },
  {
    href: "/runs",
    label: "Runs",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M6 6h12v12H6z" />
        <path d="M9 10h6M9 14h4" />
      </svg>
    ),
  },
  {
    href: "/lab",
    label: "Lab",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
        <path d="M9 3h6v6l4 8H5l4-8V3z" />
        <path d="M9 9h6" />
      </svg>
    ),
  },
];

function titleForPath(pathname: string): string {
  if (pathname === "/") return "Dashboard";
  if (pathname.startsWith("/onboarding")) return "Onboarding";
  if (pathname.startsWith("/live")) return "Live";
  if (pathname.startsWith("/analyze")) return "Analyze";
  if (pathname.startsWith("/capability")) return "Capability";
  if (pathname.startsWith("/msa")) return "MSA";
  if (pathname.startsWith("/lab")) return "Resilience Lab";
  if (pathname.startsWith("/runs/")) return "Run detail";
  if (pathname.startsWith("/runs")) return "Runs";
  return "ASPC";
}

export function Layout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [query, setQuery] = useState("");
  const isLogin = pathname === "/login";
  const pageTitle = useMemo(() => titleForPath(pathname), [pathname]);

  useEffect(() => {
    const token = getToken();
    setAuthed(!!token);
    if (!token && !isLogin) {
      router.replace("/login");
    }
  }, [pathname, isLogin, router]);

  if (isLogin) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen bg-aspc-bg text-aspc-text">
      {/* Mobile top bar */}
      <div className="fixed inset-x-0 top-0 z-40 flex h-14 items-center justify-between border-b border-aspc-border bg-aspc-panel/95 px-4 backdrop-blur md:hidden">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="rounded-pill border border-aspc-border px-3 py-1.5 text-xs text-aspc-muted"
          aria-label="Toggle navigation"
        >
          Menu
        </button>
        <Link href="/" className="font-semibold tracking-wide text-aspc-accent">
          ASPC
        </Link>
        <span className="w-12" />
      </div>

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-60 flex-col bg-aspc-panel px-3 py-5 transition-transform md:static md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Link
          href="/"
          onClick={() => setOpen(false)}
          className="mb-8 flex items-center gap-3 px-3"
        >
          <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-aspc-accent text-aspc-bg shadow-glow">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor">
              <path d="M12 2a10 10 0 1 0 10 10h-4a6 6 0 1 1-6-6V2z" />
            </svg>
          </span>
          <div>
            <div className="text-sm font-semibold tracking-wide text-aspc-text">ASPC</div>
            <div className="text-[11px] text-aspc-muted">Operator Console</div>
          </div>
        </Link>

        <nav className="flex-1 space-y-1">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-medium transition ${
                  active
                    ? "bg-aspc-elevated text-aspc-accent shadow-card"
                    : "text-aspc-muted hover:bg-aspc-elevated/60 hover:text-aspc-text"
                }`}
              >
                <span className={active ? "text-aspc-accent" : "text-aspc-muted"}>{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto px-1 pt-4">
          {authed ? (
            <button
              type="button"
              onClick={() => {
                logout();
                router.push("/login");
              }}
              className="flex w-full items-center gap-3 rounded-2xl px-3 py-2.5 text-sm text-aspc-muted transition hover:bg-aspc-elevated hover:text-aspc-text"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M10 6H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h4" />
                <path d="M14 16l4-4-4-4M18 12H9" />
              </svg>
              Log out
            </button>
          ) : (
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm text-aspc-accent hover:bg-aspc-accent-soft"
            >
              Sign in
            </Link>
          )}
        </div>
      </aside>

      {open && (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          aria-label="Close menu"
          onClick={() => setOpen(false)}
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col overflow-x-hidden px-4 pb-10 pt-20 md:px-8 md:pt-6">
        <header className="mb-6 flex flex-wrap items-center gap-4">
          <h1 className="min-w-[8rem] text-2xl font-semibold tracking-tight text-aspc-accent md:text-3xl">
            {pageTitle}
          </h1>

          <div className="relative mx-auto hidden w-full max-w-md flex-1 sm:block">
            <span className="pointer-events-none absolute inset-y-0 left-4 flex items-center text-aspc-muted">
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="7" />
                <path d="M20 20l-3-3" />
              </svg>
            </span>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && query.trim()) {
                  router.push(`/runs?q=${encodeURIComponent(query.trim())}`);
                }
              }}
              placeholder="Search runs…"
              className="w-full rounded-pill border border-aspc-border bg-aspc-elevated py-2.5 pl-11 pr-4 text-sm text-aspc-text outline-none placeholder:text-aspc-muted focus:border-aspc-accent/40"
              aria-label="Search runs"
            />
          </div>

          <div className="ml-auto flex items-center gap-2">
            <Link
              href="/runs"
              className="flex h-10 w-10 items-center justify-center rounded-full bg-aspc-elevated text-aspc-muted transition hover:text-aspc-accent"
              aria-label="Runs"
              title="Runs"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M4 6h16M4 12h16M4 18h10" />
              </svg>
            </Link>
            <div
              className="flex h-10 w-10 items-center justify-center rounded-full bg-aspc-accent text-sm font-semibold text-aspc-bg"
              aria-hidden
            >
              A
            </div>
          </div>
        </header>

        <div className="mx-auto w-full max-w-7xl flex-1">{children}</div>
      </main>
    </div>
  );
}
