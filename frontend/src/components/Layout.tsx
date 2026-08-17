"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  Beaker,
  Gauge,
  LayoutGrid,
  LineChart,
  LogOut,
  Menu,
  Radio,
  Ruler,
  ScrollText,
  Sparkles,
} from "lucide-react";
import { Led, Tooltip } from "@/components/ui";
import { api, getToken, logout } from "@/lib/api";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutGrid, tip: "Home: is the system up, and what should I do next?" },
  { href: "/onboarding", label: "Onboarding", icon: Sparkles, tip: "A guided walkthrough using a sample file." },
  { href: "/live", label: "Live", icon: Radio, tip: "Watch measurements as they arrive on a line." },
  { href: "/analyze", label: "Analyze", icon: LineChart, tip: "Upload a spreadsheet of measurements." },
  { href: "/capability", label: "Capability", icon: Gauge, tip: "Check whether the process stays inside spec." },
  { href: "/msa", label: "MSA", icon: Ruler, tip: "Check whether the gage (the measuring tool) is trustworthy." },
  { href: "/runs", label: "Runs", icon: ScrollText, tip: "Past analyses you can reopen." },
  { href: "/lab", label: "Lab", icon: Beaker, tip: "Try known good and bad examples." },
];

function titleForPath(pathname: string): string {
  if (pathname === "/") return "Overview";
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
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const isLogin = pathname === "/login";
  const pageTitle = useMemo(() => titleForPath(pathname), [pathname]);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    enabled: !isLogin && authed,
  });
  const me = useQuery({
    queryKey: ["me"],
    queryFn: api.me,
    enabled: !isLogin && authed,
    retry: false,
  });

  useEffect(() => {
    const token = getToken();
    setAuthed(!!token);
    if (!token && !isLogin) {
      router.replace("/login");
    }
  }, [pathname, isLogin, router]);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  if (isLogin) {
    return <>{children}</>;
  }

  const username = me.data?.username || "";
  const initial = (username[0] || "?").toUpperCase();
  const healthy = health.data?.status === "healthy" || health.data?.status === "ok";

  return (
    <div className="flex min-h-screen bg-aspc-bg text-aspc-text">
      <div className="fixed inset-x-0 top-0 z-40 flex h-14 items-center justify-between bg-aspc-bg/90 px-4 shadow-card backdrop-blur md:hidden">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex h-12 w-12 items-center justify-center rounded-lg bg-aspc-bg text-aspc-text shadow-card active:shadow-pressed"
          aria-label="Toggle navigation"
          title="Open or close the page list"
        >
          <Menu className="h-5 w-5" strokeWidth={1.5} />
        </button>
        <Link href="/" className="font-semibold tracking-wide text-aspc-text">
          ASPC
        </Link>
        <span className="w-12" />
      </div>

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-60 flex-col bg-aspc-bg px-3 py-5 shadow-card transition-transform duration-200 md:static md:translate-x-0 md:shadow-none ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Link href="/" onClick={() => setOpen(false)} className="mb-8 flex items-center gap-3 px-3" title="ASPC home">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-aspc-accent text-white shadow-key">
            <Activity className="h-5 w-5" strokeWidth={2} />
          </span>
          <div className="text-sm font-semibold tracking-wide text-aspc-text">ASPC</div>
        </Link>

        <nav className="flex-1 space-y-1">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname === item.href || pathname.startsWith(`${item.href}/`);
            const Icon = item.icon;
            return (
              <Tooltip key={item.href} content={item.tip} className="w-full">
                <Link
                  href={item.href}
                  onClick={() => setOpen(false)}
                  title={item.tip}
                  className={`flex min-h-touch items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition duration-200 ${
                    active
                      ? "bg-aspc-elevated text-aspc-accent shadow-recessed"
                      : "text-aspc-muted hover:bg-aspc-elevated/60 hover:text-aspc-text"
                  }`}
                >
                  <Icon className={`h-5 w-5 ${active ? "text-aspc-accent" : ""}`} strokeWidth={1.5} />
                  {item.label}
                </Link>
              </Tooltip>
            );
          })}
        </nav>

        <div className="mt-auto px-3 pt-4">
          <Tooltip content="Whether the server that stores results is reachable.">
            <span>
              <Led
                on={healthy}
                tone={healthy ? "ok" : "accent"}
                label={healthy ? "System ok" : health.isError ? "System down" : "Checking"}
              />
            </span>
          </Tooltip>
        </div>
      </aside>

      {open && (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-aspc-dark/40 md:hidden"
          aria-label="Close menu"
          onClick={() => setOpen(false)}
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col overflow-x-hidden px-4 pb-10 pt-20 md:px-8 md:pt-6">
        <header className="mb-6 flex flex-wrap items-center gap-4">
          <h1 className="min-w-[8rem] text-2xl font-extrabold tracking-tight text-aspc-text drop-shadow-[0_1px_0_#ffffff] md:text-3xl">
            {pageTitle}
          </h1>

          <div className="ml-auto flex items-center gap-3" ref={menuRef}>
            {authed ? (
              <div className="relative">
                <Tooltip content="Your account. Open this to log out.">
                  <button
                    type="button"
                    onClick={() => setMenuOpen((v) => !v)}
                    title="Your account. Open this to log out."
                    className="flex h-12 w-12 items-center justify-center rounded-full bg-aspc-accent text-sm font-bold text-white shadow-key transition duration-150 active:translate-y-[2px] active:shadow-pressed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-aspc-accent focus-visible:ring-offset-2"
                    aria-label={username ? `Account menu for ${username}` : "Account menu"}
                    aria-expanded={menuOpen}
                  >
                    {initial}
                  </button>
                </Tooltip>
                {menuOpen && (
                  <div className="absolute right-0 z-30 mt-2 w-52 rounded-lg bg-aspc-bg p-3 shadow-floating">
                    <p className="truncate font-mono text-xs font-bold uppercase tracking-wider text-aspc-muted">
                      {username || "Signed in"}
                    </p>
                    {me.data?.role && (
                      <p className="mt-0.5 font-mono text-[11px] text-aspc-muted">{me.data.role}</p>
                    )}
                    <Tooltip content="Sign out of this session">
                      <button
                        type="button"
                        title="Sign out of this session"
                        onClick={() => {
                          logout();
                          router.push("/login");
                        }}
                        className="mt-3 flex w-full min-h-touch items-center gap-2 rounded-lg px-2 text-sm text-aspc-muted hover:bg-aspc-elevated hover:text-aspc-text"
                      >
                        <LogOut className="h-4 w-4" strokeWidth={1.5} />
                        Log out
                      </button>
                    </Tooltip>
                  </div>
                )}
              </div>
            ) : (
              <Link
                href="/login"
                title="Sign in to use ASPC"
                className="rounded-lg px-4 py-2 text-sm font-bold uppercase tracking-wide text-aspc-accent"
              >
                Sign in
              </Link>
            )}
          </div>
        </header>

        <div className="mx-auto w-full max-w-6xl flex-1">{children}</div>
      </main>
    </div>
  );
}
