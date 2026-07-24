"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { getToken, logout } from "@/lib/api";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/live", label: "Live" },
  { href: "/analyze", label: "Analyze" },
  { href: "/capability", label: "Capability" },
  { href: "/msa", label: "MSA" },
  { href: "/runs", label: "Runs" },
];

export function Layout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [authed, setAuthed] = useState(false);
  const isLogin = pathname === "/login";

  useEffect(() => {
    setAuthed(!!getToken());
  }, [pathname]);

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
          className="rounded border border-aspc-border px-2 py-1 text-xs text-aspc-muted"
          aria-label="Toggle navigation"
        >
          Menu
        </button>
        <Link href="/" className="font-mono text-sm font-semibold tracking-widest text-aspc-cyan">
          ASPC
        </Link>
        <span className="w-12" />
      </div>

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-56 flex-col border-r border-aspc-border bg-aspc-panel transition-transform md:static md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="border-b border-aspc-border px-5 py-6">
          <Link href="/" className="block" onClick={() => setOpen(false)}>
            <div className="font-mono text-lg font-bold tracking-[0.2em] text-aspc-cyan">ASPC</div>
            <div className="mt-1 text-[11px] uppercase tracking-wider text-aspc-muted">
              Operator Console
            </div>
          </Link>
        </div>

        <nav className="flex-1 space-y-0.5 p-3">
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
                className={`block rounded-md px-3 py-2 text-sm transition ${
                  active
                    ? "bg-aspc-cyan/10 text-aspc-cyan shadow-glow"
                    : "text-aspc-muted hover:bg-aspc-border/40 hover:text-aspc-text"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-aspc-border p-3">
          {authed ? (
            <button
              type="button"
              onClick={() => {
                logout();
                router.push("/login");
              }}
              className="w-full rounded-md border border-aspc-border px-3 py-2 text-left text-xs text-aspc-muted hover:text-aspc-text"
            >
              Sign out
            </button>
          ) : (
            <Link
              href="/login"
              onClick={() => setOpen(false)}
              className="block rounded-md border border-aspc-border px-3 py-2 text-xs text-aspc-cyan hover:bg-aspc-cyan/10"
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

      <main className="flex-1 overflow-x-hidden px-4 pb-10 pt-20 md:px-8 md:pt-8">
        <div className="mx-auto max-w-6xl">{children}</div>
      </main>
    </div>
  );
}
