"use client";

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
  hideTitle = false,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  /** When true, only subtitle/actions render (shell already shows page title). */
  hideTitle?: boolean;
}) {
  if (hideTitle && !subtitle && !actions) return null;
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        {!hideTitle && (
          <h1 className="text-2xl font-semibold tracking-tight text-aspc-accent md:text-3xl">{title}</h1>
        )}
        {subtitle && <p className={`text-sm text-aspc-muted ${hideTitle ? "" : "mt-1"}`}>{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Panel({
  title,
  children,
  className = "",
  variant = "default",
  action,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  variant?: "default" | "accent";
  action?: ReactNode;
}) {
  const base =
    variant === "accent"
      ? "bg-aspc-accent text-aspc-bg border-transparent"
      : "bg-aspc-panel text-aspc-text border-aspc-border";
  return (
    <section className={`rounded-card border p-5 shadow-card md:p-6 ${base} ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex items-center justify-between gap-3">
          {title && (
            <h2
              className={`text-base font-semibold tracking-tight ${
                variant === "accent" ? "text-aspc-bg" : "text-aspc-text"
              }`}
            >
              {title}
            </h2>
          )}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
  className = "",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "ok" | "warn" | "stop" | "cyan" | "accent";
  className?: string;
}) {
  const toneClass =
    tone === "ok"
      ? "text-aspc-ok"
      : tone === "warn"
        ? "text-aspc-warn"
        : tone === "stop"
          ? "text-aspc-stop"
          : tone === "cyan" || tone === "accent"
            ? "text-aspc-accent"
            : "text-aspc-text";

  return (
    <div className={`rounded-card border border-aspc-border bg-aspc-panel p-5 shadow-card ${className}`}>
      <div className="text-[11px] font-medium uppercase tracking-widest text-aspc-muted">{label}</div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-aspc-muted">{hint}</div>}
    </div>
  );
}

export function DeltaChip({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean | null;
}) {
  const tone =
    positive === true ? "text-aspc-ok" : positive === false ? "text-aspc-warn" : "text-aspc-muted";
  return (
    <div className="min-w-[5.5rem] text-center">
      <div className="text-[11px] uppercase tracking-wider text-aspc-muted">{label}</div>
      <div className={`mt-1 flex items-center justify-center gap-1 text-sm font-semibold ${tone}`}>
        {positive === true && <span aria-hidden>↑</span>}
        {positive === false && <span aria-hidden>↓</span>}
        {value}
      </div>
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="mb-4 rounded-2xl border border-aspc-stop/40 bg-aspc-stop/10 px-4 py-3 text-sm text-aspc-stop">
      {message}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-aspc-muted">
      <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-aspc-accent border-t-transparent" />
      {label}
    </div>
  );
}

export function FileField({
  id,
  label,
  accept = ".csv,.parquet,.pq",
  onChange,
  required,
}: {
  id: string;
  label: string;
  accept?: string;
  onChange: (file: File | null) => void;
  required?: boolean;
}) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        required={required}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="block w-full cursor-pointer rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm file:mr-3 file:rounded-pill file:border-0 file:bg-aspc-accent-soft file:px-3 file:py-1 file:text-xs file:font-medium file:text-aspc-accent"
      />
    </div>
  );
}

export function TextInput({
  id,
  label,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { label: string }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <input
        id={id}
        {...props}
        className={`w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm text-aspc-text outline-none placeholder:text-aspc-muted/60 focus:border-aspc-accent/50 focus:ring-1 focus:ring-aspc-accent/25 ${props.className || ""}`}
      />
    </div>
  );
}

export function SelectInput({
  id,
  label,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & { label: string }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <select
        id={id}
        {...props}
        className={`w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm text-aspc-text outline-none focus:border-aspc-accent/50 focus:ring-1 focus:ring-aspc-accent/25 ${props.className || ""}`}
      >
        {children}
      </select>
    </div>
  );
}

export function PrimaryButton({
  children,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex items-center justify-center rounded-pill bg-aspc-accent px-5 py-2.5 text-sm font-semibold text-aspc-bg transition hover:bg-aspc-accent-dim disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {children}
    </button>
  );
}
