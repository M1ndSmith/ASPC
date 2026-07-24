"use client";

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3 border-b border-aspc-border pb-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-aspc-text">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-aspc-muted">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Panel({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-aspc-border bg-aspc-panel p-4 md:p-5 ${className}`}
    >
      {title && (
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-aspc-muted">
          {title}
        </h2>
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
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "ok" | "warn" | "stop" | "cyan";
}) {
  const toneClass =
    tone === "ok"
      ? "text-aspc-ok"
      : tone === "warn"
        ? "text-aspc-warn"
        : tone === "stop"
          ? "text-aspc-stop"
          : tone === "cyan"
            ? "text-aspc-cyan"
            : "text-aspc-text";

  return (
    <div className="rounded-xl border border-aspc-border bg-aspc-panel p-4">
      <div className="text-[11px] uppercase tracking-widest text-aspc-muted">{label}</div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-aspc-muted">{hint}</div>}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="mb-4 rounded-lg border border-aspc-stop/40 bg-aspc-stop/10 px-4 py-3 text-sm text-aspc-stop">
      {message}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-aspc-muted">
      <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-aspc-cyan border-t-transparent" />
      {label}
    </div>
  );
}

export function FileField({
  id,
  label,
  accept = ".csv,.parquet,.xlsx",
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
      <label htmlFor={id} className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        required={required}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="block w-full cursor-pointer rounded-lg border border-aspc-border bg-aspc-bg px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-aspc-cyan/15 file:px-3 file:py-1 file:text-xs file:font-medium file:text-aspc-cyan"
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
      <label htmlFor={id} className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <input
        id={id}
        {...props}
        className={`w-full rounded-lg border border-aspc-border bg-aspc-bg px-3 py-2 text-sm text-aspc-text outline-none placeholder:text-aspc-muted/60 focus:border-aspc-cyan/50 focus:ring-1 focus:ring-aspc-cyan/30 ${props.className || ""}`}
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
      <label htmlFor={id} className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted">
        {label}
      </label>
      <select
        id={id}
        {...props}
        className={`w-full rounded-lg border border-aspc-border bg-aspc-bg px-3 py-2 text-sm text-aspc-text outline-none focus:border-aspc-cyan/50 focus:ring-1 focus:ring-aspc-cyan/30 ${props.className || ""}`}
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
      className={`inline-flex items-center justify-center rounded-lg bg-aspc-cyan px-4 py-2 text-sm font-semibold text-aspc-bg transition hover:bg-aspc-cyan-dim disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {children}
    </button>
  );
}
