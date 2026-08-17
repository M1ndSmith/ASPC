"use client";

import {
  useId,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TdHTMLAttributes,
  type ThHTMLAttributes,
} from "react";
import { Info } from "lucide-react";
import { GLOSSARY } from "@/lib/glossary";

const LABEL = "mb-1.5 block text-xs font-bold uppercase tracking-[0.06em] text-aspc-muted font-mono";
const WELL =
  "w-full min-h-14 rounded-md border-none bg-aspc-bg px-6 py-3 font-mono text-sm text-aspc-text outline-none shadow-recessed placeholder:text-aspc-muted/50 focus-visible:shadow-[inset_4px_4px_8px_#babecc,inset_-4px_-4px_8px_#ffffff,0_0_0_2px_#ff4757] disabled:cursor-not-allowed disabled:opacity-50";

export function PageHeader({
  title,
  subtitle,
  actions,
  hideTitle = false,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  hideTitle?: boolean;
}) {
  if (hideTitle && !subtitle && !actions) return null;
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div className="max-w-3xl">
        {!hideTitle && (
          <h1 className="text-2xl font-extrabold tracking-tight text-aspc-text drop-shadow-[0_1px_0_#ffffff] md:text-3xl">
            {title}
          </h1>
        )}
        {subtitle && (
          <p className={`text-sm leading-relaxed text-aspc-muted ${hideTitle ? "" : "mt-1"}`}>{subtitle}</p>
        )}
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
  elevated = false,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  variant?: "default" | "accent";
  action?: ReactNode;
  elevated?: boolean;
}) {
  const dark = variant === "accent";
  return (
    <section
      className={`relative rounded-lg p-6 md:p-8 transition duration-300 ease-mech ${
        dark
          ? "bg-aspc-dark text-white shadow-sharp"
          : elevated
            ? "bg-aspc-bg text-aspc-text shadow-floating hover:-translate-y-0.5"
            : "bg-aspc-bg text-aspc-text shadow-card hover:-translate-y-0.5 hover:shadow-floating"
      } ${className}`}
      style={
        dark
          ? undefined
          : {
              backgroundImage:
                "radial-gradient(circle at 12px 12px, rgba(0,0,0,0.15) 2px, transparent 3px), radial-gradient(circle at calc(100% - 12px) 12px, rgba(0,0,0,0.15) 2px, transparent 3px), radial-gradient(circle at 12px calc(100% - 12px), rgba(0,0,0,0.15) 2px, transparent 3px), radial-gradient(circle at calc(100% - 12px) calc(100% - 12px), rgba(0,0,0,0.15) 2px, transparent 3px)",
            }
      }
    >
      {!dark && (
        <div className="absolute right-5 top-4 hidden gap-1 sm:flex" aria-hidden>
          <span className="h-6 w-1 rounded-full bg-aspc-elevated shadow-[inset_1px_1px_2px_rgba(0,0,0,0.12)]" />
          <span className="h-6 w-1 rounded-full bg-aspc-elevated shadow-[inset_1px_1px_2px_rgba(0,0,0,0.12)]" />
          <span className="h-6 w-1 rounded-full bg-aspc-elevated shadow-[inset_1px_1px_2px_rgba(0,0,0,0.12)]" />
        </div>
      )}
      {(title || action) && (
        <div className="mb-4 flex items-center justify-between gap-3 pr-8">
          {title && (
            <h2
              className={`text-base font-bold tracking-tight ${dark ? "text-white" : "text-aspc-text"}`}
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

export function Button({
  children,
  className = "",
  variant = "primary",
  tip,
  title,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  tip?: string;
}) {
  const look =
    variant === "primary"
      ? "bg-aspc-accent text-aspc-accent-fg shadow-key border border-white/20 hover:brightness-110 active:shadow-pressed"
      : variant === "secondary"
        ? "bg-aspc-bg text-aspc-text shadow-card hover:text-aspc-accent active:shadow-pressed"
        : "bg-transparent text-aspc-muted hover:bg-aspc-elevated hover:text-aspc-text hover:shadow-recessed";
  const btn = (
    <button
      type="button"
      title={tip || title}
      {...props}
      className={`inline-flex min-h-touch items-center justify-center rounded-lg px-5 py-2.5 text-xs font-bold uppercase tracking-[0.05em] transition duration-150 ease-mech active:translate-y-[2px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-aspc-accent focus-visible:ring-offset-2 focus-visible:ring-offset-aspc-bg disabled:cursor-not-allowed disabled:opacity-50 ${look} ${className}`}
    >
      {children}
    </button>
  );
  return tip ? <Tooltip content={tip}>{btn}</Tooltip> : btn;
}

export function PrimaryButton({
  children,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <Button className={className} {...props}>
      {children}
    </Button>
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
    <div className={`rounded-lg bg-aspc-bg p-5 shadow-card ${className}`}>
      <div className="font-mono text-xs font-bold uppercase tracking-[0.08em] text-aspc-muted">{label}</div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-aspc-muted">{hint}</div>}
    </div>
  );
}

export function DeltaChip({
  label,
  value,
  positive,
  inverted = false,
  tip,
}: {
  label: string;
  value: string;
  positive?: boolean | null;
  inverted?: boolean;
  tip?: string;
}) {
  const tone =
    positive === true ? "text-aspc-ok" : positive === false ? "text-aspc-warn" : inverted ? "text-white/70" : "text-aspc-muted";
  const chip = (
    <div className="min-w-[5.5rem] text-center">
      <div
        className={`font-mono text-xs font-bold uppercase tracking-[0.08em] ${
          inverted ? "text-white/60" : "text-aspc-muted"
        }`}
      >
        {label}
      </div>
      <div className={`mt-1 flex items-center justify-center gap-1 font-mono text-sm font-semibold ${tone}`}>
        {positive === true && <span aria-hidden>↑</span>}
        {positive === false && <span aria-hidden>↓</span>}
        {value}
      </div>
    </div>
  );
  return tip ? <Tooltip content={tip}>{chip}</Tooltip> : chip;
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="mb-4 rounded-lg border border-aspc-accent/30 bg-aspc-accent-soft px-4 py-3 text-sm text-aspc-accent"
    >
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

export function Skeleton({ className = "h-24" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-aspc-elevated shadow-recessed ${className}`} />;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-40 items-center justify-center rounded-lg bg-aspc-elevated px-4 py-8 text-center text-sm text-aspc-muted shadow-recessed">
      {children}
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
      <label htmlFor={id} className={LABEL}>
        {label}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        required={required}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="block w-full min-h-touch cursor-pointer rounded-md bg-aspc-bg px-4 py-3 text-sm shadow-recessed file:mr-3 file:rounded-md file:border-0 file:bg-aspc-accent file:px-3 file:py-1.5 file:text-xs file:font-bold file:uppercase file:tracking-wide file:text-white"
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
      <label htmlFor={id} className={LABEL}>
        {label}
      </label>
      <input id={id} {...props} className={`${WELL} ${props.className || ""}`} />
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
      <label htmlFor={id} className={LABEL}>
        {label}
      </label>
      <select id={id} {...props} className={`${WELL} ${props.className || ""}`}>
        {children}
      </select>
    </div>
  );
}

export function Led({
  on,
  label,
  tone = "ok",
  inverted = false,
}: {
  on: boolean;
  label?: string;
  tone?: "ok" | "accent" | "warn";
  inverted?: boolean;
}) {
  const color =
    !on ? "bg-aspc-border" : tone === "accent" ? "bg-aspc-accent shadow-glow" : tone === "warn" ? "bg-aspc-warn" : "bg-aspc-ok shadow-glow-ok";
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`h-2.5 w-2.5 rounded-full ${on ? "animate-pulse" : ""} ${color}`} aria-hidden />
      {label && (
        <span
          className={`font-mono text-[11px] font-bold uppercase tracking-[0.08em] ${
            inverted ? "text-white/70" : "text-aspc-muted"
          }`}
        >
          {label}
        </span>
      )}
    </span>
  );
}

export function Tooltip({
  content,
  children,
  className = "",
}: {
  content: string;
  children: ReactNode;
  className?: string;
}) {
  const tipId = useId();
  const [open, setOpen] = useState(false);
  return (
    <span
      className={`relative inline-flex ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocusCapture={() => setOpen(true)}
      onBlurCapture={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          id={tipId}
          role="tooltip"
          className="pointer-events-none absolute top-full left-1/2 z-50 mt-2 w-64 -translate-x-1/2 rounded-md bg-aspc-dark px-3 py-2 text-left text-xs font-normal normal-case tracking-normal text-white shadow-sharp"
        >
          {content}
        </span>
      )}
    </span>
  );
}

export function Term({ k, children }: { k: keyof typeof GLOSSARY; children?: ReactNode }) {
  const entry = GLOSSARY[k];
  return (
    <Tooltip content={entry.def}>
      <span tabIndex={0} className="term-underline font-medium">
        {children ?? entry.label}
      </span>
    </Tooltip>
  );
}

export function InfoTip({ content }: { content: string }) {
  return (
    <Tooltip content={content}>
      <span
        className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-aspc-bg text-aspc-muted shadow-card"
        aria-label={content}
      >
        <Info className="h-3 w-3" strokeWidth={2} />
      </span>
    </Tooltip>
  );
}

export function DataTable({
  headers,
  children,
}: {
  headers: string[];
  children: ReactNode;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[20rem] text-left text-sm">
        <thead>
          <tr className="border-b border-aspc-border font-mono text-xs font-bold uppercase tracking-[0.06em] text-aspc-muted">
            {headers.map((h) => (
              <th key={h} className="pb-2 pr-3 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Th(props: ThHTMLAttributes<HTMLTableCellElement>) {
  return <th {...props} />;
}

export function Td({ className = "", ...props }: TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={`border-b border-aspc-border/60 py-2.5 pr-3 ${className}`} {...props} />;
}

export function JsonBlock({ value, maxHeight = "24rem" }: { value: unknown; maxHeight?: string }) {
  return (
    <pre
      className="overflow-auto whitespace-pre-wrap rounded-md bg-aspc-elevated p-4 font-mono text-xs text-aspc-text shadow-recessed"
      style={{ maxHeight }}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
