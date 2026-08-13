"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Checklist } from "@/components/Checklist";
import { GateList } from "@/components/GateList";
import {
  ErrorBanner,
  PageHeader,
  Panel,
  PrimaryButton,
  SelectInput,
  Spinner,
  TextInput,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { AnalyzeResponse, Gate, Phase1Checklist, SPCReport } from "@/lib/types";

const API_KEY_STORAGE = "aspc_api_key";

function asReport(raw: AnalyzeResponse["report"]): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

type Step = 1 | 2 | 3 | 4;

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>(1);
  const [dataset, setDataset] = useState("spc_individual_in_control");
  const [catalog, setCatalog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [streamKey, setStreamKey] = useState("demo-line-1");
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? localStorage.getItem(API_KEY_STORAGE) || "" : "",
  );
  const [goLiveDone, setGoLiveDone] = useState(false);

  const report = useMemo(() => (result ? asReport(result.report) : null), [result]);
  const gates = (report?.gates ?? (result?.report as { gates?: Gate[] })?.gates) as Gate[] | undefined;
  const checklist = (report?.checklist ??
    result?.checklist ??
    (result?.report as { checklist?: Phase1Checklist })?.checklist) as Phase1Checklist | undefined;
  const limitsVersion = report?.limits?.version;

  async function loadSampleAndAnalyze() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const sample = await api.onboardingSample(dataset);
      setCatalog(sample.catalog || []);
      const blob = new Blob([sample.csv], { type: "text/csv" });
      const file = new File([blob], sample.filename, { type: "text/csv" });
      const fd = new FormData();
      fd.append("file", file);
      fd.append("ruleset", "nelson");
      const res = await api.analyzeControlChart(fd);
      setResult(res);
      setStep(2);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function doGoLive() {
    if (!limitsVersion) {
      setError("No frozen limits version — Phase I must freeze before go-live");
      return;
    }
    if (!apiKey) {
      setError("API key required (or set ASPC_DEV_INSECURE and use any key in local demos)");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      localStorage.setItem(API_KEY_STORAGE, apiKey);
      await api.registerStream({ stream_key: streamKey }, apiKey);
      await api.goLive(streamKey, { limits_version: limitsVersion }, apiKey);
      setGoLiveDone(true);
      setStep(4);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function openLive() {
    const q = new URLSearchParams({
      stream: streamKey,
      limits: limitsVersion || "",
    });
    router.push(`/live?${q.toString()}`);
  }

  return (
    <div>
      <PageHeader
        title="Onboarding"
        hideTitle
        subtitle="First stream in minutes — sample → establish → go-live → Live WS"
      />

      {error && <ErrorBanner message={error} />}

      <Panel title={`Step ${step} of 4`} className="mb-6">
        <ol className="mb-4 flex flex-wrap gap-2 text-xs text-aspc-muted">
          {["Sample + establish", "Review gates", "Go live", "Open Live"].map((label, i) => (
            <li
              key={label}
              className={`rounded-pill border px-3 py-1 ${
                step === i + 1
                  ? "border-aspc-accent text-aspc-accent"
                  : step > i + 1
                    ? "border-aspc-ok/40 text-aspc-ok"
                    : "border-aspc-border"
              }`}
            >
              {i + 1}. {label}
            </li>
          ))}
        </ol>

        {step === 1 && (
          <div className="grid gap-4 md:grid-cols-2">
            <SelectInput
              id="dataset"
              label="Sample dataset"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
            >
              <option value="spc_individual_in_control">I-MR in control</option>
              <option value="spc_individual_out_of_control">I-MR with mean shift</option>
              <option value="spc_subgroup_data">Xbar-R subgroups</option>
              {catalog
                .filter(
                  (c) =>
                    ![
                      "spc_individual_in_control",
                      "spc_individual_out_of_control",
                      "spc_subgroup_data",
                    ].includes(c),
                )
                .map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
            </SelectInput>
            <div className="flex items-end gap-3">
              <PrimaryButton type="button" onClick={loadSampleAndAnalyze} disabled={busy}>
                {busy ? "Running…" : "Load sample & establish"}
              </PrimaryButton>
              {busy && <Spinner />}
            </div>
          </div>
        )}

        {step === 2 && result && (
          <div className="space-y-4">
            <p className="text-sm text-aspc-muted">
              Run <span className="font-mono text-aspc-accent">{result.run_id}</span>
              {limitsVersion && (
                <>
                  {" "}
                  · limits <span className="font-mono text-aspc-accent">{limitsVersion}</span>
                </>
              )}
            </p>
            {gates && <GateList gates={gates} />}
            {checklist && <Checklist checklist={checklist} />}
            <div className="flex flex-wrap gap-2">
              <PrimaryButton type="button" onClick={() => setStep(3)} disabled={!limitsVersion}>
                Continue to go-live
              </PrimaryButton>
              <button
                type="button"
                className="rounded-pill border border-aspc-border px-4 py-2 text-sm text-aspc-muted"
                onClick={() => setStep(1)}
              >
                Back
              </button>
            </div>
            {!limitsVersion && (
              <p className="text-sm text-aspc-stop">
                Limits were not frozen — resolve STOP gates and retry.
              </p>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="grid gap-4 md:grid-cols-2">
            <TextInput
              id="stream_key"
              label="Stream key"
              value={streamKey}
              onChange={(e) => setStreamKey(e.target.value)}
            />
            <TextInput
              id="api_key"
              label="X-API-Key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <div className="flex items-end gap-2 md:col-span-2">
              <PrimaryButton type="button" onClick={doGoLive} disabled={busy || !limitsVersion}>
                {busy ? "Activating…" : "Register + go live"}
              </PrimaryButton>
              <button
                type="button"
                className="rounded-pill border border-aspc-border px-4 py-2 text-sm text-aspc-muted"
                onClick={() => setStep(2)}
              >
                Back
              </button>
            </div>
            <p className="text-xs text-aspc-muted md:col-span-2">
              Requires Timescale persistence for the stream registry. Limits version:{" "}
              <span className="font-mono">{limitsVersion || "—"}</span>
            </p>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <p className="text-sm text-aspc-ok">
              {goLiveDone
                ? `Stream ${streamKey} is live against frozen limits.`
                : "Ready to open Live monitoring."}
            </p>
            <PrimaryButton type="button" onClick={openLive}>
              Open Live console
            </PrimaryButton>
          </div>
        )}
      </Panel>
    </div>
  );
}
