"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Checklist } from "@/components/Checklist";
import { GateList } from "@/components/GateList";
import { GoLivePanel } from "@/components/GoLivePanel";
import { Button, ErrorBanner, PageHeader, Panel, SelectInput, Spinner, Term } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { AnalyzeResponse, Gate, Phase1Checklist, SPCReport } from "@/lib/types";

function asReport(raw: AnalyzeResponse["report"]): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

type Step = 1 | 2 | 3 | 4;

const STEPS = ["Sample + establish", "Review gates", "Go live", "Open Live"] as const;

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>(1);
  const [dataset, setDataset] = useState("spc_individual_in_control");
  const [catalog, setCatalog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [streamKey, setStreamKey] = useState("demo-line-1");
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
        subtitle={
          <>
            Walk a sample through the full path: load data, lock limits, then watch a live stream.{" "}
            <span className="font-mono text-xs">
              sample → <Term k="establish" /> → <Term k="go-live" /> → Live WS
            </span>
          </>
        }
      />

      {error && <ErrorBanner message={error} />}

      <Panel title={`Step ${step} of 4`} className="mb-6">
        <ol className="mb-4 flex flex-wrap gap-2 text-xs text-aspc-muted">
          {STEPS.map((label, i) => (
            <li
              key={label}
              className={`rounded-pill px-3 py-1 font-mono ${
                step === i + 1
                  ? "bg-aspc-accent text-white shadow-key"
                  : step > i + 1
                    ? "bg-aspc-ok/15 text-aspc-ok"
                    : "bg-aspc-elevated shadow-recessed"
              }`}
            >
              {i + 1}. {label}
            </li>
          ))}
        </ol>

        {step === 1 && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-bold">Load a sample and establish limits</h3>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What this does: ASPC runs <Term k="phase-i" /> on a known-good file so you do not need your own CSV yet.
                Samples include <Term k="i-mr" /> and <Term k="xbar-r" />.
              </p>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What you get: a chart type, proposed limits, and a list of <Term k="gate">gates</Term> to review.
              </p>
            </div>
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
                      !["spc_individual_in_control", "spc_individual_out_of_control", "spc_subgroup_data"].includes(c),
                  )
                  .map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
              </SelectInput>
              <div className="flex items-end gap-3">
                <Button
                  type="button"
                  onClick={loadSampleAndAnalyze}
                  disabled={busy}
                  tip="Load a practice file and calculate the first set of limits."
                >
                  {busy ? "Running…" : "Load sample & establish"}
                </Button>
                {busy && <Spinner />}
              </div>
            </div>
          </div>
        )}

        {step === 2 && result && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-bold">Review gates and freeze</h3>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What this does: each <Term k="gate" /> is a quality check. A <Term k="stop-gate" /> blocks freeze.
              </p>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What you get: a <Term k="limits-version" /> you can attach to a live stream.
              </p>
            </div>
            <p className="text-sm text-aspc-muted">
              Run <span className="font-mono text-aspc-accent">{result.run_id}</span>
              {limitsVersion && (
                <>
                  {" "}
                  · limits <span className="font-mono text-aspc-accent">{limitsVersion}</span>
                </>
              )}
            </p>
            {gates && (
              <Panel title="Gates">
                <GateList gates={gates} />
              </Panel>
            )}
            {checklist && (
              <Panel title="Phase I Checklist">
                <Checklist checklist={checklist} />
              </Panel>
            )}
            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => setStep(3)} disabled={!limitsVersion} tip="Next: attach these limits to a live line.">
                Continue to go-live
              </Button>
              <Button type="button" variant="secondary" onClick={() => setStep(1)} tip="Go back and pick a different sample.">
                Back
              </Button>
            </div>
            {!limitsVersion && (
              <p className="text-sm text-aspc-accent">Limits were not frozen — resolve STOP gates and retry.</p>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-bold">Activate the stream</h3>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What this does: <Term k="go-live" /> tells the engine to judge new points against these locked limits.
              </p>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What you get: a named <Term k="stream-key" /> you can open on Live.
              </p>
            </div>
            <GoLivePanel
              title="Go live"
              limitsVersion={limitsVersion}
              streamKey={streamKey}
              onStreamKeyChange={setStreamKey}
              onSuccess={() => {
                setGoLiveDone(true);
                setStep(4);
              }}
            />
            <Button type="button" variant="ghost" onClick={() => setStep(2)} tip="Go back to the quality checks.">
              Back
            </Button>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-bold">Watch the live chart</h3>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What this does: opens <Term k="phase-ii" /> monitoring for the stream you just activated.
              </p>
              <p className="mt-1 max-w-xl text-sm leading-relaxed text-aspc-muted">
                What you get: a WebSocket chart and an alert feed against frozen limits.
              </p>
            </div>
            <p className="text-sm text-aspc-ok">
              {goLiveDone
                ? `Stream ${streamKey} is live against frozen limits.`
                : "Ready to open Live monitoring."}
            </p>
            <Button type="button" onClick={openLive} tip="Open the live chart for this line.">
              Open Live console
            </Button>
          </div>
        )}
      </Panel>
    </div>
  );
}
