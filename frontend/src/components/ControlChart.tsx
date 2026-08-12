"use client";

import dynamic from "next/dynamic";
import { useMemo, type ComponentType } from "react";
import { limitAt } from "@/lib/format";

// Plotly requires browser APIs
const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-72 items-center justify-center rounded-card border border-aspc-border bg-aspc-panel text-sm text-aspc-muted">
      Loading chart…
    </div>
  ),
}) as unknown as ComponentType<Record<string, unknown>>;

export interface ControlChartProps {
  values: number[];
  ucl: number | number[];
  cl: number;
  lcl: number | number[];
  /** Indices of out-of-control points (0-based). */
  oocIndices?: number[];
  title?: string;
  height?: number;
}

export function ControlChart({
  values,
  ucl,
  cl,
  lcl,
  oocIndices = [],
  title = "Control Chart",
  height = 360,
}: ControlChartProps) {
  const x = useMemo(() => values.map((_, i) => i + 1), [values]);
  const oocSet = useMemo(() => new Set(oocIndices), [oocIndices]);

  const inControlX: number[] = [];
  const inControlY: number[] = [];
  const oocX: number[] = [];
  const oocY: number[] = [];

  values.forEach((v, i) => {
    if (oocSet.has(i)) {
      oocX.push(i + 1);
      oocY.push(v);
    } else {
      inControlX.push(i + 1);
      inControlY.push(v);
    }
  });

  const uclSeries = values.map((_, i) => limitAt(ucl, i) ?? null);
  const lclSeries = values.map((_, i) => limitAt(lcl, i) ?? null);
  const clSeries = values.map(() => cl);

  const data = [
    {
      x,
      y: uclSeries,
      type: "scatter",
      mode: "lines",
      name: "UCL",
      line: { color: "#F97316", width: 1.5, dash: "dash" },
      hoverinfo: "y+name",
    },
    {
      x,
      y: clSeries,
      type: "scatter",
      mode: "lines",
      name: "CL",
      line: { color: "#E8C547", width: 2 },
      hoverinfo: "y+name",
    },
    {
      x,
      y: lclSeries,
      type: "scatter",
      mode: "lines",
      name: "LCL",
      line: { color: "#F97316", width: 1.5, dash: "dash" },
      hoverinfo: "y+name",
    },
    {
      x: inControlX,
      y: inControlY,
      type: "scatter",
      mode: "lines+markers",
      name: "Value",
      line: { color: "#F2F2F4", width: 1.5 },
      marker: { color: "#F2F2F4", size: 6 },
    },
    {
      x: oocX,
      y: oocY,
      type: "scatter",
      mode: "markers",
      name: "OOC",
      marker: { color: "#2DD4BF", size: 10, symbol: "x", line: { width: 2, color: "#2DD4BF" } },
    },
  ];

  const layout = {
    title: { text: title, font: { color: "#F2F2F4", size: 14 }, x: 0, xanchor: "left" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#141418",
    font: { color: "#8A8A96", family: "Sora, sans-serif", size: 11 },
    margin: { t: 40, r: 16, b: 40, l: 48 },
    height,
    xaxis: {
      title: "Subgroup / Index",
      gridcolor: "#2A2A32",
      zeroline: false,
      color: "#8A8A96",
    },
    yaxis: {
      title: "Value",
      gridcolor: "#2A2A32",
      zeroline: false,
      color: "#8A8A96",
    },
    legend: {
      orientation: "h",
      y: 1.12,
      x: 1,
      xanchor: "right",
      font: { size: 10 },
    },
    hovermode: "closest",
  };

  const config = { displayModeBar: false, responsive: true };

  return (
    <div className="w-full overflow-hidden rounded-card border border-aspc-border bg-aspc-panel p-2 shadow-card">
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
