/** Shared API / report types for the operator console. */

export type GateStatus = "ok" | "warn" | "stop";

export interface Gate {
  step: string;
  status: GateStatus;
  reason: string;
  detail?: Record<string, unknown>;
}

export interface ChecklistItem {
  item: string;
  passed: boolean;
  reason: string;
}

export interface Phase1Checklist {
  passed: boolean;
  items: ChecklistItem[];
}

export interface LimitSet {
  center: number;
  ucl: number | number[];
  lcl: number | number[];
}

export interface ControlLimits {
  version: string;
  chart_type: string;
  subgroup_size?: number;
  components: Record<string, LimitSet>;
}

export interface Signal {
  rule_id: string;
  rule_name: string;
  index: number;
  value: number;
  description: string;
  side?: string;
}

export interface SPCReport {
  analysis_type?: string;
  chart_type: string;
  phase?: string;
  limits: ControlLimits;
  plotted_values: number[];
  secondary_values?: number[];
  secondary_name?: string;
  signals?: Signal[];
  gates?: Gate[];
  checklist?: Phase1Checklist;
  summary?: Record<string, unknown>;
  source_file?: string;
  created_at?: string;
}

export interface AnalyzeResponse {
  status: string;
  run_id: string;
  analysis_type: string;
  report: SPCReport | Record<string, unknown>;
  html_report?: string | null;
}

export interface RunSummary {
  run_id: string;
  analysis_type: string;
  limits_version?: string | null;
  source_file?: string | null;
  user_id?: string | null;
  created_at: string;
}

export interface RunDetail extends RunSummary {
  report: Record<string, unknown>;
}

export interface StreamInfo {
  stream_key: string;
  active: boolean;
  chart_type?: string;
  limits_version?: string;
}

export interface LivePointMessage {
  type?: "point" | "limits" | "alert" | "error";
  value?: number;
  index?: number;
  ts?: string;
  ucl?: number;
  center?: number;
  lcl?: number;
  message?: string;
  limits?: { ucl: number | number[]; center: number; lcl: number | number[] };
  signals?: Signal[];
  signal?: Signal;
}

export interface TokenResponse {
  access_token: string;
  token_type?: string;
}
