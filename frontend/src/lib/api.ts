import type {
  AnalyzeResponse,
  RunDetail,
  RunSummary,
  StreamInfo,
  TokenResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "aspc_token";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export function reportUrl(runId: string): string {
  return `${API_URL}/reports/${runId}`;
}

async function parseError(res: Response): Promise<string> {
  const text = await res.text();
  if (!text) return res.statusText || `HTTP ${res.status}`;
  try {
    const body = JSON.parse(text) as { detail?: unknown; message?: string };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => (typeof d === "object" && d && "msg" in d ? String(d.msg) : String(d))).join("; ");
    }
    if (body.message) return body.message;
    return text;
  } catch {
    return text;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(`Cannot reach API at ${API_URL}`, 0);
  }

  if (!res.ok) {
    if (res.status === 401 && typeof window !== "undefined") {
      logout();
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
    throw new ApiError(await parseError(res), res.status);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

async function upload(path: string, form: FormData): Promise<AnalyzeResponse> {
  const headers = new Headers();
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { method: "POST", headers, body: form });
  } catch {
    throw new ApiError(`Cannot reach API at ${API_URL}`, 0);
  }

  if (!res.ok) {
    if (res.status === 401 && typeof window !== "undefined") {
      logout();
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
    throw new ApiError(await parseError(res), res.status);
  }

  return res.json() as Promise<AnalyzeResponse>;
}

export async function login(username: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username, password, grant_type: "password" });

  let res: Response;
  try {
    res = await fetch(`${API_URL}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
  } catch {
    throw new ApiError(`Cannot reach API at ${API_URL}`, 0);
  }

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }

  const data = (await res.json()) as TokenResponse;
  if (!data.access_token) {
    throw new ApiError("No access_token in response", res.status);
  }
  setToken(data.access_token);
}

export const api = {
  health: () => request<{ status: string; version?: string }>("/health"),

  listRuns: (params?: { analysis_type?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.analysis_type) q.set("analysis_type", params.analysis_type);
    if (params?.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return request<{ runs: RunSummary[] }>(`/runs${qs ? `?${qs}` : ""}`);
  },

  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),

  analyzeControlChart: (form: FormData) => upload("/analyze/control-chart", form),

  analyzeCapability: (form: FormData) => upload("/analyze/capability", form),

  analyzeMsa: (form: FormData) => upload("/analyze/msa", form),

  /** Optional — returns empty list if the streams endpoint is unavailable. */
  listStreams: async (): Promise<{ streams: StreamInfo[] }> => {
    try {
      return await request<{ streams: StreamInfo[] }>("/streams");
    } catch (err) {
      if (err instanceof ApiError && (err.status === 404 || err.status === 0)) {
        return { streams: [] };
      }
      throw err;
    }
  },
};
