"""FastAPI application — typed batch endpoints over spc_core.

Endpoints:
  POST /auth/token
  POST /analyze/control-chart | /capability | /msa
  GET  /runs/{run_id} | /runs | /reports/{run_id}
  GET  /health | /metrics
  GET  /stream/replay  (SSE)
  WS   /ws/live/{stream_key}
  POST /streams/register | /streams/{key}/go-live | /alerts/{id}/ack
  GET  /streams
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from adapters.factory import get_repository
from adapters.io_files import FileReadError, load_columns, save_upload
from adapters.render_plotly import (
    render_capability_html,
    render_control_chart_html,
    render_msa_html,
    save_html,
)
from adapters.stream import FileReplaySource, stream_evaluate
from apps.config import get_config
from spc_core import (
    ChartType,
    bias_study,
    capability_analysis,
    check_normality,
    establish,
    gage_rr_anova,
    gage_rr_range,
    ingest,
    linearity_study,
    phase1_checklist,
    stability_study,
)
from spc_core.models import ControlLimits, LimitSet
from spc_core.report import CapabilityReport, MSAReport, SPCReport

cfg = get_config()
repo = get_repository(cfg)

app = FastAPI(
    title="ASPC — Statistical Process Control API",
    description="Correct, tested SPC core with batch analysis and Phase II streaming.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    REQUESTS = Counter("aspc_http_requests_total", "HTTP requests", ["method", "path", "status"])
    ANALYZE_LATENCY = Histogram("aspc_analyze_seconds", "Analyze endpoint latency", ["kind"])
    _PROM = True
except ImportError:  # pragma: no cover
    _PROM = False


# ---- auth helpers -------------------------------------------------------------

def _create_access_token(subject: str) -> str:
    try:
        from jose import jwt
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "python-jose required for JWT auth") from exc
    expire = datetime.now(timezone.utc) + timedelta(minutes=cfg.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict[str, Any]:
    """Validate Bearer JWT. When auth is disabled, return anonymous user."""
    if not cfg.auth_enabled:
        return {"username": "anonymous", "auth": "disabled"}
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        from jose import JWTError, jwt
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "python-jose required for JWT auth") from exc
    try:
        data = jwt.decode(
            credentials.credentials,
            cfg.jwt_secret,
            algorithms=[cfg.jwt_algorithm],
        )
        username = data.get("sub")
        if not username:
            raise HTTPException(401, "Invalid token")
        return {"username": username, "auth": "jwt"}
    except JWTError as exc:
        raise HTTPException(401, "Invalid or expired token") from exc


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """Require X-API-Key for ingest / stream mutation endpoints."""
    keys = cfg.api_keys
    # Also accept env-only keys if yaml list empty
    if not keys:
        env_keys = os.getenv("ASPC_API_KEYS", "")
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    if not keys:
        # Dev mode: no keys configured — allow with warning header path
        return x_api_key or "dev"
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")
    return x_api_key


# ---- response models ----------------------------------------------------------

class AnalyzeResponse(BaseModel):
    status: str = "success"
    run_id: str
    analysis_type: str
    report: dict[str, Any]
    html_report: Optional[str] = None
    checklist: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class StreamRegisterRequest(BaseModel):
    stream_key: str
    topic: Optional[str] = None
    chart_type: Optional[str] = None
    ruleset: str = "nelson"
    meta: Optional[dict[str, Any]] = None


class GoLiveRequest(BaseModel):
    limits_version: str
    ruleset: Optional[str] = None


# ---- helpers ------------------------------------------------------------------

def _save_file(file: UploadFile) -> Path:
    content = file.file.read()
    try:
        return save_upload(
            content,
            cfg.temp_upload_dir,
            file.filename or "upload.csv",
            max_bytes=cfg.max_file_size_bytes,
            allowed_extensions=cfg.allowed_extensions,
        )
    except FileReadError as exc:
        raise HTTPException(400, str(exc)) from exc


def _load(path: Path) -> dict[str, list]:
    try:
        return load_columns(path)
    except FileReadError as exc:
        raise HTTPException(400, str(exc)) from exc


def _limits_from_stored(stored: dict[str, Any]) -> ControlLimits:
    payload = stored["payload"]
    components = {
        name: LimitSet(**comp) for name, comp in payload["components"].items()
    }
    return ControlLimits(
        chart_type=ChartType(payload["chart_type"]),
        subgroup_size=payload["subgroup_size"],
        components=components,
        sigma=payload.get("sigma"),
        source_n_points=payload.get("source_n_points"),
        notes=payload.get("notes") or {},
    )


# ---- endpoints ----------------------------------------------------------------

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "ASPC Statistical Process Control API",
        "version": "2.0.0",
        "endpoints": {
            "auth": "POST /auth/token",
            "control_chart": "POST /analyze/control-chart",
            "capability": "POST /analyze/capability",
            "msa": "POST /analyze/msa",
            "runs": "GET /runs",
            "report": "GET /reports/{run_id}",
            "streams": "GET /streams",
            "stream_register": "POST /streams/register",
            "go_live": "POST /streams/{key}/go-live",
            "ack_alert": "POST /alerts/{id}/ack",
            "ws_live": "WS /ws/live/{stream_key}",
            "stream_replay": "GET /stream/replay",
            "metrics": "GET /metrics",
            "health": "GET /health",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")


@app.get("/metrics")
async def metrics():
    if not _PROM:
        return PlainTextResponse(
            "# prometheus-client not installed\n", media_type="text/plain"
        )
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/auth/token", response_model=TokenResponse)
async def login_for_access_token(form: OAuth2PasswordRequestForm = Depends()):
    """MVP auth: accept any username if password matches ASPC_ADMIN_PASSWORD (default admin)."""
    expected = os.getenv("ASPC_ADMIN_PASSWORD", cfg.admin_password)
    if form.password != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = _create_access_token(form.username)
    return TokenResponse(access_token=token)


@app.post("/analyze/control-chart", response_model=AnalyzeResponse)
async def analyze_cc(
    file: UploadFile = File(...),
    value_col: Optional[str] = Form(None),
    subgroup_col: Optional[str] = Form(None),
    sample_size_col: Optional[str] = Form(None),
    opportunity_col: Optional[str] = Form(None),
    chart_type: Optional[str] = Form(None),
    ruleset: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    include_records: bool = Form(False),
    _user: dict = Depends(get_current_user),
):
    t0 = time.perf_counter()
    path = _save_file(file)
    columns = _load(path)
    frame = ingest(
        columns,
        value_col=value_col,
        subgroup_col=subgroup_col,
        sample_size_col=sample_size_col,
        opportunity_col=opportunity_col,
    )
    cmap = frame.column_map
    if cmap.value_col is None:
        raise HTTPException(400, "Could not detect measurement column")

    values = columns[cmap.value_col]
    ct = ChartType(chart_type) if chart_type else None
    pipeline = establish(
        values,
        subgroup_ids=columns.get(cmap.subgroup_col) if cmap.subgroup_col else None,
        sample_sizes=columns.get(cmap.sample_size_col) if cmap.sample_size_col else None,
        opportunities=columns.get(cmap.opportunity_col) if cmap.opportunity_col else None,
        chart_type=ct,
        ruleset=ruleset or cfg.ruleset,
        acf_threshold=cfg.acf_threshold,
    )
    checklist = phase1_checklist(
        pipeline,
        min_subgroups=cfg.min_phase1_points,
        phase2_enabled=False,
    )
    result = pipeline.chart
    report = SPCReport.from_chart_result(
        result,
        normality=pipeline.normality,
        autocorrelation=pipeline.autocorrelation,
        source_file=str(path),
        gates=pipeline.gates,
        checklist=checklist,
        include_records=include_records,
    )
    report_dict = report.model_dump(mode="json")

    limits_meta = {
        "source_file": str(path),
        "frozen": pipeline.frozen,
        "stopped": pipeline.stopped,
        "checklist_passed": bool(checklist.get("passed")),
        "limits_version": report.limits.version,
    }
    # Only persist freezeable limits — STOP / failed checklist must not produce
    # a go-live-eligible limits version.
    if pipeline.frozen:
        repo.save_limits(
            report_dict["limits"], report.limits.version,
            report.chart_type.value, meta=limits_meta,
        )
    run_id = repo.save_run(
        "control_chart", report_dict,
        limits_version=report.limits.version if pipeline.frozen else None,
        source_file=str(path),
        user_id=user_id or _user.get("username"),
    )

    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html = render_control_chart_html(report)
        html_path = str(save_html(html, Path(cfg.report_dir) / f"{run_id}_control_chart.html"))

    if _PROM:
        ANALYZE_LATENCY.labels(kind="control_chart").observe(time.perf_counter() - t0)

    return AnalyzeResponse(
        run_id=run_id,
        analysis_type="control_chart",
        report=report_dict,
        html_report=html_path,
        checklist=checklist,
    )


@app.post("/analyze/capability", response_model=AnalyzeResponse)
async def analyze_cap(
    file: UploadFile = File(...),
    usl: float = Form(...),
    lsl: float = Form(...),
    target: Optional[float] = Form(None),
    value_col: Optional[str] = Form(None),
    subgroup_col: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    _user: dict = Depends(get_current_user),
):
    if usl <= lsl:
        raise HTTPException(400, f"USL ({usl}) must be greater than LSL ({lsl})")

    path = _save_file(file)
    columns = _load(path)
    frame = ingest(columns, value_col=value_col, subgroup_col=subgroup_col)
    cmap = frame.column_map
    if cmap.value_col is None:
        raise HTTPException(400, "Could not detect measurement column")

    values = [float(v) for v in columns[cmap.value_col] if v is not None]
    subgroups = None
    if cmap.subgroup_col and cmap.subgroup_col in columns:
        from spc_core.limits import build_subgroups
        subgroups = build_subgroups(values, columns[cmap.subgroup_col])

    normality = check_normality(values)
    result = capability_analysis(values, usl=usl, lsl=lsl, target=target, subgroups=subgroups)
    report = CapabilityReport.from_capability(result, normality=normality, source_file=str(path))
    report_dict = report.model_dump(mode="json")

    run_id = repo.save_run(
        "capability", report_dict, source_file=str(path),
        user_id=user_id or _user.get("username"),
    )
    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html = render_capability_html(report)
        html_path = str(save_html(html, Path(cfg.report_dir) / f"{run_id}_capability.html"))

    return AnalyzeResponse(
        run_id=run_id, analysis_type="capability",
        report=report_dict, html_report=html_path,
    )


@app.post("/analyze/msa", response_model=AnalyzeResponse)
async def analyze_msa(
    file: UploadFile = File(...),
    study_type: Optional[str] = Form(None),
    method: str = Form("anova"),
    tolerance: Optional[float] = Form(None),
    part_col: Optional[str] = Form(None),
    operator_col: Optional[str] = Form(None),
    measurement_col: Optional[str] = Form(None),
    trial_col: Optional[str] = Form(None),
    reference_col: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    _user: dict = Depends(get_current_user),
):
    path = _save_file(file)
    columns = _load(path)
    frame = ingest(
        columns, value_col=measurement_col, part_col=part_col,
        operator_col=operator_col, trial_col=trial_col, reference_col=reference_col,
    )
    cmap = frame.column_map

    st = study_type
    if st is None:
        if cmap.part_col and cmap.operator_col:
            st = "gage_rr"
        elif cmap.reference_col:
            refs = set(columns[cmap.reference_col])
            st = "linearity" if len(refs) > 1 else "bias"
        elif cmap.date_col:
            st = "stability"
        else:
            st = "gage_rr"

    st_lower = st.lower().replace(" ", "_").replace("&", "")
    if st_lower in ("gage_rr", "gage_r&r", "grr"):
        if not cmap.part_col or not cmap.operator_col or not cmap.value_col:
            raise HTTPException(400, "Gage R&R needs part, operator, and measurement columns")
        fn = gage_rr_anova if method == "anova" else gage_rr_range
        result = fn(
            columns[cmap.part_col], columns[cmap.operator_col],
            columns[cmap.value_col], tolerance=tolerance,
        )
        report = MSAReport.from_gage_rr(result, source_file=str(path))
    elif st_lower == "bias":
        if not cmap.reference_col or not cmap.value_col:
            raise HTTPException(400, "Bias study needs reference and measurement columns")
        result = bias_study(columns[cmap.value_col], columns[cmap.reference_col])
        report = MSAReport.from_bias(result, source_file=str(path))
    elif st_lower == "linearity":
        if not cmap.reference_col or not cmap.value_col:
            raise HTTPException(400, "Linearity study needs reference and measurement columns")
        result = linearity_study(columns[cmap.value_col], columns[cmap.reference_col])
        report = MSAReport.from_linearity(result, source_file=str(path))
    elif st_lower == "stability":
        if not cmap.value_col:
            raise HTTPException(400, "Stability study needs a measurement column")
        result = stability_study(columns[cmap.value_col])
        report = MSAReport.from_stability(result, source_file=str(path))
    else:
        raise HTTPException(400, f"Unknown study_type: {study_type}")

    report_dict = report.model_dump(mode="json")
    run_id = repo.save_run(
        "msa", report_dict, source_file=str(path),
        user_id=user_id or _user.get("username"),
    )
    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html = render_msa_html(report)
        html_path = str(save_html(html, Path(cfg.report_dir) / f"{run_id}_msa.html"))

    return AnalyzeResponse(
        run_id=run_id, analysis_type="msa",
        report=report_dict, html_report=html_path,
    )


@app.get("/runs")
async def list_runs(
    analysis_type: Optional[str] = None,
    limit: int = Query(50, le=500),
    _user: dict = Depends(get_current_user),
):
    return {"runs": repo.list_runs(analysis_type=analysis_type, limit=limit)}


@app.get("/runs/{run_id}")
async def get_run(run_id: str, _user: dict = Depends(get_current_user)):
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run not found: {run_id}")
    return run


@app.get("/reports/{run_id}")
async def get_report_html(run_id: str):
    """Serve a previously generated HTML report, or rebuild from stored JSON."""
    path = Path(cfg.report_dir)
    for suffix in ("_control_chart.html", "_capability.html", "_msa.html"):
        candidate = path / f"{run_id}{suffix}"
        if candidate.exists():
            return HTMLResponse(candidate.read_text(encoding="utf-8"))

    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run not found: {run_id}")
    report = run["report"]
    html = f"""<!DOCTYPE html><html><head><title>Run {run_id}</title></head>
<body><h1>{run['analysis_type']}</h1>
<pre>{json.dumps(report, indent=2, default=str)}</pre></body></html>"""
    return HTMLResponse(html)


@app.post("/streams/register")
async def register_stream(
    body: StreamRegisterRequest,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(get_current_user),
):
    if not hasattr(repo, "register_stream"):
        raise HTTPException(
            501,
            "Stream registry requires TimescaleDB backend (persistence.backend=timescale)",
        )
    key = repo.register_stream(
        body.stream_key,
        topic=body.topic,
        chart_type=body.chart_type,
        ruleset=body.ruleset,
        active=False,
        meta=body.meta,
    )
    repo.save_audit(
        "stream_register",
        {"stream_key": key, "topic": body.topic},
        user_id=_user.get("username"),
    )
    return {"stream_key": key, "active": False}


@app.post("/streams/{stream_key}/go-live")
async def go_live(
    stream_key: str,
    body: GoLiveRequest,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(get_current_user),
):
    if not hasattr(repo, "register_stream"):
        raise HTTPException(501, "Stream registry requires TimescaleDB backend")
    stored = repo.get_limits(body.limits_version)
    if not stored:
        raise HTTPException(404, f"Limits version not found: {body.limits_version}")
    meta = stored.get("meta") or {}
    if meta.get("frozen") is False or meta.get("stopped") is True:
        raise HTTPException(
            409,
            "Cannot go-live: Phase I limits were not frozen (STOP gate fired). "
            "Re-run Phase I after resolving stop conditions.",
        )
    if meta.get("checklist_passed") is False:
        raise HTTPException(
            409,
            "Cannot go-live: Phase I checklist did not pass. "
            "Resolve checklist items before enabling Phase II.",
        )
    ruleset = body.ruleset or cfg.ruleset
    repo.register_stream(
        stream_key,
        limits_version=body.limits_version,
        chart_type=stored.get("chart_type"),
        ruleset=ruleset,
        active=True,
    )
    repo.save_audit(
        "stream_go_live",
        {"stream_key": stream_key, "limits_version": body.limits_version},
        user_id=_user.get("username"),
    )
    return {
        "stream_key": stream_key,
        "limits_version": body.limits_version,
        "active": True,
        "ruleset": ruleset,
    }


@app.get("/streams")
async def list_streams(
    active_only: bool = False,
    _user: dict = Depends(get_current_user),
):
    if not hasattr(repo, "list_streams"):
        return {"streams": []}
    return {"streams": repo.list_streams(active_only=active_only)}


@app.post("/alerts/{event_id}/ack")
async def ack_alert(
    event_id: int,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(get_current_user),
):
    if not hasattr(repo, "ack_alert"):
        raise HTTPException(501, "Alert ack requires TimescaleDB backend")
    updated = repo.ack_alert(event_id, acked_by=_user.get("username"))
    if updated is None:
        raise HTTPException(404, f"Alert not found: {event_id}")
    return updated


@app.websocket("/ws/live/{stream_key}")
async def ws_live(websocket: WebSocket, stream_key: str):
    """Subscribe to Redis channel ``spc:live:{stream_key}`` and forward messages."""
    await websocket.accept()
    channel = f"spc:live:{stream_key}"
    try:
        import asyncio

        import redis.asyncio as aioredis
    except ImportError:
        await websocket.send_json(
            {"error": "redis package required for live websocket; pip install redis"}
        )
        await websocket.close()
        return

    client = aioredis.from_url(cfg.redis_url, decode_responses=True)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(channel)
        await websocket.send_json({"event": "subscribed", "channel": channel})
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg.get("type") == "message":
                data = msg["data"]
                try:
                    payload = json.loads(data) if isinstance(data, str) else data
                except json.JSONDecodeError:
                    payload = {"raw": data}
                await websocket.send_json(payload)
            else:
                # Keepalive / detect client disconnect
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await client.close()


@app.get("/stream/replay")
async def stream_replay(
    file_path: str = Query(..., description="Path to a CSV already on the server"),
    value_col: str = Query("measurement"),
    limits_version: str = Query(..., description="Frozen Phase I limits version"),
    _user: dict = Depends(get_current_user),
):
    """SSE stream of OOC signals while replaying a CSV against frozen limits."""
    stored = repo.get_limits(limits_version)
    if not stored:
        raise HTTPException(404, f"Limits version not found: {limits_version}")

    limits = _limits_from_stored(stored)

    if not Path(file_path).exists():
        raise HTTPException(404, f"File not found: {file_path}")

    def event_gen():
        source = FileReplaySource(file_path, value_col=value_col, delay_s=0.0)
        signals = stream_evaluate(source, limits, ruleset=cfg.ruleset)
        for s in signals:
            yield f"data: {s.model_dump_json()}\n\n"
        yield 'data: {"event": "done"}\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def run() -> None:
    """Entry point for the ``aspc-api`` console script."""
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
