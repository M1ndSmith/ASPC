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

import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from adapters.factory import get_repository
from adapters.io_files import FileReadError, load_columns, resolve_under, save_upload_stream
from adapters.protocols import StreamingOpsRepository
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
    checklist_ready_for_golive,
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

_ANALYZE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aspc-analyze")

# Rate limiting (slowapi) — required when auth is enabled (checked at lifespan)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    _RATE_LIMIT = True
except ImportError:  # pragma: no cover
    Limiter = None  # type: ignore[misc, assignment]
    _rate_limit_exceeded_handler = None  # type: ignore[misc, assignment]
    RateLimitExceeded = Exception  # type: ignore[misc, assignment]
    get_remote_address = None  # type: ignore[misc, assignment]
    _RATE_LIMIT = False

# Password hashing — bcrypt is required (no plaintext fallback).
try:
    import bcrypt as _bcrypt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "bcrypt is required for ASPC auth. Install with: pip install 'aspc[apps]' "
        "or: pip install bcrypt"
    ) from exc

_admin_password_hash: bytes | None = None


def _ensure_password_hash() -> bytes:
    global _admin_password_hash
    if _admin_password_hash is not None:
        return _admin_password_hash
    raw = cfg.admin_password
    if raw.startswith(("$2a$", "$2b$", "$2y$")):
        _admin_password_hash = raw.encode("utf-8")
    else:
        _admin_password_hash = _bcrypt.hashpw(raw.encode("utf-8"), _bcrypt.gensalt())
    return _admin_password_hash


def _verify_password(plain: str) -> bool:
    stored = _ensure_password_hash()
    try:
        return bool(_bcrypt.checkpw(plain.encode("utf-8"), stored))
    except Exception:
        return False


def _startup_security_checks() -> None:
    """Refuse insecure production defaults unless ASPC_DEV_INSECURE=1.

    Weak secrets are refused even when auth is disabled — only ``dev_insecure``
    bypasses these checks.
    """
    _ensure_password_hash()
    if cfg.dev_insecure:
        return
    if cfg.jwt_secret == "change-me-in-production":
        raise RuntimeError(
            "ASPC_JWT_SECRET is still the default 'change-me-in-production'. "
            "Set a strong secret, or set ASPC_DEV_INSECURE=1 for local development only."
        )
    if cfg.admin_password == "admin":
        raise RuntimeError(
            "ASPC_ADMIN_PASSWORD is still the default 'admin'. "
            "Set a strong password, or set ASPC_DEV_INSECURE=1 for local development only."
        )
    if not cfg.api_keys:
        raise RuntimeError(
            "ASPC_API_KEYS is empty. Set at least one API key for stream mutations, "
            "or set ASPC_DEV_INSECURE=1 for local development only."
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _startup_security_checks()
    if not _RATE_LIMIT and cfg.auth_enabled and not cfg.dev_insecure:
        raise RuntimeError(
            "slowapi is required when auth is enabled. Install with: pip install slowapi "
            "or set ASPC_DEV_INSECURE=1 for local development only."
        )
    yield


app = FastAPI(
    title="ASPC — Statistical Process Control API",
    description=(
        "Correct, tested SPC core with batch analysis and Phase II streaming. "
        "Optional JWT roles (admin/analyst/operator) and tenant_id for multi-user demos."
    ),
    version="2.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)

limiter = None
if _RATE_LIMIT:
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    REQUESTS = Counter("aspc_http_requests_total", "HTTP requests", ["method", "path", "status"])
    ANALYZE_LATENCY = Histogram("aspc_analyze_seconds", "Analyze endpoint latency", ["kind"])
    _PROM = True
except ImportError:  # pragma: no cover
    _PROM = False


# ---- auth helpers -------------------------------------------------------------

_VALID_ROLES = frozenset({"admin", "analyst", "operator"})


def _create_access_token(
    subject: str,
    *,
    role: str | None = None,
    tenant_id: str | None = None,
) -> str:
    try:
        from jose import jwt
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "python-jose required for JWT auth") from exc
    expire = datetime.now(UTC) + timedelta(minutes=cfg.jwt_expire_minutes)
    role = role or cfg.default_role or "admin"
    if role not in _VALID_ROLES:
        role = "operator"
    payload: dict[str, Any] = {"sub": subject, "exp": expire, "role": role}
    tid = tenant_id if tenant_id is not None else cfg.default_tenant_id
    if tid:
        payload["tenant_id"] = str(tid)
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        from jose import JWTError, jwt
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "python-jose required for JWT auth") from exc
    try:
        data = jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(401, "Invalid or expired token") from exc
    username = data.get("sub")
    if not username:
        raise HTTPException(401, "Invalid token")
    role = data.get("role") or cfg.default_role or "admin"
    if role not in _VALID_ROLES:
        role = "operator"
    out: dict[str, Any] = {"username": username, "auth": "jwt", "role": role}
    if data.get("tenant_id"):
        out["tenant_id"] = data["tenant_id"]
    elif cfg.default_tenant_id:
        out["tenant_id"] = cfg.default_tenant_id
    return out


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """Validate Bearer JWT. When auth is disabled, return anonymous user."""
    if not cfg.auth_enabled:
        out: dict[str, Any] = {
            "username": "anonymous",
            "auth": "disabled",
            "role": "admin",
        }
        if cfg.default_tenant_id:
            out["tenant_id"] = cfg.default_tenant_id
        return out
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(credentials.credentials)


def require_roles(*roles: str):
    """Dependency factory: require JWT role in ``roles`` (admin always allowed)."""
    allowed = frozenset(roles) | frozenset({"admin"})

    def _dep(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        role = user.get("role") or "operator"
        if role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{role}' cannot perform this action (need one of {sorted(allowed)})",
            )
        return user

    return _dep


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Require X-API-Key for ingest / stream mutation endpoints.

    Fail-closed when no keys are configured, unless ``ASPC_DEV_INSECURE=1``.
    """
    keys = cfg.api_keys
    if not keys:
        env_keys = os.getenv("ASPC_API_KEYS", "")
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    if not keys:
        if cfg.dev_insecure:
            return x_api_key or "dev"
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "API keys not configured. Set ASPC_API_KEYS, or ASPC_DEV_INSECURE=1 for local dev.",
        )
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")
    return x_api_key


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "Invalid run_id")
    return run_id


# ---- response models ----------------------------------------------------------

class AnalyzeResponse(BaseModel):
    status: str = "success"
    run_id: str
    analysis_type: str
    report: dict[str, Any]
    html_report: str | None = None
    checklist: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    checks: dict[str, str] | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class StreamRegisterRequest(BaseModel):
    stream_key: str
    topic: str | None = None
    chart_type: str | None = None
    ruleset: str = "nelson"
    meta: dict[str, Any] | None = None


class GoLiveRequest(BaseModel):
    limits_version: str
    ruleset: str | None = None


# ---- helpers ------------------------------------------------------------------

def _save_file(file: UploadFile) -> Path:
    try:
        return save_upload_stream(
            file.file,
            cfg.temp_upload_dir,
            file.filename or "upload.csv",
            max_bytes=cfg.max_file_size_bytes,
            allowed_extensions=cfg.allowed_extensions,
        )
    except FileReadError as exc:
        raise HTTPException(400, str(exc)) from exc


def _maybe_html(render_fn, report: Any, filename: str) -> str | None:
    if not cfg.config["reports"]["auto_generate"]:
        return None
    try:
        return str(save_html(render_fn(report), Path(cfg.report_dir) / filename))
    except (ImportError, OSError):
        return None


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
    checks: dict[str, str] = {"api": "ok"}
    try:
        if hasattr(repo, "list_runs"):
            repo.list_runs(limit=1)
        checks["persistence"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["persistence"] = f"error: {exc}"
    try:
        import redis as redis_lib

        r = redis_lib.Redis.from_url(cfg.redis_url, socket_connect_timeout=1)
        r.ping()
        r.close()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"unavailable: {exc}"

    degraded = any(
        not v.startswith(("ok", "unavailable")) for k, v in checks.items() if k != "api"
    )
    return HealthResponse(status="degraded" if degraded else "healthy", checks=checks)


@app.get("/metrics")
async def metrics(_user: dict = Depends(get_current_user)):
    if not _PROM:
        return PlainTextResponse(
            "# prometheus-client not installed\n", media_type="text/plain"
        )
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


def _rate_limit(limit: str):
    """Apply slowapi limit when available; otherwise no-op."""
    if _RATE_LIMIT and limiter is not None:
        return limiter.limit(limit)

    def _noop(fn):
        return fn

    return _noop


@app.post("/auth/token", response_model=TokenResponse)
@_rate_limit("10/minute")
async def login_for_access_token(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
):
    """Issue a JWT for the admin user or an optional demo user from config.auth.users."""
    role = cfg.default_role or "admin"
    tenant_id = cfg.default_tenant_id
    if form.username == cfg.admin_username:
        if not _verify_password(form.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        demo = next(
            (u for u in cfg.auth_users if isinstance(u, dict) and u.get("username") == form.username),
            None,
        )
        if demo is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raw = str(demo.get("password") or "")
        ok = False
        try:
            if raw.startswith(("$2a$", "$2b$", "$2y$")):
                ok = bool(_bcrypt.checkpw(form.password.encode("utf-8"), raw.encode("utf-8")))
            else:
                ok = form.password == raw
        except Exception:
            ok = False
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        role = str(demo.get("role") or "operator")
        tenant_id = demo.get("tenant_id")
    token = _create_access_token(form.username, role=role, tenant_id=tenant_id)
    return TokenResponse(access_token=token)


@app.get("/auth/me")
async def auth_me(_user: dict = Depends(get_current_user)):
    """Return current JWT identity (username, role, optional tenant_id)."""
    return {
        "username": _user.get("username"),
        "role": _user.get("role"),
        "tenant_id": _user.get("tenant_id"),
        "auth": _user.get("auth"),
    }


@app.post("/analyze/control-chart", response_model=AnalyzeResponse)
@_rate_limit("20/minute")
async def analyze_cc(
    request: Request,
    file: UploadFile = File(...),
    value_col: str | None = Form(None),
    subgroup_col: str | None = Form(None),
    sample_size_col: str | None = Form(None),
    opportunity_col: str | None = Form(None),
    chart_type: str | None = Form(None),
    ruleset: str | None = Form(None),
    valid_range_min: float | None = Form(None),
    valid_range_max: float | None = Form(None),
    include_records: bool = Form(False),
    msa_file: UploadFile | None = File(None),
    msa_tolerance: float | None = Form(None),
    _user: dict = Depends(get_current_user),
):
    t0 = time.perf_counter()
    try:
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

        msa_kwargs: dict[str, Any] = {}
        if msa_file is not None and msa_file.filename:
            msa_path = _save_file(msa_file)
            msa_cols = _load(msa_path)
            msa_frame = ingest(msa_cols)
            msa_map = msa_frame.column_map
            if (
                msa_map.value_col is None
                or msa_map.part_col is None
                or msa_map.operator_col is None
            ):
                raise HTTPException(
                    400,
                    "MSA file needs measurement, part, and operator columns",
                )
            msa_kwargs = {
                "msa_parts": msa_cols[msa_map.part_col],
                "msa_operators": msa_cols[msa_map.operator_col],
                "msa_measurements": msa_cols[msa_map.value_col],
                "msa_tolerance": msa_tolerance,
            }

        values = columns[cmap.value_col]
        ct = ChartType(chart_type) if chart_type else None
        valid_range = None
        if valid_range_min is not None or valid_range_max is not None:
            if valid_range_min is None or valid_range_max is None:
                raise HTTPException(400, "Both valid_range_min and valid_range_max are required")
            valid_range = (valid_range_min, valid_range_max)

        def _run_establish():
            return establish(
                values,
                subgroup_ids=columns.get(cmap.subgroup_col) if cmap.subgroup_col else None,
                sample_sizes=columns.get(cmap.sample_size_col) if cmap.sample_size_col else None,
                opportunities=columns.get(cmap.opportunity_col) if cmap.opportunity_col else None,
                chart_type=ct,
                ruleset=ruleset or cfg.ruleset,
                acf_threshold=cfg.acf_threshold,
                valid_range=valid_range,
                **msa_kwargs,
            )

        import asyncio

        pipeline = await asyncio.get_running_loop().run_in_executor(
            _ANALYZE_POOL, _run_establish
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
        # Exclude phase2_enabled — that item flips only after go-live itself.
        "checklist_passed": checklist_ready_for_golive(checklist),
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
        user_id=_user.get("username"),
    )

    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html_path = _maybe_html(
            render_control_chart_html, report, f"{run_id}_control_chart.html"
        )

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
@_rate_limit("20/minute")
async def analyze_cap(
    request: Request,
    file: UploadFile = File(...),
    usl: float = Form(...),
    lsl: float = Form(...),
    target: float | None = Form(None),
    value_col: str | None = Form(None),
    subgroup_col: str | None = Form(None),
    _user: dict = Depends(get_current_user),
):
    if usl <= lsl:
        raise HTTPException(400, f"USL ({usl}) must be greater than LSL ({lsl})")

    try:
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
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    report = CapabilityReport.from_capability(result, normality=normality, source_file=str(path))
    report_dict = report.model_dump(mode="json")

    run_id = repo.save_run(
        "capability", report_dict, source_file=str(path),
        user_id=_user.get("username"),
    )
    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html_path = _maybe_html(render_capability_html, report, f"{run_id}_capability.html")

    return AnalyzeResponse(
        run_id=run_id, analysis_type="capability",
        report=report_dict, html_report=html_path,
    )


@app.post("/analyze/msa", response_model=AnalyzeResponse)
@_rate_limit("20/minute")
async def analyze_msa(
    request: Request,
    file: UploadFile = File(...),
    study_type: str | None = Form(None),
    method: str = Form("anova"),
    tolerance: float | None = Form(None),
    part_col: str | None = Form(None),
    operator_col: str | None = Form(None),
    measurement_col: str | None = Form(None),
    trial_col: str | None = Form(None),
    reference_col: str | None = Form(None),
    _user: dict = Depends(get_current_user),
):
    try:
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
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    report_dict = report.model_dump(mode="json")
    run_id = repo.save_run(
        "msa", report_dict, source_file=str(path),
        user_id=_user.get("username"),
    )
    html_path = None
    if cfg.config["reports"]["auto_generate"]:
        html_path = _maybe_html(render_msa_html, report, f"{run_id}_msa.html")

    return AnalyzeResponse(
        run_id=run_id, analysis_type="msa",
        report=report_dict, html_report=html_path,
    )


class ContinuousMSARequest(BaseModel):
    measured: list[float]
    reference: list[float]
    tolerance: float = 1.0
    alpha: float = 0.2


@app.post("/analyze/msa-continuous")
@_rate_limit("20/minute")
async def analyze_msa_continuous(
    request: Request,
    body: ContinuousMSARequest,
    _user: dict = Depends(get_current_user),
):
    """Evaluate continuous MSA drift (EWMA bias + rolling R) on paired reference injections."""
    from spc_core.msa_stream import ContinuousMSA

    if len(body.measured) != len(body.reference):
        raise HTTPException(400, "measured and reference must have the same length")
    if not body.measured:
        raise HTTPException(400, "measured/reference must be non-empty")
    try:
        monitor = ContinuousMSA(tolerance=body.tolerance, alpha=body.alpha)
        for m, r in zip(body.measured, body.reference, strict=True):
            monitor.observe_reference(m, r)
        summary = monitor.summary()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "success", "summary": summary}


@app.get("/runs")
async def list_runs(
    analysis_type: str | None = None,
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
async def get_report_html(run_id: str, _user: dict = Depends(get_current_user)):
    """Serve a previously generated HTML report, or rebuild from stored JSON."""
    safe_id = _safe_run_id(run_id)
    report_root = Path(cfg.report_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("_control_chart.html", "_capability.html", "_msa.html"):
        try:
            candidate = resolve_under(report_root, f"{safe_id}{suffix}")
        except FileReadError as exc:
            raise HTTPException(400, str(exc)) from exc
        if candidate.exists() and candidate.is_file():
            return HTMLResponse(candidate.read_text(encoding="utf-8"))

    run = repo.get_run(safe_id)
    if not run:
        raise HTTPException(404, f"Run not found: {safe_id}")
    report = run["report"]
    safe_json = html.escape(json.dumps(report, indent=2, default=str))
    safe_type = html.escape(str(run["analysis_type"]))
    safe_title = html.escape(safe_id)
    html_body = f"""<!DOCTYPE html><html><head><title>Run {safe_title}</title></head>
<body><h1>{safe_type}</h1>
<pre>{safe_json}</pre></body></html>"""
    return HTMLResponse(html_body)


@app.post("/streams/register")
async def register_stream(
    body: StreamRegisterRequest,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(require_roles("analyst", "admin")),
):
    if not isinstance(repo, StreamingOpsRepository):
        raise HTTPException(
            501,
            "Stream registry requires TimescaleDB backend (persistence.backend=timescale)",
        )
    meta = dict(body.meta or {})
    if _user.get("tenant_id") and "tenant_id" not in meta:
        meta["tenant_id"] = _user["tenant_id"]
    key = repo.register_stream(
        body.stream_key,
        topic=body.topic,
        chart_type=body.chart_type,
        ruleset=body.ruleset,
        active=False,
        meta=meta or None,
    )
    # Persist tenant_id column when supported
    if _user.get("tenant_id") and hasattr(repo, "set_stream_tenant"):
        try:
            repo.set_stream_tenant(key, _user["tenant_id"])  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    repo.save_audit(
        "stream_register",
        {"stream_key": key, "topic": body.topic, "tenant_id": _user.get("tenant_id")},
        user_id=_user.get("username"),
    )
    return {"stream_key": key, "active": False, "tenant_id": _user.get("tenant_id")}


@app.post("/streams/{stream_key}/go-live")
async def go_live(
    stream_key: str,
    body: GoLiveRequest,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(require_roles("analyst", "admin")),
):
    if not isinstance(repo, StreamingOpsRepository):
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
    stream_meta: dict[str, Any] = {}
    if _user.get("tenant_id"):
        stream_meta["tenant_id"] = _user["tenant_id"]
    if cfg.webhook_url:
        stream_meta.setdefault("webhook_url", cfg.webhook_url)
    repo.register_stream(
        stream_key,
        limits_version=body.limits_version,
        chart_type=stored.get("chart_type"),
        ruleset=ruleset,
        active=True,
        meta=stream_meta or None,
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
        "tenant_id": _user.get("tenant_id"),
    }


@app.get("/streams")
async def list_streams(
    active_only: bool = False,
    _user: dict = Depends(get_current_user),
):
    if not isinstance(repo, StreamingOpsRepository):
        return {"streams": []}
    streams = repo.list_streams(active_only=active_only)
    tid = _user.get("tenant_id")
    if tid:
        streams = [
            s
            for s in streams
            if (s.get("tenant_id") == tid)
            or ((s.get("meta") or {}).get("tenant_id") == tid)
            or (s.get("tenant_id") is None and not (s.get("meta") or {}).get("tenant_id"))
        ]
    return {"streams": streams}


@app.post("/alerts/{event_id}/ack")
async def ack_alert(
    event_id: int,
    _key: str = Depends(require_api_key),
    _user: dict = Depends(get_current_user),
):
    if not isinstance(repo, StreamingOpsRepository):
        raise HTTPException(501, "Alert ack requires TimescaleDB backend")
    updated = repo.ack_alert(event_id, acked_by=_user.get("username"))
    if updated is None:
        raise HTTPException(404, f"Alert not found: {event_id}")
    return updated


@app.websocket("/ws/live/{stream_key}")
async def ws_live(websocket: WebSocket, stream_key: str):
    """Subscribe to Redis channel ``spc:live:{stream_key}`` and forward messages.

    When auth is enabled, the client must send a first JSON message
    ``{"type": "auth", "token": "<jwt>"}`` after the handshake (no token in the URL).
    """
    await websocket.accept()
    user: dict[str, Any] = {"role": "admin"}
    if cfg.auth_enabled:
        try:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            if not isinstance(payload, dict) or payload.get("type") != "auth":
                await websocket.close(code=1008, reason="Expected auth message")
                return
            token = payload.get("token")
            if not token or not isinstance(token, str):
                await websocket.close(code=1008, reason="Missing token")
                return
            user = _decode_token(token)
        except (WebSocketDisconnect, json.JSONDecodeError, HTTPException):
            await websocket.close(code=1008, reason="Invalid or expired token")
            return

    tid = user.get("tenant_id") if (cfg.redis_tenant_prefix or user.get("tenant_id")) else None
    if tid:
        channel = f"spc:live:{tid}:{stream_key}"
    else:
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
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                except TimeoutError:
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
    file_path: str = Query(..., description="CSV filename under the upload directory"),
    value_col: str = Query("measurement"),
    limits_version: str = Query(..., description="Frozen Phase I limits version"),
    _user: dict = Depends(get_current_user),
):
    """SSE stream of OOC signals while replaying a CSV against frozen limits.

    ``file_path`` must resolve inside the configured upload directory.
    """
    stored = repo.get_limits(limits_version)
    if not stored:
        raise HTTPException(404, f"Limits version not found: {limits_version}")

    limits = _limits_from_stored(stored)

    try:
        resolved = resolve_under(cfg.temp_upload_dir, file_path)
    except FileReadError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    def event_gen():
        source = FileReplaySource(str(resolved), value_col=value_col, delay_s=0.0)
        signals = stream_evaluate(source, limits, ruleset=cfg.ruleset)
        for s in signals:
            yield f"data: {s.model_dump_json()}\n\n"
        yield 'data: {"event": "done"}\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---- onboarding / explain / lab / excel / ops ---------------------------------

class ExplainRequest(BaseModel):
    signal: dict[str, Any]
    limits_version: str | None = None
    gates: list[dict[str, Any]] | None = None
    checklist: dict[str, Any] | None = None


class CounterfactualRequest(BaseModel):
    """Sandbox re-establish without mutating stored frozen limits."""
    values: list[float]
    ruleset: str = "nelson"
    chart_type: str | None = None
    valid_range_min: float | None = None
    valid_range_max: float | None = None


@app.get("/onboarding/sample")
async def onboarding_sample(
    dataset: str = Query("spc_individual_in_control"),
    _user: dict = Depends(get_current_user),
):
    """Return a sample CSV body from ``sample_data`` for the onboarding wizard."""
    import csv
    import io

    from sample_data import DATASET_CATALOG, get_dataset

    known = sorted(
        {
            "spc_individual_in_control",
            "spc_individual_out_of_control",
            "spc_subgroup_data",
            "spc_c_chart_data",
            "spc_p_chart_data",
            "spc_np_chart_data",
            "spc_u_chart_data",
            "msa_gage_rr_excellent",
            "msa_gage_rr_poor",
            "msa_bias_study",
            "msa_linearity_study",
            "msa_stability_study",
            "capability_excellent",
            "capability_skewed_data",
            "capability_off_center",
            "capability_high_variation",
        }
        | set(DATASET_CATALOG.keys() if isinstance(DATASET_CATALOG, dict) else [])
    )
    try:
        cols = get_dataset(dataset)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown dataset: {dataset}. Known: {known}") from exc
    buf = io.StringIO()
    keys = list(cols.keys())
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    n = len(next(iter(cols.values())))
    for i in range(n):
        writer.writerow({k: cols[k][i] for k in keys})
    return {
        "dataset": dataset,
        "filename": f"{dataset}.csv",
        "csv": buf.getvalue(),
        "catalog": known,
    }


@app.post("/onboarding/demo-stream")
async def onboarding_demo_stream(
    limits_version: str = Query(...),
    stream_key: str = Query("demo-line-1"),
    _key: str = Depends(require_api_key),
    _user: dict = Depends(require_roles("analyst", "admin")),
):
    """Register + go-live a demo stream against a frozen limits version (Timescale only)."""
    body = GoLiveRequest(limits_version=limits_version)
    await register_stream(
        StreamRegisterRequest(stream_key=stream_key, meta={"demo": True}),
        _key=_key,
        _user=_user,
    )
    return await go_live(stream_key, body, _key=_key, _user=_user)


@app.post("/analyze/explain")
async def analyze_explain(
    body: ExplainRequest,
    _user: dict = Depends(get_current_user),
):
    """Deterministic Explainable SPC Copilot — structured why for one signal."""
    from spc_core.explain import explain_signal

    limits = None
    if body.limits_version:
        stored = repo.get_limits(body.limits_version)
        if stored:
            limits = stored.get("payload") or stored
    return explain_signal(
        body.signal,
        limits_version=body.limits_version,
        limits=limits,
        gates=body.gates,
        checklist=body.checklist,
    )


@app.post("/analyze/counterfactual")
async def analyze_counterfactual(
    body: CounterfactualRequest,
    _user: dict = Depends(get_current_user),
):
    """Sandbox establish — never writes limits to the repository."""
    from spc_core import ChartType, establish, phase1_checklist
    from spc_core.explain import explain_signals

    kwargs: dict[str, Any] = {"ruleset": body.ruleset}
    if body.chart_type:
        kwargs["chart_type"] = ChartType(body.chart_type)
    if body.valid_range_min is not None and body.valid_range_max is not None:
        kwargs["valid_range"] = (body.valid_range_min, body.valid_range_max)
    pipe = establish(body.values, **kwargs)
    checklist = phase1_checklist(pipe)
    report = pipe.chart
    signals = list(report.signals or [])
    return {
        "frozen": pipe.frozen,
        "limits_version": report.limits.version if report.limits else None,
        "gates": [
            g.model_dump() if hasattr(g, "model_dump") else {"step": g.step, "status": g.status, "reason": g.reason}
            for g in (pipe.gates or [])
        ],
        "checklist": checklist,
        "explanations": explain_signals(
            signals,
            limits_version=report.limits.version if report.limits else None,
            limits=report.limits,
        ),
        "note": "Counterfactual only — live frozen limits were not modified.",
    }


@app.get("/limits/{version_a}/diff/{version_b}")
async def limits_diff(
    version_a: str,
    version_b: str,
    _user: dict = Depends(get_current_user),
):
    from spc_core.explain import diff_limits

    a = repo.get_limits(version_a)
    b = repo.get_limits(version_b)
    if not a:
        raise HTTPException(404, f"Limits version not found: {version_a}")
    if not b:
        raise HTTPException(404, f"Limits version not found: {version_b}")
    pa = a.get("payload") or a
    pb = b.get("payload") or b
    if "version" not in pa:
        pa = {**pa, "version": version_a}
    if "version" not in pb:
        pb = {**pb, "version": version_b}
    return diff_limits(pa, pb)


@app.get("/lab/cases")
async def lab_cases(_user: dict = Depends(get_current_user)):
    from resilience_data import load_manifest

    cases = load_manifest()
    return {
        "cases": [
            {
                "id": c.get("id"),
                "category": c.get("category"),
                "entry": c.get("entry"),
                "description": c.get("description") or c.get("title"),
            }
            for c in cases
        ]
    }


@app.post("/lab/cases/{case_id}/run")
async def lab_run_case(case_id: str, _user: dict = Depends(require_roles("analyst", "admin"))):
    from resilience_data import load_manifest
    from resilience_data.runner import run_case

    cases = {c["id"]: c for c in load_manifest()}
    if case_id not in cases:
        raise HTTPException(404, f"Unknown case: {case_id}")
    return run_case(cases[case_id])


@app.get("/runs/{run_id}/export.xlsx")
async def export_run_xlsx(run_id: str, _user: dict = Depends(get_current_user)):
    from fastapi.responses import Response

    from adapters.excel import spc_report_to_xlsx_bytes

    safe = _safe_run_id(run_id)
    detail = repo.get_run(safe)
    if not detail:
        raise HTTPException(404, f"Run not found: {run_id}")
    report = detail.get("report") if isinstance(detail, dict) else None
    if not isinstance(report, dict):
        raise HTTPException(400, "Run has no SPC report to export")
    try:
        data = spc_report_to_xlsx_bytes(report)
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from exc
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe}.xlsx"'},
    )


@app.get("/ops/summary")
async def ops_summary(_user: dict = Depends(get_current_user)):
    """Multi-stream ops snapshot for the overview dashboard."""
    streams: list[dict[str, Any]] = []
    if isinstance(repo, StreamingOpsRepository):
        streams = repo.list_streams(active_only=False)
        tid = _user.get("tenant_id")
        if tid:
            streams = [
                s
                for s in streams
                if s.get("tenant_id") == tid
                or (s.get("meta") or {}).get("tenant_id") == tid
                or s.get("tenant_id") is None
            ]
    active = [s for s in streams if s.get("active")]
    recent = repo.list_runs(limit=20) if hasattr(repo, "list_runs") else []
    checklist_debt = 0
    for r in recent:
        # heuristic: control-chart runs without limits_version
        if r.get("analysis_type") == "control_chart" and not r.get("limits_version"):
            checklist_debt += 1
    return {
        "streams_total": len(streams),
        "streams_active": len(active),
        "streams": [
            {
                "stream_key": s.get("stream_key"),
                "active": s.get("active"),
                "limits_version": s.get("limits_version"),
                "chart_type": s.get("chart_type"),
                "tenant_id": s.get("tenant_id") or (s.get("meta") or {}).get("tenant_id"),
            }
            for s in streams
        ],
        "recent_runs": len(recent),
        "checklist_debt": checklist_debt,
    }


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
