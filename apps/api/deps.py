"""Shared FastAPI dependencies and app-level state for ASPC API routers."""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from adapters.factory import get_repository
from apps.config import get_config

cfg = get_config()
repo = get_repository(cfg)

_bearer = HTTPBearer(auto_error=False)

try:
    import bcrypt as _bcrypt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "bcrypt is required for ASPC auth. Install with: pip install 'aspc[apps]' "
        "or: pip install bcrypt"
    ) from exc

_admin_password_hash: bytes | None = None
_user_password_hashes: dict[str, bytes] = {}

VALID_ROLES = frozenset({"admin", "analyst", "operator"})


def ensure_password_hash() -> bytes:
    global _admin_password_hash
    if _admin_password_hash is not None:
        return _admin_password_hash
    raw = cfg.admin_password
    if raw.startswith(("$2a$", "$2b$", "$2y$")):
        _admin_password_hash = raw.encode("utf-8")
    else:
        _admin_password_hash = _bcrypt.hashpw(raw.encode("utf-8"), _bcrypt.gensalt())
    return _admin_password_hash


def _hash_password(raw: str) -> bytes:
    if raw.startswith(("$2a$", "$2b$", "$2y$")):
        return raw.encode("utf-8")
    return _bcrypt.hashpw(raw.encode("utf-8"), _bcrypt.gensalt())


def verify_password(plain: str, *, username: str | None = None) -> bool:
    """Verify against admin password or an optional demo user entry."""
    if username and username != cfg.admin_username:
        user = find_user(username)
        if user is None:
            return False
        stored = _user_password_hashes.get(username)
        if stored is None:
            stored = _hash_password(str(user.get("password") or ""))
            _user_password_hashes[username] = stored
        try:
            return bool(_bcrypt.checkpw(plain.encode("utf-8"), stored))
        except Exception:
            return False
    stored = ensure_password_hash()
    try:
        return bool(_bcrypt.checkpw(plain.encode("utf-8"), stored))
    except Exception:
        return False


def find_user(username: str) -> dict[str, Any] | None:
    for u in cfg.auth_users:
        if isinstance(u, dict) and u.get("username") == username:
            return u
    if username == cfg.admin_username:
        return {
            "username": cfg.admin_username,
            "role": cfg.default_role or "admin",
            "tenant_id": cfg.default_tenant_id,
        }
    return None


def create_access_token(
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
    user = find_user(subject) or {}
    role = role or str(user.get("role") or cfg.default_role or "admin")
    if role not in VALID_ROLES:
        role = "operator"
    tid = tenant_id if tenant_id is not None else user.get("tenant_id", cfg.default_tenant_id)
    payload: dict[str, Any] = {"sub": subject, "exp": expire, "role": role}
    if tid:
        payload["tenant_id"] = str(tid)
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
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
    if role not in VALID_ROLES:
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
    if not cfg.auth_enabled:
        return {
            "username": "anonymous",
            "auth": "disabled",
            "role": "admin",
            **({"tenant_id": cfg.default_tenant_id} if cfg.default_tenant_id else {}),
        }
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_roles(*roles: str) -> Callable[..., dict[str, Any]]:
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


def startup_security_checks() -> None:
    ensure_password_hash()
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


def reset_password_hash_cache() -> None:
    """Test helper: force re-hash after cfg mutation."""
    global _admin_password_hash, _user_password_hashes
    _admin_password_hash = None
    _user_password_hashes = {}
