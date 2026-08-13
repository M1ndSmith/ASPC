#!/usr/bin/env python3
"""Optional ops digest — print multi-stream summary (stdout / JSON).

Wire to cron or SMTP later; for now this is the in-app companion CLI.

Usage:
  ASPC_DEV_INSECURE=1 python scripts/ops_digest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.factory import get_repository  # noqa: E402
from adapters.protocols import StreamingOpsRepository  # noqa: E402
from apps.config import get_config  # noqa: E402


def main() -> int:
    cfg = get_config()
    repo = get_repository(cfg)
    streams = []
    if isinstance(repo, StreamingOpsRepository):
        streams = repo.list_streams(active_only=False)
    runs = repo.list_runs(limit=20) if hasattr(repo, "list_runs") else []
    payload = {
        "streams_total": len(streams),
        "streams_active": sum(1 for s in streams if s.get("active")),
        "recent_runs": len(runs),
        "streams": [
            {
                "stream_key": s.get("stream_key"),
                "active": s.get("active"),
                "limits_version": s.get("limits_version"),
            }
            for s in streams
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
