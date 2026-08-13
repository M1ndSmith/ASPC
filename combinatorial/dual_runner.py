"""Dual runner: batch + stream matrix execution."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from combinatorial.batch_runner import run_batch_case, write_static_fixture
from combinatorial.matrix import CaseSpec, build_matrix, load_config
from combinatorial.stream_runner import run_stream_case


def run_case(case: CaseSpec, *, seed: int = 0) -> dict[str, Any]:
    if case.modality == "batch":
        return run_batch_case(case, seed=seed)
    if case.modality == "stream":
        return run_stream_case(case, seed=seed)
    # both: run stream path (includes establish + parity); also attach batch establish summary
    stream = run_stream_case(case, seed=seed)
    if case.entry == "phase2_parity":
        return stream
    batch = run_batch_case(case, seed=seed)
    return {
        "id": case.id,
        "status": "PASS"
        if batch["status"] == "PASS" and stream["status"] == "PASS"
        else "FAIL"
        if "FAIL" in (batch["status"], stream["status"])
        else "ERROR",
        "expect_class": case.expect_class,
        "mismatches": list(batch.get("mismatches") or []) + list(stream.get("mismatches") or []),
        "observed": {"batch": batch.get("observed"), "stream": stream.get("observed")},
        "adversarial": case.adversarial,
        "modality": case.modality,
    }


def run_matrix(
    *,
    mode: str | None = None,
    seed: int | None = None,
    max_sparse_cases: int | None = None,
    adversarial: bool | None = None,
    output_dir: str | Path | None = None,
    write_fixtures: bool = True,
) -> dict[str, Any]:
    cfg = load_config()
    mode = mode or cfg.get("mode", "sparse")
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    max_sparse = int(max_sparse_cases if max_sparse_cases is not None else cfg.get("max_sparse_cases", 80))
    adv = bool(cfg.get("adversarial", True) if adversarial is None else adversarial)
    out = Path(output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cases = build_matrix(mode=mode, seed=seed, max_sparse_cases=max_sparse, adversarial=adv)
    if write_fixtures:
        static_dir = out / "static"
        for i, c in enumerate(cases):
            write_static_fixture(c, static_dir, seed=i)

    results = [run_case(c, seed=i) for i, c in enumerate(cases)]
    counts = Counter(r["status"] for r in results)
    summary = {
        "mode": mode,
        "seed": seed,
        "n_cases": len(results),
        "counts": dict(counts),
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
