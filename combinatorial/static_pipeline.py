"""Persist generated static fixtures for the matrix."""
from __future__ import annotations

from pathlib import Path

from combinatorial.batch_runner import write_static_fixture
from combinatorial.matrix import CaseSpec, build_matrix, load_config


def generate_static(cases: list[CaseSpec] | None = None, *, output_dir: str | Path | None = None) -> list[Path]:
    cfg = load_config()
    out = Path(output_dir or cfg["output_dir"]) / "static"
    if cases is None:
        cases = build_matrix(
            mode=cfg.get("mode", "sparse"),
            seed=int(cfg.get("seed", 42)),
            max_sparse_cases=int(cfg.get("max_sparse_cases", 80)),
            adversarial=bool(cfg.get("adversarial", True)),
        )
    return [write_static_fixture(c, out, seed=i) for i, c in enumerate(cases)]
