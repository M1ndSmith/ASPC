"""CLI: python -m combinatorial generate|run|report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="combinatorial", description="spc_core combinatorial matrix")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("generate", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["sparse", "exhaustive"], default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--max-sparse", type=int, default=None)

    args = parser.parse_args(argv)

    from combinatorial.dual_runner import run_matrix
    from combinatorial.matrix import build_matrix, load_config
    from combinatorial.report import full_report
    from combinatorial.static_pipeline import generate_static

    cfg = load_config()
    mode = args.mode or cfg.get("mode", "sparse")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    max_sparse = args.max_sparse if args.max_sparse is not None else int(cfg.get("max_sparse_cases", 80))

    if args.cmd == "generate":
        cases = build_matrix(mode=mode, seed=seed, max_sparse_cases=max_sparse)
        paths = generate_static(cases)
        print(f"Wrote {len(paths)} fixtures under combinatorial/out/static")
        return 0

    if args.cmd == "run":
        summary = run_matrix(mode=mode, seed=seed, max_sparse_cases=max_sparse)
        print(json.dumps({"counts": summary["counts"], "n_cases": summary["n_cases"]}, indent=2))
        fails = [r for r in summary["results"] if r["status"] != "PASS"]
        return 1 if fails else 0

    if args.cmd == "report":
        out = full_report(mode=mode)
        print(f"Wrote reports under {out['out']}")
        print(json.dumps(out["coverage"], indent=2)[:2000])
        fails = [r for r in out["summary"]["results"] if r["status"] != "PASS"]
        return 1 if fails else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
