"""CLI: ``python -m sample_data --out examples/data``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import write_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write deterministic ASPC demo CSVs for CLI / manual testing."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/data"),
        help="Output directory (default: examples/data)",
    )
    args = parser.parse_args(argv)
    paths = write_all(args.out)
    print(f"Wrote {len(paths)} CSVs to {args.out.resolve()}")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
