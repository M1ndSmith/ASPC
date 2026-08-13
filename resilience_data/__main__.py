"""CLI: ``python -m resilience_data [--force]`` — write committed CSVs under cases/."""
from __future__ import annotations

import argparse
from pathlib import Path

from resilience_data import CASES_DIR, ROOT, write_csv
from resilience_data.generators import CASE_BUILDERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic resilience_data CSVs for spc_core judgment."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CASES_DIR,
        help="Output cases directory (default: resilience_data/cases)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSVs",
    )
    args = parser.parse_args(argv)

    written: list[Path] = []
    skipped = 0
    for _case_id, (rel, builder) in sorted(CASE_BUILDERS.items()):
        path = args.out / rel
        if path.exists() and not args.force:
            skipped += 1
            continue
        cols = builder()
        written.append(write_csv(cols, path))

    print(f"Wrote {len(written)} CSVs under {args.out} ({skipped} skipped; use --force)")
    for p in written:
        try:
            print(f"  {p.relative_to(ROOT)}")
        except ValueError:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
