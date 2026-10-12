#!/usr/bin/env python3
"""Render (or --check) the public documents from the evidence ledger.

    uv run --no-sync scripts/11_render_docs.py           # write rendered documents
    uv run --no-sync scripts/11_render_docs.py --check   # exit 1 on any failure

--check first runs the scripts/10_evidence_ledger.py --check logic, then re-renders
every registered template (tfm3lab.publish.PUBLIC_DOCS) in memory, compares it
byte for byte with the committed output and rejects numeric tokens no placeholder
produced (allowlist: configs/public_numbers_allowlist.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tfm3lab.publish import run

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="re-render in memory and compare")
    args = ap.parse_args(argv)
    return run(ROOT, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
