#!/usr/bin/env python3
"""Generate (or --check) the evidence ledger.

    uv run --no-sync scripts/10_evidence_ledger.py           # write both files
    uv run --no-sync scripts/10_evidence_ledger.py --check   # exit 1 if stale

Writes results/evidence_ledger.json and docs/evidence-ledger.md from the claim
specs in tfm3lab.claims, checking each source's sha256 against a run manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tfm3lab.claims import EXEMPTIONS, SPECS
from tfm3lab.ledger import run

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    args = ap.parse_args(argv)
    return run(ROOT, SPECS, EXEMPTIONS, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
