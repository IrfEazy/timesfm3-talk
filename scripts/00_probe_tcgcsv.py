#!/usr/bin/env python3
"""Probe: confirm TCGCSV's schema still matches what tfm3lab/data/mtg.py
assumes, before running a real ingest.

This project's Experiment A depends on TCGCSV's undocumented archive shape
(see tfm3lab/data/mtg.py's module docstring for what was verified and
when). The service can change without notice — this script re-checks it
in under 30 seconds so a schema drift is caught here, not halfway through
a multi-hour historical backfill.

Usage: uv run scripts/00_probe_tcgcsv.py
Exit code 0: all checks passed. Exit code 1: something drifted — see the
printed message for what to update, and fetch_mtgjson_fallback() as the
interim path.
"""

from __future__ import annotations

import datetime as dt
import sys

import requests

from tfm3lab.data.mtg import (
    DEFAULT_CARDS,
    MAGIC_CATEGORY_ID,
    TCGCSV_ARCHIVE_START,
    probe_archive_available,
    resolve_card_specs,
)


def main() -> int:
    session = requests.Session()
    ok = True

    print(f"1. Resolving {len(DEFAULT_CARDS)} DEFAULT_CARDS against the live catalog...")
    try:
        resolved = resolve_card_specs(DEFAULT_CARDS, category_id=MAGIC_CATEGORY_ID, session=session)
        print(resolved.to_string(index=False))
    except Exception as e:
        ok = False
        print(f"   FAILED: {e}")

    print(f"\n2. Checking the earliest known archive date ({TCGCSV_ARCHIVE_START}) is reachable...")
    if probe_archive_available(TCGCSV_ARCHIVE_START, session=session):
        print("   OK.")
    else:
        ok = False
        print("   FAILED: earliest archive not reachable at the expected date.")

    day_before = TCGCSV_ARCHIVE_START - dt.timedelta(days=1)
    print(f"3. Checking the day BEFORE that ({day_before}) is correctly absent...")
    if not probe_archive_available(day_before, session=session):
        print("   OK — confirms TCGCSV_ARCHIVE_START, not just that archives exist somewhere.")
    else:
        print(
            "   NOTE: an archive exists earlier than TCGCSV_ARCHIVE_START — "
            "more history is available for free, update the constant in tfm3lab/data/mtg.py."
        )

    print()
    if ok:
        print("All checks passed — tfm3lab/data/mtg.py's assumptions still hold.")
        return 0
    print(
        "SCHEMA DRIFT DETECTED. Do not run 01_fetch_data.py's MTG ingest until "
        "tfm3lab/data/mtg.py is updated to match. Interim fallback: "
        "tfm3lab.data.mtg.fetch_mtgjson_fallback() still works against MTGJSON's "
        "90-day AllPrices window (less statistical power — log that loss if used)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
