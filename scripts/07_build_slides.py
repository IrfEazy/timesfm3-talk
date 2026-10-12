#!/usr/bin/env python3
"""Render slides/talk.md from slides/talk.md.template through the publication gate.

Every number on the slides comes from the evidence ledger via ``{{claim.field}}``
placeholders (``tfm3lab.publish``); this wrapper only renders the slide document
(same output path as before) and prints the Marp command. ``slides/talk.html`` is
not produced here: Marp needs Node and is not assumed to be available.

    npx @marp-team/marp-cli slides/talk.md -o slides/talk.html
    npx @marp-team/marp-cli slides/talk.md -o slides/talk.pdf

Usage: uv run --no-sync scripts/07_build_slides.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from tfm3lab.publish import PUBLIC_DOCS, write_docs

ROOT = Path(__file__).resolve().parent.parent
SLIDES = tuple(d for d in PUBLIC_DOCS if d.output == "slides/talk.md")


def main() -> int:
    rc = write_docs(ROOT, SLIDES)
    if rc == 0:
        print("Render it with: npx @marp-team/marp-cli slides/talk.md -o slides/talk.html")
    return rc


if __name__ == "__main__":
    sys.exit(main())
