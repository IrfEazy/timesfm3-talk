"""Publication gate: public documents are rendered from the evidence ledger.

Every public document is ``<template> -> <output>``; the registry
:data:`PUBLIC_DOCS` is the only place that lists them. Templates cite results with
``{{claim_id.field}}`` placeholders (:func:`tfm3lab.ledger.render`). ``--check``:

1. regenerates the evidence ledger in memory and compares it with the committed one;
2. re-renders every template from the committed ledger and compares it byte for byte
   with the committed output (stale outputs fail);
3. scans every rendered body for numeric tokens that no placeholder produced
   (:func:`tfm3lab.ledger.find_untraced_numbers`) with the occurrence-scoped allowlist
   in ``configs/public_numbers_allowlist.json``; unused allowlist entries fail.

Each output starts with an HTML comment naming its template. The comment is added
by the renderer after the scan, so the scan covers authored text only.

Slide documents (``front_matter=True``) start with a Marp YAML front-matter block. Its
CSS sizes and colours are not claims, so that one leading block (``---`` ... ``---`` at
the very start of the template) is blanked before the scan, and the generated comment
goes right after it (Marp needs the front matter first). Nothing else is exempt: slide
directives and every other comment in the body are scanned like prose, and a
placeholder inside the front matter is an error.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import ledger as L
from .claims import EXEMPTIONS, SPECS

ALLOWLIST_PATH = "configs/public_numbers_allowlist.json"
RENDERER = "scripts/11_render_docs.py"


@dataclass(frozen=True)
class PublicDoc:
    template: str
    output: str
    front_matter: bool = False  # Marp slide document: leading YAML block is exempt from the scan


# The single registry of public documents (later packages append README).
PUBLIC_DOCS: tuple[PublicDoc, ...] = (
    PublicDoc("docs/report.md.template", "docs/report.md"),
    PublicDoc("slides/talk.md.template", "slides/talk.md", front_matter=True),
    PublicDoc("docs/talk-outline.md.template", "docs/talk-outline.md"),
)


def header(doc: PublicDoc) -> str:
    return f"<!-- generated from {doc.template} by {RENDERER}; do not edit -->\n\n"


_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.DOTALL)


def split_front_matter(text: str) -> tuple[str, str]:
    """``(front matter block, rest)``; the block is empty when the text has none.

    Only a block that opens on the very first line is front matter.
    """
    m = _FRONT_MATTER_RE.match(text)
    return (text[: m.end()], text[m.end() :]) if m else ("", text)


def _blank(block: str) -> str:
    """Replace every character but newlines by a space (keeps line numbers)."""
    return re.sub(r"[^\n]", " ", block)


def load_allowlist(root: Path, path: str = ALLOWLIST_PATH) -> list[L.AllowedLiteral]:
    p = Path(root) / path
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise L.LedgerError(f"{path}: invalid JSON ({exc})") from exc
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise L.LedgerError(f"{path}: expected an object with an 'entries' list")
    out = []
    for e in entries:
        try:
            out.append(
                L.AllowedLiteral(
                    doc=e["doc"],
                    literal=e["literal"],
                    context=e["context"],
                    reason=e["reason"],
                    count=int(e["expected_count"]),
                )
            )
        except (KeyError, TypeError) as exc:
            raise L.LedgerError(f"{path}: bad entry {e!r} ({exc!r})") from exc
    return out


def load_committed_ledger(root: Path) -> dict:
    p = Path(root) / L.LEDGER_JSON
    if not p.is_file():
        raise L.LedgerError(f"{L.LEDGER_JSON} is missing; run scripts/10_evidence_ledger.py")
    return json.loads(p.read_text())


def render_doc(root: Path, doc: PublicDoc, ledger: Mapping) -> tuple[str, str]:
    """``(public text, marked body)``: the file content and the text to scan."""
    tpl = Path(root) / doc.template
    if not tpl.is_file():
        raise L.LedgerError(f"template {doc.template} not found")
    text = tpl.read_text()
    marked = L.render(text, ledger, mark=True)
    if not doc.front_matter:
        return header(doc) + L.strip_marks(marked), marked
    fm, rest = split_front_matter(marked)
    if not fm:
        raise L.LedgerError(f"{doc.template}: expected a leading front-matter block")
    if L.strip_marks(fm) != fm:
        raise L.LedgerError(f"{doc.template}: placeholders are not allowed in the front matter")
    return fm + header(doc) + L.strip_marks(rest), _blank(fm) + rest


def scan_doc(
    marked: str, doc: PublicDoc, allowlist: Iterable[L.AllowedLiteral]
) -> list[L.UntracedNumber]:
    return L.find_untraced_numbers(marked, allowlist, doc=doc.output)


def write_docs(root: Path, docs: Iterable[PublicDoc] = PUBLIC_DOCS, out: Callable = print) -> int:
    root = Path(root)
    try:
        ledger = load_committed_ledger(root)
        for doc in docs:
            text, _ = render_doc(root, doc, ledger)
            (root / doc.output).write_text(text)
            out(f"wrote {doc.output}")
    except L.LedgerError as exc:
        out(f"render error: {exc}")
        return 1
    return 0


def check_docs(
    root: Path,
    docs: Iterable[PublicDoc] = PUBLIC_DOCS,
    out: Callable = print,
    *,
    specs=SPECS,
    exemptions: Mapping[str, str] = EXEMPTIONS,
) -> int:
    root = Path(root)
    rc = L.run(root, specs, exemptions, check=True, out=out)
    if rc != 0:
        return rc
    bad = False
    try:
        ledger = load_committed_ledger(root)
        allowlist = load_allowlist(root)
        docs = list(docs)
        for doc in docs:
            text, marked = render_doc(root, doc, ledger)
            out_path = root / doc.output
            old = out_path.read_text() if out_path.is_file() else ""
            if old != text:
                bad = True
                out(f"STALE {doc.output}:\n{L._diff_summary(doc.output, old, text)}")
            for u in scan_doc(marked, doc, allowlist):
                bad = True
                out(f"UNTRACED NUMBER in {doc.template} line {u.line}: {u.literal!r} ({u.context})")
        known = {d.output for d in docs}
        for e in allowlist:
            if e.doc not in known:
                bad = True
                out(f"allowlist entry for unregistered document {e.doc!r}: {e.literal!r}")
    except L.LedgerError as exc:
        out(f"publish error: {exc}")
        return 1
    if bad:
        out("public documents failed the publication gate; run scripts/11_render_docs.py")
        return 1
    out("public documents are fresh and every number is traced")
    return 0


def run(root: Path, *, check: bool = False, out: Callable = print) -> int:
    return check_docs(root, out=out) if check else write_docs(root, out=out)
