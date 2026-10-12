"""Evidence ledger: every public number traces to a committed results file.

Pipeline: claim specs (hand-written, :mod:`tfm3lab.claims`) + committed parquet
tables + run manifests -> generated ``results/evidence_ledger.json`` and
``docs/evidence-ledger.md``. Prose cites claims with ``{{claim_id.field}}``
placeholders; :func:`render` substitutes formatted values and
:func:`find_untraced_numbers` rejects any other numeric token.

Invariants enforced here (all errors are :class:`LedgerError` or ``ValueError``):

* ids are unique and match ``[a-z0-9_.-]+``;
* every selector matches exactly one row (never "the first of several");
* missing columns and non-finite numbers are errors, unless the field declares a
  row-level ``nan_if`` condition (:func:`when`) that the row satisfies (rendered
  ``n/a``, raw ``null``); NaN in any other row, and ``inf`` always, is an error;
* every expanding spec declares ``expected_claims`` (exact count);
* one shared formatter (``Decimal(str(x))``, ``ROUND_HALF_EVEN``, negative zero
  normalised to ``0``);
* the sha256 of each source file (from bytes) must equal the one recorded for the
  same basename in the NEWEST manifest listing that basename (an older matching
  manifest does not suffice); a source with no manifest listing is an error unless
  listed in the explicit exemption table. "Newest" is by the manifest's recorded
  ``_meta.written_at_utc`` when every listing manifest has one, else by filename
  order (ties on timestamp also fall back to filename order).

Manifest layouts understood: ``outputs`` as a list of ``{path, sha256}``;
``outputs`` as a dict ``name -> {path, sha256}``; and a single top-level
``output`` + ``output_sha256`` pair (inference self-check). Files are matched by
basename.

Literal guard policy: the scan looks at the *rendered* text, so numbers inside
Markdown link URLs, image paths and fenced code blocks are NOT exempt; they must
be produced by a placeholder or be covered by an occurrence-scoped allowlist
entry (doc path + literal + exact context + reason + expected count). Provenance
is per occurrence (sentinel marks around substituted text): a typed number equal
to a substituted value elsewhere is still flagged. Digits inside identifiers
(``h28``, ``v2``) are numeric tokens too. Single tokens: optional sign (ASCII
hyphen or U+2212), thousands grouped with comma / space / thin space / NBSP,
decimals, leading-dot decimals, scientific notation, optional ``%``. An en dash
is not part of a token, so ``10–20`` yields two tokens. An allowlist entry with
zero occurrences (unused) is an error.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = 1
LEDGER_JSON = "results/evidence_ledger.json"
LEDGER_MD = "docs/evidence-ledger.md"
MANIFEST_DIR = "results/manifests"
NA_TEXT = "n/a"

_ID_RE = re.compile(r"^[a-z0-9_.-]+$")
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class LedgerError(ValueError):
    """Any violation of a ledger invariant."""


# --------------------------------------------------------------------------- formatting


def format_decimal(x: float | int, decimals: int) -> str:
    """Shared float formatter: ``Decimal(str(x))`` quantised half-even, no ``-0``."""
    if isinstance(x, bool):
        raise LedgerError(f"cannot format bool {x!r} as a number")
    xf = float(x)
    if not math.isfinite(xf):
        raise LedgerError(f"non-finite number {x!r}")
    if decimals < 0:
        raise LedgerError("decimals must be >= 0")
    q = Decimal(str(xf)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
    if q == 0:
        q = q.copy_abs()
    return format(q, "f")


def format_integer(x: float | int) -> str:
    if isinstance(x, bool):
        raise LedgerError(f"cannot format bool {x!r} as an integer")
    xf = float(x)
    if not math.isfinite(xf) or xf != int(xf):
        raise LedgerError(f"{x!r} is not an integer")
    return str(int(xf))


Clause = tuple[tuple[str, tuple[Any, ...]], ...]


def when(**cols: Any) -> Clause:
    """Row condition: every named column's value is in the given value(s)."""
    if not cols:
        raise LedgerError("when(): at least one column required")
    return tuple(
        sorted(
            (c, tuple(v) if isinstance(v, (list, tuple, set, frozenset)) else (v,))
            for c, v in cols.items()
        )
    )


def _norm_nan_if(x: Any) -> tuple[Clause, ...]:
    """None -> (); one clause (from :func:`when` or a dict) or a tuple of clauses."""
    if not x:
        return ()
    if isinstance(x, Mapping):
        return (when(**x),)
    if isinstance(x[0], tuple) and x[0] and isinstance(x[0][0], str):
        return (tuple(x),)  # a single clause
    return tuple(c if isinstance(c, tuple) else when(**c) for c in x)


def _clause_holds(clause: Clause, row: Mapping[str, Any], who: str) -> bool:
    for col, values in clause:
        if col not in row:
            raise LedgerError(f"{who}: nan_if column {col!r} missing from the table")
        v = _py(row[col])
        if _is_missing(v) or v not in values:
            return False
    return True


@dataclass(frozen=True)
class FieldSpec:
    """One declared field: source column plus a format policy.

    ``nan_if`` is a row-level condition (OR of :func:`when` clauses, each an AND of
    ``column in values``) under which a missing value renders ``n/a``; a missing
    value in any row not meeting it is an error.
    """

    column: str
    kind: str = "float"  # float | int | text | bool
    decimals: int = 3
    nan_if: tuple[Clause, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("float", "int", "text", "bool"):
            raise LedgerError(f"unknown field kind {self.kind!r}")
        object.__setattr__(self, "nan_if", _norm_nan_if(self.nan_if))

    def format(self, cell: Any, row: Mapping[str, Any] | None = None) -> tuple[Any, str]:
        """Return ``(raw, text)`` for one cell, applying the conditional NaN policy."""
        if _is_missing(cell):
            if row is not None and any(_clause_holds(c, row, self.column) for c in self.nan_if):
                return None, NA_TEXT
            raise LedgerError(
                f"column {self.column!r}: missing value (row does not meet the nan_if condition)"
            )
        if self.kind == "float":
            raw = float(cell)
            return raw, format_decimal(raw, self.decimals)
        if self.kind == "int":
            text = format_integer(cell)
            return int(text), text
        if self.kind == "bool":
            if not isinstance(cell, bool):
                raise LedgerError(f"column {self.column!r}: {cell!r} is not a bool")
            return bool(cell), "true" if bool(cell) else "false"
        if not isinstance(cell, str):
            raise LedgerError(f"column {self.column!r}: {cell!r} is not text")
        return cell, cell


def F(column: str, decimals: int = 3, nan_if: Any = None) -> FieldSpec:
    return FieldSpec(column, "float", decimals, nan_if)


def I(column: str, nan_if: Any = None) -> FieldSpec:  # noqa: E743
    return FieldSpec(column, "int", 0, nan_if)


def T(column: str, nan_if: Any = None) -> FieldSpec:
    return FieldSpec(column, "text", 0, nan_if)


def B(column: str, nan_if: Any = None) -> FieldSpec:
    return FieldSpec(column, "bool", 0, nan_if)


def _is_missing(cell: Any) -> bool:
    if cell is None or cell is pd.NA or cell is pd.NaT:
        return True
    return isinstance(cell, float) and math.isnan(cell)


# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class ClaimSpec:
    """Declarative claim(s) over one results table.

    ``id`` is a ``str.format`` pattern over the ``expand`` key columns (values are
    lower-cased and sanitised to ``[a-z0-9_.-]``). With ``expand`` one claim is
    created per distinct key combination; each selector (``select`` plus the key
    equalities) must match exactly one row. Without ``counts`` the claim is a
    *row claim* built from ``fields``. With ``counts`` it is a *count claim*: the
    selector may match many rows, each ``counts`` entry is the number of selected
    rows satisfying its ``where`` equalities (empty ``where`` = all selected rows)
    and ``unique`` columns must hold one single value across the selected rows.
    ``expected_claims`` is the exact number of claims an expanding spec must produce.
    """

    id: str
    source: str
    group: str
    select: Mapping[str, Any] = field(default_factory=dict)
    expand: tuple[str, ...] = ()
    fields: Mapping[str, FieldSpec] = field(default_factory=dict)
    counts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    unique: Mapping[str, FieldSpec] = field(default_factory=dict)
    expected_claims: int | None = None

    def __post_init__(self) -> None:
        if self.expand and self.expected_claims is None:
            raise LedgerError(f"{self.id}: expanding spec must declare expected_claims")
        if self.counts and self.fields:
            raise LedgerError(f"{self.id}: counts and fields are mutually exclusive")
        names = list(self.fields) + list(self.counts) + list(self.unique)
        if len(set(names)) != len(names):
            raise LedgerError(f"{self.id}: duplicate field names")
        for n in names:
            if not _FIELD_RE.match(n):
                raise LedgerError(f"{self.id}: bad field name {n!r}")


def _py(v: Any) -> Any:
    """numpy scalar -> python scalar (JSON-safe, comparable)."""
    return v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v


def _sanitize(v: Any) -> str:
    return re.sub(r"[^a-z0-9_.-]", "_", str(_py(v)).lower())


def _mask(df: pd.DataFrame, where: Mapping[str, Any], who: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in where.items():
        if col not in df.columns:
            raise LedgerError(f"{who}: missing column {col!r}")
        mask &= (df[col] == val).fillna(False).astype(bool)
    return mask


# --------------------------------------------------------------------------- integrity


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_time(data: Mapping[str, Any], name: str) -> datetime | None:
    meta = data.get("_meta")
    ts = meta.get("written_at_utc") if isinstance(meta, dict) else None
    if ts is None:
        return None
    try:
        t = datetime.fromisoformat(str(ts))
    except ValueError as exc:
        raise LedgerError(f"manifest {name}: bad written_at_utc {ts!r}") from exc
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def load_manifest_index(root: Path) -> dict[str, list[tuple[str, str, datetime | None]]]:
    """basename -> filename-sorted [(manifest filename, sha256, recorded time or None)]."""
    index: dict[str, list[tuple[str, str, datetime | None]]] = {}
    mdir = Path(root) / MANIFEST_DIR
    for mp in sorted(mdir.glob("*.json")) if mdir.is_dir() else []:
        try:
            data = json.loads(mp.read_text())
        except json.JSONDecodeError as exc:
            raise LedgerError(f"manifest {mp.name}: invalid JSON ({exc})") from exc
        if not isinstance(data, dict):
            continue
        when_ = _manifest_time(data, mp.name)
        entries: list[tuple[str, str]] = []
        outputs = data.get("outputs")
        if isinstance(outputs, list):
            items: Iterable[tuple[str | None, Any]] = [(None, o) for o in outputs]
        elif isinstance(outputs, dict):
            items = list(outputs.items())
        else:
            items = []
        for name, item in items:
            if isinstance(item, dict) and isinstance(item.get("sha256"), str):
                base = Path(str(item.get("path") or name or "")).name
                if base:
                    entries.append((base, item["sha256"]))
        if isinstance(data.get("output"), str) and isinstance(data.get("output_sha256"), str):
            entries.append((Path(data["output"]).name, data["output_sha256"]))
        for base, sha in entries:
            index.setdefault(base, []).append((mp.name, sha, when_))
    for v in index.values():
        v.sort(key=lambda e: e[0])
    return index


def _newest(recorded: list[tuple[str, str, datetime | None]]) -> tuple[str, str]:
    """(filename, sha256) of the newest listing: by recorded time if all have one."""
    if all(t is not None for _, _, t in recorded):
        best = max(recorded, key=lambda e: (e[2], e[0]))
    else:
        best = max(recorded, key=lambda e: e[0])
    return best[0], best[1]


def resolve_manifest(
    source: str,
    sha: str,
    index: Mapping[str, list[tuple[str, str, datetime | None]]],
    exemptions: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Return ``(manifest filename, exemption reason)``; error if neither applies.

    The NEWEST manifest listing the basename must record ``sha``.
    """
    base = Path(source).name
    recorded = index.get(base, [])
    newest = _newest(recorded) if recorded else None
    if newest and newest[1] == sha:
        return newest[0], None
    if source in exemptions:
        reason = exemptions[source].strip()
        if not reason:
            raise LedgerError(f"{source}: exemption needs a reason")
        return None, reason
    if newest:
        raise LedgerError(
            f"{source}: sha256 {sha[:12]} does not match the newest manifest {newest[0]} "
            f"({newest[1][:12]}); results changed without a manifest "
            f"(listed in: {', '.join(m for m, _, _ in recorded)})"
        )
    raise LedgerError(f"{source}: no manifest records this file and it is not exempt")


# --------------------------------------------------------------------------- build


def _combined_pairs(names: Iterable[str]) -> list[tuple[str, str, str]]:
    names = list(names)
    out = []
    for n in names:
        if n.endswith("_low") and n[:-4] + "_high" in names:
            out.append((n[:-4], n, n[:-4] + "_high"))
    return out


def _expand_claims(spec: ClaimSpec, df: pd.DataFrame) -> list[tuple[str, dict[str, Any]]]:
    """(claim id, full selector) for every claim a spec produces."""
    select = {k: _py(v) for k, v in spec.select.items()}
    if not spec.expand:
        if "{" in spec.id:
            raise LedgerError(f"{spec.id}: id pattern needs expand keys")
        return [(spec.id, select)]
    for k in spec.expand:
        if k not in df.columns:
            raise LedgerError(f"{spec.id}: missing expand column {k!r}")
    sub = df[_mask(df, select, spec.id)][list(spec.expand)]
    if sub.isna().any().any():
        raise LedgerError(f"{spec.id}: missing values in expand key columns")
    combos = sorted({tuple(_py(v) for v in row) for row in sub.itertuples(index=False)}, key=str)
    if not combos:
        raise LedgerError(f"{spec.id}: selector matches no rows")
    out = []
    for combo in combos:
        keys = dict(zip(spec.expand, combo, strict=True))
        try:
            cid = spec.id.format(**{k: _sanitize(v) for k, v in keys.items()})
        except KeyError as exc:
            raise LedgerError(f"{spec.id}: id pattern uses unknown key {exc}") from exc
        out.append((cid, {**select, **keys}))
    return out


def _build_claim(
    spec: ClaimSpec, cid: str, selector: dict[str, Any], df: pd.DataFrame
) -> dict[str, Any]:
    if not _ID_RE.match(cid):
        raise LedgerError(f"{cid}: id must match [a-z0-9_.-]+")
    rows = df[_mask(df, selector, cid)]
    fields: dict[str, dict[str, Any]] = {}

    def col(name: str) -> str:
        if name not in df.columns:
            raise LedgerError(f"{cid}: missing column {name!r} in {spec.source}")
        return name

    def put(name: str, fs: FieldSpec, cell: Any, row: Mapping[str, Any] | None = None) -> None:
        try:
            raw, text = fs.format(cell, row)
        except LedgerError as exc:
            raise LedgerError(f"{cid}.{name}: {exc}") from exc
        fields[name] = {"raw": raw, "text": text}

    if spec.counts:
        if rows.empty:
            raise LedgerError(f"{cid}: selector matches no rows")
        for name, where in spec.counts.items():
            n = int(_mask(rows, where, f"{cid}.{name}").sum())
            fields[name] = {"raw": n, "text": str(n)}
        for name, fs in spec.unique.items():
            vals = rows[col(fs.column)]
            distinct = vals.drop_duplicates()
            if len(distinct) != 1:
                raise LedgerError(
                    f"{cid}.{name}: column {fs.column!r} has {len(distinct)} distinct values"
                )
            put(name, fs, _py(distinct.iloc[0]))
    else:
        if len(rows) != 1:
            raise LedgerError(f"{cid}: selector {selector} matches {len(rows)} rows, expected 1")
        row = rows.iloc[0]
        for name, fs in spec.fields.items():
            put(name, fs, _py(row[col(fs.column)]), row)
        for base, lo, hi in _combined_pairs(spec.fields):
            if base in fields:
                raise LedgerError(f"{cid}: derived field {base!r} collides with a declared one")
            a, b = spec.fields[lo], spec.fields[hi]
            if (a.kind, a.decimals) != (b.kind, b.decimals):
                raise LedgerError(f"{cid}: {lo}/{hi} use different format policies")
            raws = [fields[lo]["raw"], fields[hi]["raw"]]
            fields[base] = {
                "raw": raws,
                "text": f"[{fields[lo]['text']}, {fields[hi]['text']}]",
            }
    return {
        "id": cid,
        "group": spec.group,
        "source": spec.source,
        "selector": dict(sorted(selector.items())),
        "kind": "count" if spec.counts else "row",
        "fields": dict(sorted(fields.items())),
    }


def build_ledger(
    specs: Iterable[ClaimSpec],
    root: Path,
    exemptions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate every spec against the files under ``root``."""
    root = Path(root)
    exemptions = exemptions or {}
    index = load_manifest_index(root)
    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, Any]] = {}
    claims: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.source not in sources:
            path = root / spec.source
            if not path.is_file():
                raise LedgerError(f"{spec.id}: source {spec.source} not found")
            sha = file_sha256(path)
            manifest, exemption = resolve_manifest(spec.source, sha, index, exemptions)
            sources[spec.source] = {
                "sha256": sha,
                "manifest": manifest,
                "exemption": exemption,
            }
            tables[spec.source] = pd.read_parquet(path)
        expanded = _expand_claims(spec, tables[spec.source])
        if spec.expected_claims is not None and len(expanded) != spec.expected_claims:
            raise LedgerError(
                f"{spec.id}: expected {spec.expected_claims} claims, expanded to {len(expanded)}"
            )
        for cid, selector in expanded:
            if cid in claims:
                raise LedgerError(f"duplicate claim id {cid!r}")
            claim = _build_claim(spec, cid, selector, tables[spec.source])
            claim.update(
                source_sha256=sources[spec.source]["sha256"],
                manifest=sources[spec.source]["manifest"],
            )
            if sources[spec.source]["exemption"]:
                claim["exemption"] = sources[spec.source]["exemption"]
            claims[cid] = claim
    return {
        "schema": SCHEMA_VERSION,
        "sources": dict(sorted(sources.items())),
        "claims": [claims[k] for k in sorted(claims)],
    }


def dump_json(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- markdown


def _cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(ledger: Mapping[str, Any]) -> str:
    """Render the committed ledger JSON (grouped by experiment)."""
    lines = [
        "# Evidence ledger",
        "",
        "Generated by `scripts/10_evidence_ledger.py` from `results/evidence_ledger.json`; "
        "do not edit by hand. Every value is read from the cited committed results file "
        "(sha256 checked against a run manifest) and formatted by one shared function.",
        "",
    ]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for c in ledger["claims"]:
        groups.setdefault(c["group"], []).append(c)
    for group in sorted(groups):
        claims = groups[group]
        lines += [f"## {group}", ""]
        srcs = sorted({c["source"] for c in claims})
        for s in srcs:
            info = ledger["sources"][s]
            man = info["manifest"] or f"exempt: {info['exemption']}"
            lines.append(f"- `{s}` sha256 `{info['sha256'][:16]}` ({_cell(man)})")
        lines += ["", "| id | value | CI | n | source | manifest |", "|---|---|---|---|---|---|"]
        for c in claims:
            f = c["fields"]
            if "value" in f:
                value = f["value"]["text"]
            else:
                value = "; ".join(
                    f"{k}={v['text']}" for k, v in f.items() if k not in ("n", "verdict")
                )
            pairs = [k for k, v in f.items() if isinstance(v["raw"], list)]
            ci_name = "ci" if "ci" in pairs else (pairs[0] if pairs else None)
            ci = f[ci_name]["text"] if ci_name else ""
            n = f["n"]["text"] if "n" in f else ""
            man = c["manifest"] or "exempt"
            lines.append(
                f"| `{c['id']}` | {_cell(value)} | {_cell(ci)} | {_cell(n)} "
                f"| `{Path(c['source']).name}` | `{man}` |"
            )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- generate / check


def generate(
    root: Path, specs: Iterable[ClaimSpec], exemptions: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Regenerate ``(ledger_json_text, markdown_text)`` in memory."""
    ledger = build_ledger(specs, root, exemptions)
    return dump_json(ledger), render_markdown(ledger)


def _diff_summary(name: str, old: str, new: str, limit: int = 20) -> str:
    diff = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"{name} (committed)", f"{name} (regenerated)", n=0
        )
    )
    shown = diff[:limit]
    more = f"\n... {len(diff) - limit} more diff lines" if len(diff) > limit else ""
    return "\n".join(shown) + more


def run(
    root: Path,
    specs: Iterable[ClaimSpec],
    exemptions: Mapping[str, str] | None = None,
    *,
    check: bool = False,
    out=print,
) -> int:
    """Write (default) or check the two committed ledger files. Returns exit code."""
    root = Path(root)
    try:
        json_text, md_text = generate(root, specs, exemptions)
    except LedgerError as exc:
        out(f"ledger error: {exc}")
        return 1
    targets = {LEDGER_JSON: json_text, LEDGER_MD: md_text}
    if check:
        bad = False
        for rel, new in targets.items():
            p = root / rel
            old = p.read_text() if p.is_file() else ""
            if old != new:
                bad = True
                out(f"STALE {rel}:\n{_diff_summary(rel, old, new)}")
        if bad:
            out("evidence ledger is stale; run scripts/10_evidence_ledger.py")
            return 1
        out("evidence ledger is fresh")
        return 0
    for rel, new in targets.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new)
        out(f"wrote {rel}")
    return 0


# --------------------------------------------------------------------------- renderer

_MARK_OPEN = "\ue000"
_MARK_CLOSE = "\ue001"
_FILL = "\ue002"
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


def _index(ledger: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {c["id"]: c for c in ledger["claims"]}


def render(template_text: str, ledger: Mapping[str, Any], *, mark: bool = False) -> str:
    """Substitute ``{{claim_id.field}}``; unknown id/field or leftover ``{{`` is an error.

    With ``mark=True`` substituted values are wrapped in private-use sentinels so
    :func:`find_untraced_numbers` can tell placeholder output from authored text;
    use :func:`strip_marks` to get the public text.
    """
    if any(ch in template_text for ch in (_MARK_OPEN, _MARK_CLOSE, _FILL)):
        raise LedgerError("template contains reserved private-use characters")
    claims = _index(ledger)

    def sub(m: re.Match[str]) -> str:
        ref = m.group(1)
        cid, dot, fname = ref.rpartition(".")
        if not dot or cid not in claims:
            raise LedgerError(f"unknown claim id in placeholder {{{{{ref}}}}}")
        fields = claims[cid]["fields"]
        if fname not in fields:
            raise LedgerError(f"unknown field {fname!r} for claim {cid!r}")
        text = fields[fname]["text"]
        return f"{_MARK_OPEN}{text}{_MARK_CLOSE}" if mark else text

    out = _PLACEHOLDER_RE.sub(sub, template_text)
    if "{{" in _blank_marked(out):
        raise LedgerError("leftover '{{' after rendering (malformed placeholder)")
    return out


def strip_marks(text: str) -> str:
    return text.replace(_MARK_OPEN, "").replace(_MARK_CLOSE, "")


# --------------------------------------------------------------------------- literal guard

_GROUP_SEP = "[, \u2009\u00a0\u202f]"
_EXP = "[eE][+\\-\u2212]?\\d+"
_NUMBER_RE = re.compile(
    "(?:(?<![\\w.])[-\u2212])?(?:"
    + rf"(?:\d+(?:\.\d+)?|(?<![\w.])\.\d+){_EXP}"  # scientific
    + rf"|(?<!\d)\d{{1,3}}(?:{_GROUP_SEP}\d{{3}}(?!\d))+(?:\.\d+)?"  # grouped thousands
    + r"|\d+(?:\.\d+)?"  # plain / decimal
    + r"|(?<![\w.])\.\d+"  # leading-dot decimal
    + ")%?"
)


@dataclass(frozen=True)
class AllowedLiteral:
    """Occurrence-scoped exemption for one authored numeric literal."""

    doc: str
    literal: str
    context: str
    reason: str
    count: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise LedgerError(f"allowlist {self.literal!r}: reason required")
        if self.context.count(self.literal) != 1:
            raise LedgerError(
                f"allowlist {self.literal!r}: context must contain the literal exactly once"
            )
        if self.count < 1:
            raise LedgerError(f"allowlist {self.literal!r}: count must be >= 1")


@dataclass(frozen=True)
class UntracedNumber:
    literal: str
    start: int
    line: int
    context: str


def _blank_marked(text: str) -> str:
    return re.sub(
        f"{_MARK_OPEN}[^{_MARK_CLOSE}]*{_MARK_CLOSE}", lambda m: _FILL * len(m.group(0)), text
    )


def find_untraced_numbers(
    rendered_text: str, allowlist: Iterable[AllowedLiteral] = (), *, doc: str = ""
) -> list[UntracedNumber]:
    """Numeric tokens not produced by a placeholder and not allowlisted.

    ``rendered_text`` must come from ``render(..., mark=True)``. Each allowlist
    entry for ``doc`` must cover exactly ``entry.count`` token occurrences, else
    :class:`LedgerError`. Links and fenced code are scanned like any other text.
    """
    blanked = _blank_marked(rendered_text)
    tokens = [(m.start(), m.group(0)) for m in _NUMBER_RE.finditer(blanked)]
    covered: set[int] = set()
    starts = {s: t for s, t in tokens}
    for entry in allowlist:
        if entry.doc != doc:
            continue
        offset = entry.context.index(entry.literal)
        hits = []
        pos = blanked.find(entry.context)
        while pos != -1:
            tok_start = pos + offset
            if starts.get(tok_start) == entry.literal:
                hits.append(tok_start)
            pos = blanked.find(entry.context, pos + len(entry.context))
        if len(hits) != entry.count:
            unused = "unused allowlist entry: " if not hits else "allowlist "
            raise LedgerError(
                f"{unused}{entry.literal!r} in {entry.doc!r}: expected {entry.count} "
                f"occurrence(s) in context {entry.context!r}, found {len(hits)}"
            )
        covered.update(hits)
    out = []
    for start, tok in tokens:
        if start in covered:
            continue
        line = blanked.count("\n", 0, start) + 1
        ctx = strip_marks(rendered_text[max(0, start - 20) : start + len(tok) + 20])
        out.append(UntracedNumber(tok, start, line, ctx.replace("\n", " ")))
    return out
