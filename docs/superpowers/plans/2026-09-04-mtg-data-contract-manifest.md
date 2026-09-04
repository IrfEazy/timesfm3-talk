# MTG Data Contract + Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MTG price ingestion reproducible and lossless: an explicit `--as-of` policy (no implicit `date.today()` in a full run), HTTP retry/atomic-download hardening that never turns a transient failure into silent missingness, an explicit price-selection policy that resolves `productId`/`subTypeName` ambiguity instead of last-row-wins, a JSON run manifest, and a reusable data-quality table/figure.

**Architecture:** All changes are additive to the existing `tfm3lab.data.mtg` contract — `PriceSelectionPolicy`/`PricePoint` change what a price *is*, retry/atomic-download change how bytes reach disk, and `IngestReport` carries what `build_card_series` learned out to a new `tfm3lab.manifest` module. `scripts/01_fetch_data.py` is the only production caller and gets the CLI policy layer; library functions keep permissive defaults so tests/notebooks are unaffected.

**Tech Stack:** Python 3.12, `requests` + `urllib3.util.retry.Retry` (HTTP retry), `py7zr` (archive extraction, already a dependency), `hashlib` (sha256), `pandas`/`numpy`, `matplotlib` (data-quality figure), `pytest` (offline fixtures only).

**Spec:** `docs/superpowers/specs/2026-09-04-mtg-data-contract-manifest-design.md`

## Global Constraints

- `uv` only — never `pip`/`uv pip`. Run tests via `uv run pytest`, lint via `uv run ruff check src tests scripts`.
- Default `pytest` run stays fully offline — no network, no HF, no GPU. Anything hitting a live service is opt-in (existing `TFM3LAB_RUN_LIVE_FETCH_SMOKE` convention in `tests/test_mtg_live.py`), unaffected by this plan except one call-site signature fix.
- Never modify committed `results/*.parquet` by hand.
- All metrics/coverage stats respect `observed=True` — the data-quality table's `fallback_rate` is exactly `1 - observed_rate`, not a separate ad-hoc definition.
- `windows.py`'s origin convention is untouched — this plan doesn't touch backtest/window code at all.
- Add a unit test for every bug/behavior fixed — no code change without a corresponding offline test.

---

## Task 1: `PriceSelectionPolicy` + subtype-aware price resolution (pure functions)

**Files:**
- Modify: `src/tfm3lab/data/mtg.py` (imports, `_price_from_row`, new `PriceSelectionPolicy`, `PricePoint`, `_resolve_subtype_row`)
- Test: `tests/test_mtg.py` (modify 4 existing `_price_from_row` tests, add new tests)

**Interfaces:**
- Produces: `PriceSelectionPolicy(enum.Enum)` with members `MARKET_THEN_MID`, `MARKET_ONLY`, `MID_ONLY`; `PricePoint` (frozen dataclass: `price: float`, `field_used: str`, `subtype: str | None`); `_price_from_row(row: dict, policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID) -> tuple[float | None, str | None]`; `_resolve_subtype_row(product_id: int, rows: list[dict], date: dt.date) -> dict`.
- Consumes: nothing new (pure functions over dicts).

- [ ] **Step 1: Write the failing tests**

Replace the four existing `_price_from_row` tests in `tests/test_mtg.py` (they currently assert a bare `float | None` return — the contract changes to `tuple[float | None, str | None]`) and add new ones. Replace this block:

```python
def test_price_from_row_prefers_market_price():
    assert _price_from_row({"marketPrice": 12.5, "midPrice": 99.0}) == 12.5


def test_price_from_row_falls_back_to_mid_when_market_is_null():
    assert _price_from_row({"marketPrice": None, "midPrice": 8.25}) == 8.25


def test_price_from_row_returns_none_when_both_missing():
    assert _price_from_row({"marketPrice": None, "midPrice": None}) is None


def test_price_from_row_ignores_low_and_high_price():
    # low/high are listing extremes, never a substitute for market/mid.
    row = {"marketPrice": None, "midPrice": None, "lowPrice": 1.0, "highPrice": 500.0}
    assert _price_from_row(row) is None
```

with:

```python
def test_price_from_row_prefers_market_price():
    assert _price_from_row({"marketPrice": 12.5, "midPrice": 99.0}) == (12.5, "market")


def test_price_from_row_falls_back_to_mid_when_market_is_null():
    assert _price_from_row({"marketPrice": None, "midPrice": 8.25}) == (8.25, "mid")


def test_price_from_row_returns_none_when_both_missing():
    assert _price_from_row({"marketPrice": None, "midPrice": None}) == (None, None)


def test_price_from_row_ignores_low_and_high_price():
    # low/high are listing extremes, never a substitute for market/mid.
    row = {"marketPrice": None, "midPrice": None, "lowPrice": 1.0, "highPrice": 500.0}
    assert _price_from_row(row) == (None, None)


def test_price_from_row_market_only_policy_does_not_fall_back_to_mid():
    row = {"marketPrice": None, "midPrice": 8.25}
    assert _price_from_row(row, PriceSelectionPolicy.MARKET_ONLY) == (None, None)


def test_price_from_row_mid_only_policy_ignores_market():
    row = {"marketPrice": 12.5, "midPrice": 8.25}
    assert _price_from_row(row, PriceSelectionPolicy.MID_ONLY) == (8.25, "mid")


def test_resolve_subtype_row_single_row_returned_directly():
    rows = [{"productId": 1, "subTypeName": "Normal", "marketPrice": 5.0}]
    assert _resolve_subtype_row(1, rows, dt.date(2024, 2, 8)) is rows[0]


def test_resolve_subtype_row_prefers_normal_among_foil_variant():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 30.0},
    ]
    resolved = _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
    assert resolved["subTypeName"] == "Normal"


def test_resolve_subtype_row_treats_missing_subtype_as_normal():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "marketPrice": 30.0},  # no subTypeName key at all
    ]
    resolved = _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
    assert resolved["marketPrice"] == 30.0


def test_resolve_subtype_row_raises_on_two_normal_rows():
    rows = [
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 5.0},
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 6.0},
    ]
    with pytest.raises(ValueError, match="ambiguous productId"):
        _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))


def test_resolve_subtype_row_raises_when_no_normal_among_multiple_non_normal():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "subTypeName": "Foil Etched", "marketPrice": 120.0},
    ]
    with pytest.raises(ValueError, match="ambiguous productId"):
        _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
```

Update the import line at the top of `tests/test_mtg.py` from:

```python
from tfm3lab.data.mtg import CardSpec, _price_from_row, resolve_card_specs
```

to:

```python
import datetime as dt

from tfm3lab.data.mtg import (
    CardSpec,
    PriceSelectionPolicy,
    _price_from_row,
    _resolve_subtype_row,
    resolve_card_specs,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mtg.py -v`
Expected: FAIL — `ImportError: cannot import name 'PriceSelectionPolicy'` (or `_resolve_subtype_row`).

- [ ] **Step 3: Implement**

In `src/tfm3lab/data/mtg.py`, add `import enum` to the import block (alphabetical, after `import difflib`):

```python
import datetime as dt
import difflib
import enum
import gzip
```

Right after the `CardSpec` dataclass definition (before `DEFAULT_CARDS`), add:

```python
class PriceSelectionPolicy(enum.Enum):
    """Which TCGplayer price field to trust for one row.

    MARKET_THEN_MID matches this project's original behavior: prefer
    marketPrice, fall back to midPrice when marketPrice is null (thin-volume
    days). MARKET_ONLY/MID_ONLY never fall back — a null preferred field
    means "no price today", not "use the other one".
    """

    MARKET_THEN_MID = "market_then_mid"
    MARKET_ONLY = "market_only"
    MID_ONLY = "mid_only"


@dataclass(frozen=True)
class PricePoint:
    """One resolved price observation: the value, which TCGplayer field it
    came from, and the subTypeName of the winning row (None if the archive
    didn't carry one)."""

    price: float
    field_used: str
    subtype: str | None
```

Replace the existing `_price_from_row` function:

```python
def _price_from_row(row: dict) -> float | None:
    """TCGplayer marketPrice, falling back to midPrice when marketPrice is
    null (happens on thin-volume days for less liquid printings). Never
    lowPrice/highPrice — those are listing extremes, not price estimates.
    """
    price = row.get("marketPrice")
    if price is None:
        price = row.get("midPrice")
    return float(price) if price is not None else None
```

with:

```python
def _price_from_row(
    row: dict, policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID
) -> tuple[float | None, str | None]:
    """Resolves one row to (price, field_used) per `policy`. Never
    lowPrice/highPrice — those are listing extremes, not price estimates.
    Returns (None, None) when the policy's field(s) are null.
    """
    market = row.get("marketPrice")
    mid = row.get("midPrice")
    if policy is PriceSelectionPolicy.MARKET_ONLY:
        return (float(market), "market") if market is not None else (None, None)
    if policy is PriceSelectionPolicy.MID_ONLY:
        return (float(mid), "mid") if mid is not None else (None, None)
    if market is not None:
        return float(market), "market"
    if mid is not None:
        return float(mid), "mid"
    return None, None


_PREFERRED_SUBTYPES = ("Normal", None)


def _resolve_subtype_row(product_id: int, rows: list[dict], date: dt.date) -> dict:
    """When one productId has multiple rows in a day's archive (distinct
    subTypeName — e.g. Normal vs Foil bundled together), picks the row this
    project tracks: subTypeName in ("Normal", None). Raises if that
    resolution isn't unique — no silent last-row-wins.
    """
    if len(rows) == 1:
        return rows[0]
    preferred = [r for r in rows if r.get("subTypeName") in _PREFERRED_SUBTYPES]
    if len(preferred) == 1:
        return preferred[0]
    subtypes_seen = [r.get("subTypeName") for r in rows]
    raise ValueError(
        f"ambiguous productId {product_id} on {date.isoformat()}: {len(rows)} rows, "
        f"subTypeName values {subtypes_seen} — no unique 'Normal' row to prefer"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mtg.py -v`
Expected: PASS (all `test_price_from_row_*` and `test_resolve_subtype_row_*` tests green; `test_resolve_card_specs_*` tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/data/mtg.py tests/test_mtg.py
git commit -m "feat: add configurable price-selection policy and subtype ambiguity resolution"
```

---

## Task 2: HTTP retry, atomic download, `ArchiveNotAvailableError`, `IngestReport`

**Files:**
- Modify: `src/tfm3lab/data/mtg.py` (imports, `fetch_groups`, `fetch_products`, `resolve_card_specs` default sessions, `fetch_daily_prices`, `build_card_series`)
- Modify: `tests/test_mtg_live.py:33-42` (unpack the new `(series, report)` return)
- Test: `tests/test_mtg.py` (new tests, offline)

**Interfaces:**
- Consumes: `PriceSelectionPolicy`, `PricePoint`, `_price_from_row`, `_resolve_subtype_row` from Task 1.
- Produces: `ArchiveNotAvailableError(Exception)`; `IngestReport` (frozen dataclass: `resolved_cards: pd.DataFrame`, `archive_hashes: dict[str, str]`, `price_field_counts: dict[str, int]`); `_session_with_retries() -> requests.Session`; `_sha256_file(path: Path) -> str`; `_download_archive_atomic(url: str, dest: Path, session) -> str | None`; `fetch_daily_prices(...) -> tuple[dict[int, dict[int, PricePoint]], str | None]` (return type changed — was `dict[int, dict[int, float]]`); `build_card_series(...) -> tuple[list[SeriesData], IngestReport]` (return type changed — was `list[SeriesData]`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mtg.py` (append at the end; add `import hashlib`, `import json`, `import requests` and `from pathlib import Path` to its top-of-file imports alongside the existing `pytest` import — `json`/`Path` are needed to build a tiny real `.7z` fixture for the two integration-style tests):

```python
import hashlib
import json
from pathlib import Path

from tfm3lab.data.mtg import (
    ArchiveNotAvailableError,
    _download_archive_atomic,
    _session_with_retries,
    fetch_daily_prices,
)


def test_session_with_retries_configures_backoff_for_429_and_5xx():
    session = _session_with_retries()
    adapter = session.get_adapter("https://tcgcsv.com")
    retry = adapter.max_retries
    assert retry.total == 3
    assert set(retry.status_forcelist) == {429, 500, 502, 503, 504}


class _FakeStreamResponse:
    def __init__(self, status_code, chunks=(b"",)):
        self.status_code = status_code
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size):
        yield from self._chunks


class _FakeDownloadSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, **kwargs):
        return self._response


def test_download_archive_atomic_returns_none_on_404(tmp_path):
    session = _FakeDownloadSession(_FakeStreamResponse(404))
    dest = tmp_path / "archive.7z"
    result = _download_archive_atomic("http://example.test/x", dest, session)
    assert result is None
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_archive_atomic_writes_final_file_and_returns_hash(tmp_path):
    session = _FakeDownloadSession(_FakeStreamResponse(200, chunks=[b"hello ", b"world"]))
    dest = tmp_path / "archive.7z"
    result = _download_archive_atomic("http://example.test/x", dest, session)
    assert dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()
    assert result == hashlib.sha256(b"hello world").hexdigest()
    assert dest.read_bytes() == b"hello world"


def test_download_archive_atomic_raises_on_5xx_and_leaves_no_final_file(tmp_path):
    session = _FakeDownloadSession(_FakeStreamResponse(500))
    dest = tmp_path / "archive.7z"
    with pytest.raises(requests.HTTPError):
        _download_archive_atomic("http://example.test/x", dest, session)
    assert not dest.exists()


def test_fetch_daily_prices_raises_archive_not_available_on_missing_archive(tmp_path, monkeypatch):
    import tfm3lab.data.mtg as mtg_module

    monkeypatch.setattr(mtg_module, "_download_archive_atomic", lambda url, dest, session: None)
    with pytest.raises(ArchiveNotAvailableError):
        fetch_daily_prices(dt.date(2024, 2, 8), {2809}, raw_dir=tmp_path, session=object())


def _write_archive(tmp_path: Path, date: dt.date, category_id: int, group_id: int, results: list):
    import py7zr

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"results": results}), encoding="utf-8")
    archive_path = tmp_path / f"tcgcsv-prices-{date.isoformat()}.ppmd.7z"
    with py7zr.SevenZipFile(archive_path, mode="w") as z:
        z.write(payload_path, f"{date.isoformat()}/{category_id}/{group_id}/prices")
    return archive_path


def test_fetch_daily_prices_resolves_normal_subtype_among_multiple_rows(tmp_path):
    date = dt.date(2024, 2, 8)
    _write_archive(
        tmp_path,
        date,
        1,
        2809,
        [
            {"productId": 239857, "subTypeName": "Foil", "marketPrice": 90.0},
            {"productId": 239857, "subTypeName": "Normal", "marketPrice": 30.0},
        ],
    )
    out, archive_sha256 = fetch_daily_prices(date, {2809}, raw_dir=tmp_path, session=object())
    assert out[2809][239857].price == 30.0
    assert out[2809][239857].subtype == "Normal"
    assert out[2809][239857].field_used == "market"
    assert archive_sha256 is not None


def test_fetch_daily_prices_raises_on_unresolvable_subtype_ambiguity(tmp_path):
    date = dt.date(2024, 2, 8)
    _write_archive(
        tmp_path,
        date,
        1,
        2809,
        [
            {"productId": 239857, "subTypeName": "Foil", "marketPrice": 90.0},
            {"productId": 239857, "subTypeName": "Foil Etched", "marketPrice": 120.0},
        ],
    )
    with pytest.raises(ValueError, match="ambiguous productId"):
        fetch_daily_prices(date, {2809}, raw_dir=tmp_path, session=object())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mtg.py -v`
Expected: FAIL — `ImportError: cannot import name 'ArchiveNotAvailableError'` (or similar for the other new names).

- [ ] **Step 3: Implement**

Add to the import block at the top of `src/tfm3lab/data/mtg.py`:

```python
import hashlib
import os
```

and after the `requests` import:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
```

After the `PricePoint` dataclass (from Task 1), add:

```python
class ArchiveNotAvailableError(Exception):
    """TCGCSV has no archive published for this date (404) — a legitimate
    gap, distinct from a transient failure."""

    def __init__(self, date: dt.date):
        self.date = date
        super().__init__(f"no TCGCSV archive available for {date.isoformat()}")


@dataclass(frozen=True)
class IngestReport:
    """What build_card_series learned while fetching, for the caller to pass
    into manifest.build_fetch_manifest — not part of SeriesData itself."""

    resolved_cards: pd.DataFrame
    archive_hashes: dict[str, str]
    price_field_counts: dict[str, int]


def _session_with_retries() -> requests.Session:
    """A session that retries 429/5xx with backoff before giving up — a
    transient failure must raise, not silently become a missing day."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive_atomic(url: str, dest: Path, session: requests.Session) -> str | None:
    """Streams `url` to `dest` atomically (temp `.part` file, hashed while
    streaming, then renamed) and returns the sha256 hex digest, or None on a
    404 (caller turns that into ArchiveNotAvailableError). Any other
    non-2xx status raises after the session's own retries are exhausted —
    no transient failure is silently swallowed here.
    """
    r = session.get(url, headers=_HEADERS, timeout=180, stream=True)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with open(part, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            digest.update(chunk)
    os.replace(part, dest)
    return digest.hexdigest()
```

Change the three `session or requests.Session()` defaults in `fetch_groups`, `fetch_products`, and `resolve_card_specs` to `session or _session_with_retries()`.

Replace `fetch_daily_prices` entirely:

```python
def fetch_daily_prices(
    date: dt.date,
    group_ids: set[int],
    category_id: int = MAGIC_CATEGORY_ID,
    session: requests.Session | None = None,
    raw_dir: Path = config.RAW_DIR,
    price_policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID,
) -> tuple[dict[int, dict[int, PricePoint]], str | None]:
    """Downloads one day's archive (cached under raw_dir, atomic write +
    sha256), extracts only the requested groups' price files, and returns
    ({group_id: {product_id: PricePoint}}, archive_sha256).

    Raises ArchiveNotAvailableError on a 404 (legitimate gap — TCGCSV never
    published this date). Any other HTTP failure, after the session's own
    retries are exhausted, propagates as requests.HTTPError — a transient
    429/5xx must not silently become a missing day.

    When multiple rows share a productId in one day's file (distinct
    subTypeName — e.g. Normal vs Foil bundled together), the row is resolved
    by preferring subTypeName in ("Normal", None); anything left ambiguous
    raises ValueError naming the productId/date/subtypes seen.
    """
    import py7zr

    session = session or _session_with_retries()
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / f"tcgcsv-prices-{date.isoformat()}.ppmd.7z"
    if archive_path.exists():
        archive_sha256 = _sha256_file(archive_path)
    else:
        archive_sha256 = _download_archive_atomic(_archive_url(date), archive_path, session)
        if archive_sha256 is None:
            raise ArchiveNotAvailableError(date)

    targets = [f"{date.isoformat()}/{category_id}/{gid}/prices" for gid in group_ids]
    out: dict[int, dict[int, PricePoint]] = {gid: {} for gid in group_ids}
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        present = [t for t in targets if t in set(z.getnames())]
        if not present:
            return out, archive_sha256
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(path=tmp, targets=present)
            for target in present:
                group_id = int(target.split("/")[2])
                payload = json.loads((Path(tmp) / target).read_text(encoding="utf-8"))
                rows_by_product: dict[int, list[dict]] = {}
                for row in payload.get("results", []):
                    rows_by_product.setdefault(int(row["productId"]), []).append(row)
                for product_id, rows in rows_by_product.items():
                    winning = _resolve_subtype_row(product_id, rows, date)
                    price, field_used = _price_from_row(winning, price_policy)
                    if price is not None:
                        out[group_id][product_id] = PricePoint(
                            price=price, field_used=field_used, subtype=winning.get("subTypeName")
                        )
    return out, archive_sha256
```

Replace `build_card_series` entirely:

```python
def build_card_series(
    cards: tuple[CardSpec, ...] = DEFAULT_CARDS,
    start: dt.date = TCGCSV_ARCHIVE_START,
    end: dt.date | None = None,
    raw_dir: Path = config.RAW_DIR,
    max_ffill_days: int = 3,
    session: requests.Session | None = None,
    price_policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID,
) -> tuple[list[SeriesData], IngestReport]:
    """Builds one SeriesData per card over [start, end], daily frequency, and
    an IngestReport (resolved card specs, archive sha256 per date, price
    field usage counts) for the caller to pass into
    manifest.build_fetch_manifest.

    Missing days (archive gaps, or the product simply unlisted that day) are
    forward-filled up to `max_ffill_days`; `SeriesData.observed` marks which
    points are real vs. filled, per the project's rule that filled points
    must never be scored as if they were observations.

    `end` defaults to `date.today()` for library callers (notebooks, tests)
    — a full experiment run must pass an explicit `end` (see
    scripts/01_fetch_data.py's --as-of/--allow-live-end policy); that
    constraint is enforced at the script layer, not here.
    """
    end = end or dt.date.today()
    session = session or _session_with_retries()
    resolved = resolve_card_specs(cards, session=session)
    group_ids = set(resolved["group_id"])

    date_range = pd.date_range(start, end, freq="D")
    raw = pd.DataFrame(index=date_range, columns=resolved["label"], dtype=float)

    archive_hashes: dict[str, str] = {}
    price_field_counts: dict[str, int] = {"market": 0, "mid": 0}

    for date in date_range:
        day = date.date()
        try:
            prices_by_group, archive_sha256 = fetch_daily_prices(
                day, group_ids, raw_dir=raw_dir, session=session, price_policy=price_policy
            )
        except ArchiveNotAvailableError:
            continue  # archive missing for this day — left as NaN, filled below
        if archive_sha256 is not None:
            archive_hashes[day.isoformat()] = archive_sha256
        for _, row in resolved.iterrows():
            point = prices_by_group.get(row["group_id"], {}).get(row["product_id"])
            if point is not None:
                raw.loc[date, row["label"]] = point.price
                price_field_counts[point.field_used] += 1

    observed = raw.notna()
    filled = raw.ffill(limit=max_ffill_days)

    series_list = []
    for label in resolved["label"]:
        series_list.append(
            SeriesData(
                name=label,
                values=filled[label].to_numpy(dtype=float),
                dates=filled.index.to_numpy(),
                observed=observed[label].to_numpy(),
            )
        )
    report = IngestReport(
        resolved_cards=resolved, archive_hashes=archive_hashes, price_field_counts=price_field_counts
    )
    return series_list, report
```

Finally, update `tests/test_mtg_live.py:33-42` to unpack the new tuple return:

```python
def test_build_card_series_returns_observed_data_for_a_small_window():
    series, report = build_card_series(
        cards=DEFAULT_CARDS[:2],
        start=dt.date(2024, 2, 8),
        end=dt.date(2024, 2, 14),
    )
    assert len(series) == 2
    for s in series:
        assert len(s.values) == 7
        assert s.observed.all()  # this window is known-good, no archive gaps
    assert report.resolved_cards["label"].tolist() == [c.label for c in DEFAULT_CARDS[:2]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mtg.py -v`
Expected: PASS — all tests including the new retry/atomic-download/ambiguity ones.

Run: `uv run ruff check src/tfm3lab/data/mtg.py tests/test_mtg.py tests/test_mtg_live.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/data/mtg.py tests/test_mtg.py tests/test_mtg_live.py
git commit -m "feat: harden MTG HTTP fetch with retry, atomic download, and explicit 404 handling"
```

---

## Task 3: Reproducibility manifest (`src/tfm3lab/manifest.py`)

**Files:**
- Create: `src/tfm3lab/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (pure, independent module — takes plain dicts/DataFrames).
- Produces: `write_manifest(payload: dict, path: Path, *, as_of: dt.date, live_end: bool = False) -> Path`; `build_fetch_manifest(*, date_range: tuple[dt.date, dt.date], resolved_cards: pd.DataFrame, archive_hashes: dict[str, str], price_field_counts: dict[str, int], coverage_stats: list[dict]) -> dict`. Both consumed by Task 7 (`scripts/01_fetch_data.py`), fed with `IngestReport` fields from Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
"""Unit tests for tfm3lab.manifest — pure functions, no network."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from tfm3lab.manifest import build_fetch_manifest, write_manifest


def test_write_manifest_includes_common_meta_fields(tmp_path):
    path = write_manifest(
        {"hello": "world"}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1), live_end=False
    )
    data = json.loads(path.read_text())
    assert data["hello"] == "world"
    assert data["_meta"]["as_of"] == "2026-09-01"
    assert data["_meta"]["live_end"] is False
    assert "git_sha" in data["_meta"]
    assert "written_at_utc" in data["_meta"]
    assert set(data["_meta"]["package_versions"]) == {"requests", "py7zr", "pandas"}


def test_write_manifest_creates_parent_directories(tmp_path):
    path = write_manifest({}, tmp_path / "nested" / "m.json", as_of=dt.date(2026, 9, 1))
    assert path.exists()


def test_build_fetch_manifest_computes_price_field_pct():
    payload = build_fetch_manifest(
        date_range=(dt.date(2024, 2, 8), dt.date(2026, 9, 1)),
        resolved_cards=pd.DataFrame([{"label": "A", "group_id": 1, "product_id": 2}]),
        archive_hashes={"2024-02-08": "abc123"},
        price_field_counts={"market": 3, "mid": 1},
        coverage_stats=[{"series": "A", "observed_rate": 0.9, "fallback_rate": 0.1}],
    )
    assert payload["price_field_used_pct"] == {"market": 0.75, "mid": 0.25}
    assert payload["date_range"] == {"start": "2024-02-08", "end": "2026-09-01"}
    assert payload["archive_hashes"] == {"2024-02-08": "abc123"}
    assert payload["resolved_cards"] == [{"label": "A", "group_id": 1, "product_id": 2}]
    assert payload["coverage"][0]["series"] == "A"


def test_build_fetch_manifest_empty_price_field_counts_gives_empty_pct():
    payload = build_fetch_manifest(
        date_range=(dt.date(2024, 2, 8), dt.date(2024, 2, 8)),
        resolved_cards=pd.DataFrame(columns=["label"]),
        archive_hashes={},
        price_field_counts={},
        coverage_stats=[],
    )
    assert payload["price_field_used_pct"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tfm3lab.manifest'`.

- [ ] **Step 3: Implement**

Create `src/tfm3lab/manifest.py`:

```python
"""Reproducibility manifest for tfm3lab data-fetch (and, later, model) runs.

write_manifest() carries the parts every run shares (git SHA, as_of,
live_end, timestamp, package versions); callers merge in a payload built by
a per-script function like build_fetch_manifest() below. JSON, not parquet
— small, human-diffable, meant to be read next to the results/ artifact it
describes, not queried at scale.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

_TRACKED_PACKAGES = ("requests", "py7zr", "pandas")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def write_manifest(
    payload: dict, path: Path, *, as_of: dt.date, live_end: bool = False
) -> Path:
    """Merges `payload` under a common `_meta` block and writes indented
    JSON to `path` (parent directories created as needed)."""
    manifest = {
        "_meta": {
            "git_sha": _git_sha(),
            "as_of": as_of.isoformat(),
            "live_end": live_end,
            "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "package_versions": _package_versions(),
        },
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def build_fetch_manifest(
    *,
    date_range: tuple[dt.date, dt.date],
    resolved_cards: pd.DataFrame,
    archive_hashes: dict[str, str],
    price_field_counts: dict[str, int],
    coverage_stats: list[dict],
) -> dict:
    """MTG-specific fetch payload: date range, resolved card specs, archive
    hashes, price-field usage %, and per-card observed/forward-filled %
    (`coverage_stats` — see figdata.data_quality_table, which computes the
    same numbers for the data-quality figure; this function doesn't
    recompute them, just carries what the caller already has).
    """
    total_points = sum(price_field_counts.values())
    price_field_pct = (
        {k: v / total_points for k, v in price_field_counts.items()} if total_points else {}
    )
    return {
        "date_range": {"start": date_range[0].isoformat(), "end": date_range[1].isoformat()},
        "resolved_cards": resolved_cards.to_dict(orient="records"),
        "archive_hashes": archive_hashes,
        "price_field_used_pct": price_field_pct,
        "coverage": coverage_stats,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS.

Run: `uv run ruff check src/tfm3lab/manifest.py tests/test_manifest.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/manifest.py tests/test_manifest.py
git commit -m "feat: add JSON reproducibility manifest for data-fetch runs"
```

---

## Task 4: `--as-of` support in `build_cpi_series` (`macro.py`)

**Files:**
- Modify: `src/tfm3lab/data/macro.py`
- Test: `tests/test_macro.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_cpi_series(url: str = FRED_CPI_URL, end: dt.date | None = None) -> SeriesData` (new optional `end` param, backward compatible — omitting it keeps today's full-history behavior).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_macro.py` (add `import datetime as dt` and `from tfm3lab.data import macro` to its imports):

```python
import datetime as dt

import pandas as pd
import pytest

from tfm3lab.data import macro
from tfm3lab.data.macro import compute_yoy


def test_build_cpi_series_trims_to_end_date(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    yoy = pd.Series([float(i) for i in range(15)], index=dates)
    monkeypatch.setattr(macro, "fetch_cpi_yoy", lambda url=macro.FRED_CPI_URL: yoy)

    series = macro.build_cpi_series(end=dt.date(2020, 10, 1))

    assert pd.Timestamp(series.dates[-1]) <= pd.Timestamp("2020-10-01")
    assert len(series.values) == 10


def test_build_cpi_series_without_end_keeps_full_series(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    yoy = pd.Series([1.0] * 5, index=dates)
    monkeypatch.setattr(macro, "fetch_cpi_yoy", lambda url=macro.FRED_CPI_URL: yoy)

    series = macro.build_cpi_series()

    assert len(series.values) == 5
```

(Keep the existing `test_compute_yoy_*` tests below unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_macro.py -v`
Expected: FAIL — `TypeError: build_cpi_series() got an unexpected keyword argument 'end'`.

- [ ] **Step 3: Implement**

In `src/tfm3lab/data/macro.py`, add `import datetime as dt` to the imports (after `from __future__ import annotations`), then replace `build_cpi_series`:

```python
def build_cpi_series(url: str = FRED_CPI_URL, end: dt.date | None = None) -> SeriesData:
    yoy = fetch_cpi_yoy(url)
    if end is not None:
        yoy = yoy[yoy.index.date <= end]
    return SeriesData(
        name="CPI_YoY",
        values=yoy.to_numpy(dtype=float),
        dates=yoy.index.to_numpy(),
        observed=np.ones(len(yoy), dtype=bool),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_macro.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/data/macro.py tests/test_macro.py
git commit -m "feat: let build_cpi_series trim to an explicit end date"
```

---

## Task 5: Data-quality table (`figdata.py`)

**Files:**
- Modify: `src/tfm3lab/figdata.py`
- Test: `tests/test_figdata.py`

**Interfaces:**
- Consumes: `SeriesData` (existing, from `tfm3lab.backtest`), `find_glitches` (existing, same file).
- Produces: `data_quality_table(series_list: list[SeriesData]) -> pd.DataFrame` with columns `series`, `observed_rate`, `fallback_rate`, `max_gap_days`, `glitch_count`, `price_min`, `price_max`, `log_return_volatility`. Consumed by Task 6 (`plots.plot_data_quality`) and Task 7 (`scripts/01_fetch_data.py`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_figdata.py` (add `from tfm3lab.backtest import SeriesData` to its imports):

```python
from tfm3lab.backtest import SeriesData


def _series(name, values, observed, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(values)).to_numpy()
    return SeriesData(
        name=name, values=np.array(values, dtype=float), dates=dates, observed=np.array(observed)
    )


def test_data_quality_table_observed_and_fallback_rates():
    s = _series("A", [10.0] * 8, [True, True, False, False, True, True, True, True])
    table = figdata.data_quality_table([s])
    row = table.iloc[0]
    assert row["series"] == "A"
    assert row["observed_rate"] == pytest.approx(6 / 8)
    assert row["fallback_rate"] == pytest.approx(2 / 8)


def test_data_quality_table_max_gap_days_is_longest_unobserved_run():
    observed = [True, False, False, False, True, False, True]
    s = _series("A", [10.0] * 7, observed)
    table = figdata.data_quality_table([s])
    assert table.iloc[0]["max_gap_days"] == 3


def test_data_quality_table_glitch_count_reuses_find_glitches():
    values = [10.0, 10.0, 20.0, 10.0, 10.0]  # spike-and-revert at index 2
    s = _series("A", values, [True] * 5)
    table = figdata.data_quality_table([s])
    assert table.iloc[0]["glitch_count"] == 1


def test_data_quality_table_price_range_and_volatility():
    s = _series("A", [10.0, 20.0, 10.0], [True, True, True])
    table = figdata.data_quality_table([s])
    row = table.iloc[0]
    assert row["price_min"] == 10.0
    assert row["price_max"] == 20.0
    assert row["log_return_volatility"] > 0


def test_data_quality_table_one_row_per_series():
    s1 = _series("A", [10.0, 10.0], [True, True])
    s2 = _series("B", [5.0, 5.0], [True, True])
    table = figdata.data_quality_table([s1, s2])
    assert list(table["series"]) == ["A", "B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_figdata.py -k data_quality -v`
Expected: FAIL — `AttributeError: module 'tfm3lab.figdata' has no attribute 'data_quality_table'`.

- [ ] **Step 3: Implement**

In `src/tfm3lab/figdata.py`, add this after `find_glitches` (before the `ForecastSlice` dataclass):

```python
def _max_consecutive_false(mask: np.ndarray) -> int:
    """Longest run of False (unobserved) entries in a boolean array."""
    best = current = 0
    for observed in mask:
        if observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def data_quality_table(series_list: list) -> pd.DataFrame:
    """Per-card data-quality summary: observed rate, forward-fill rate, the
    longest run of consecutive unobserved points (assumes daily frequency —
    true for build_card_series's output, the only caller), glitch count
    (reusing find_glitches against the raw, not truth-reconstructed, price
    frame), price range, and log-return volatility restricted to observed
    points.
    """
    frames = [
        pd.DataFrame({"series": s.name, "index": np.arange(len(s.values)), "date": s.dates, "value": s.values})
        for s in series_list
    ]
    raw = pd.concat(frames, ignore_index=True)
    glitch_counts = find_glitches(raw).groupby("series").size()

    rows = []
    for s in series_list:
        n = len(s.values)
        observed_rate = float(np.mean(s.observed)) if n else float("nan")
        fallback_rate = 1.0 - observed_rate if n else float("nan")
        obs_values = s.values[s.observed]
        if s.observed.sum() >= 2:
            volatility = float(np.std(np.diff(np.log(obs_values))))
        else:
            volatility = float("nan")
        rows.append(
            {
                "series": s.name,
                "observed_rate": observed_rate,
                "fallback_rate": fallback_rate,
                "max_gap_days": _max_consecutive_false(s.observed),
                "glitch_count": int(glitch_counts.get(s.name, 0)),
                "price_min": float(np.min(s.values)) if n else float("nan"),
                "price_max": float(np.max(s.values)) if n else float("nan"),
                "log_return_volatility": volatility,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_figdata.py -v`
Expected: PASS (all tests, including the pre-existing ones — unaffected).

Run: `uv run ruff check src/tfm3lab/figdata.py tests/test_figdata.py`
Expected: no errors (watch line length — wrap the `frames = [...]` comprehension if ruff flags E501 at 100 cols).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/figdata.py tests/test_figdata.py
git commit -m "feat: add reusable per-card data-quality table"
```

---

## Task 6: Data-quality figure (`plots.py`)

**Files:**
- Modify: `src/tfm3lab/plots.py`
- Test: `tests/test_plots_smoke.py`

**Interfaces:**
- Consumes: the `pd.DataFrame` shape produced by Task 5's `data_quality_table` (columns `series`, `observed_rate`, `fallback_rate`, `glitch_count`, plus the others, unused by the plot).
- Produces: `plot_data_quality(table: pd.DataFrame, axes=None)` → returns a length-3 array/tuple of `Axes`. Consumed by Task 7.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_plots_smoke.py`:

```python
def test_plot_data_quality_three_panels_one_bar_per_series():
    table = pd.DataFrame(
        {
            "series": ["A", "B"],
            "observed_rate": [0.9, 0.7],
            "fallback_rate": [0.1, 0.3],
            "max_gap_days": [1, 3],
            "glitch_count": [0, 2],
            "price_min": [1.0, 2.0],
            "price_max": [10.0, 20.0],
            "log_return_volatility": [0.01, 0.05],
        }
    )
    axes = plots.plot_data_quality(table)
    assert len(axes) == 3
    assert len(axes[0].patches) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots_smoke.py -k data_quality -v`
Expected: FAIL — `AttributeError: module 'tfm3lab.plots' has no attribute 'plot_data_quality'`.

- [ ] **Step 3: Implement**

In `src/tfm3lab/plots.py`, add this after `plot_glitch_vignette` (before `save`):

```python
def plot_data_quality(table: pd.DataFrame, axes=None):
    """Three-panel bar chart, one bar per card/series: observed rate,
    forward-fill rate, and glitch count. Pairs with
    figdata.data_quality_table."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax0, ax1, ax2 = axes
    labels = table["series"]

    ax0.bar(labels, table["observed_rate"], color=PALETTE["model"])
    ax0.set_title("Observed rate")
    ax0.set_ylim(0, 1)

    ax1.bar(labels, table["fallback_rate"], color=PALETTE["baseline"])
    ax1.set_title("Forward-fill rate")
    ax1.set_ylim(0, 1)

    ax2.bar(labels, table["glitch_count"], color=PALETTE["alert"])
    ax2.set_title("Glitch count")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    return axes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: PASS (all tests, including pre-existing ones).

Run: `uv run ruff check src/tfm3lab/plots.py tests/test_plots_smoke.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/plots.py tests/test_plots_smoke.py
git commit -m "feat: add data-quality figure paired with figdata.data_quality_table"
```

---

## Task 7: `--as-of`/`--allow-live-end` CLI + wiring (`scripts/01_fetch_data.py`)

**Files:**
- Modify: `scripts/01_fetch_data.py`
- Test: `tests/test_fetch_data_cli.py` (new)

**Interfaces:**
- Consumes: `build_card_series(...) -> tuple[list[SeriesData], IngestReport]` and `PriceSelectionPolicy` (Task 2), `build_market_series(end=...)` (existing, unchanged), `build_cpi_series(end=...)` (Task 4), `figdata.data_quality_table` (Task 5), `plots.plot_data_quality`/`plots.save`/`plots.apply_style` (Task 6, existing), `manifest.write_manifest`/`manifest.build_fetch_manifest` (Task 3).
- Produces: `build_parser() -> argparse.ArgumentParser`, `resolve_as_of(args: argparse.Namespace) -> tuple[dt.date, bool]` — both importable by the test via the existing `importlib.util.spec_from_file_location` pattern (see `tests/test_exp_covariates_leakage.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fetch_data_cli.py`:

```python
"""CLI-parsing tests for scripts/01_fetch_data.py — argparse only, no
network, no fetch. Loaded via importlib rather than a package import:
scripts/ are thin CLIs, not part of the tfm3lab package — same pattern as
tests/test_exp_covariates_leakage.py.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "01_fetch_data.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("fetch_data_01", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch01():
    return _load_script_module()


def test_as_of_and_allow_live_end_are_mutually_exclusive(fetch01):
    parser = fetch01.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--as-of", "2026-09-01", "--allow-live-end"])


def test_resolve_as_of_requires_one_of_the_two_flags(fetch01):
    parser = fetch01.build_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        fetch01.resolve_as_of(args)


def test_resolve_as_of_parses_explicit_date(fetch01):
    parser = fetch01.build_parser()
    args = parser.parse_args(["--as-of", "2026-09-01"])
    as_of, live_end = fetch01.resolve_as_of(args)
    assert as_of == dt.date(2026, 9, 1)
    assert live_end is False


def test_resolve_as_of_allow_live_end_uses_today_and_warns(fetch01, capsys):
    parser = fetch01.build_parser()
    args = parser.parse_args(["--allow-live-end"])
    as_of, live_end = fetch01.resolve_as_of(args)
    assert as_of == dt.date.today()
    assert live_end is True
    assert "WARNING" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_data_cli.py -v`
Expected: FAIL — `AttributeError: module 'fetch_data_01' has no attribute 'build_parser'`.

- [ ] **Step 3: Implement**

Replace `scripts/01_fetch_data.py` in full:

```python
#!/usr/bin/env python3
"""Fetch and cache the raw data every experiment reads.

  - MTG card prices (TCGCSV, full history since 2024-02-08 by default)
  - Market series: S&P 500, VIX, gold, oil (yfinance)
  - CPI YoY inflation (FRED)

Idempotent: TCGCSV's own daily archives are cached individually under
data/raw/ (re-running only downloads missing days); this script's parquet
outputs under data/cache/ are simply overwritten each run.

The full MTG backfill (~2.5 years, ~900+ daily archives) takes a while on
first run — use --mtg-start to fetch a shorter window (e.g. for a quick
local sanity check before committing to the full history on Colab).

A full run must freeze its end date with --as-of YYYY-MM-DD, recorded in
the run's manifest — --allow-live-end (dev only) uses today() instead and
is not reproducible.

Usage:
    uv run scripts/01_fetch_data.py --as-of 2026-09-01
    uv run scripts/01_fetch_data.py --as-of 2026-09-01 --mtg-start 2026-08-01  # quick check
    uv run scripts/01_fetch_data.py --as-of 2026-09-01 --skip-mtg             # market+CPI only
    uv run scripts/01_fetch_data.py --allow-live-end                         # dev only
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config, figdata, plots
from tfm3lab.backtest import SeriesData
from tfm3lab.data.macro import build_cpi_series
from tfm3lab.data.market import build_market_series
from tfm3lab.data.mtg import TCGCSV_ARCHIVE_START, PriceSelectionPolicy, build_card_series
from tfm3lab.manifest import build_fetch_manifest, write_manifest

plots.apply_style()


def _series_list_to_frame(series_list: list[SeriesData]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {"date": s.dates, "value": s.values, "observed": s.observed, "series": s.name}
        )
        for s in series_list
    ]
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-mtg", action="store_true", help="skip the (slow) MTG/TCGCSV ingest")
    parser.add_argument(
        "--mtg-start",
        default=None,
        help="override MTG start date (YYYY-MM-DD); default is the full TCGCSV history",
    )
    end_group = parser.add_mutually_exclusive_group()
    end_group.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="freeze the run's end date for reproducibility; required unless --allow-live-end",
    )
    end_group.add_argument(
        "--allow-live-end",
        action="store_true",
        help="use today() as the end date (dev only) — not reproducible, "
        "records live_end=true in the manifest",
    )
    return parser


def resolve_as_of(args: argparse.Namespace) -> tuple[dt.date, bool]:
    """Returns (as_of, live_end). Exits (SystemExit) if neither --as-of nor
    --allow-live-end was given — a full run must not silently fall back to
    today()."""
    if args.allow_live_end:
        print(
            "WARNING: --allow-live-end — using today() as the run's end date; "
            "not reproducible, do not use for a full/final experiment run.",
            file=sys.stderr,
        )
        return dt.date.today(), True
    if args.as_of:
        return dt.date.fromisoformat(args.as_of), False
    raise SystemExit(
        "error: one of --as-of YYYY-MM-DD or --allow-live-end is required for a full fetch run"
    )


def main() -> None:
    args = build_parser().parse_args()
    as_of, live_end = resolve_as_of(args)
    utc_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not args.skip_mtg:
        start = dt.date.fromisoformat(args.mtg_start) if args.mtg_start else TCGCSV_ARCHIVE_START
        print(f"Fetching MTG card prices from TCGCSV, {start} .. {as_of} (as_of)...")
        mtg_series, ingest_report = build_card_series(
            start=start, end=as_of, price_policy=PriceSelectionPolicy.MARKET_THEN_MID
        )
        mtg_df = _series_list_to_frame(mtg_series)
        mtg_df.to_parquet(config.CACHE_DIR / "mtg_prices.parquet", index=False)
        print(
            f"  {len(mtg_series)} cards, {mtg_df['date'].min()} .. {mtg_df['date'].max()}, "
            f"{mtg_df['observed'].mean():.1%} observed (rest forward-filled)"
        )

        quality_table = figdata.data_quality_table(mtg_series)
        quality_table.to_parquet(config.RESULTS_DIR / "mtg_data_quality.parquet", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plots.plot_data_quality(quality_table, axes=axes)
        plots.save(fig, "mtg_data_quality")
        print(f"  data-quality table + figure written under {config.RESULTS_DIR}")

        manifest_payload = build_fetch_manifest(
            date_range=(start, as_of),
            resolved_cards=ingest_report.resolved_cards,
            archive_hashes=ingest_report.archive_hashes,
            price_field_counts=ingest_report.price_field_counts,
            coverage_stats=quality_table[["series", "observed_rate", "fallback_rate"]].to_dict(
                orient="records"
            ),
        )
        manifest_path = write_manifest(
            manifest_payload,
            config.RESULTS_DIR / "manifests" / f"fetch-{as_of.isoformat()}-{utc_ts}.json",
            as_of=as_of,
            live_end=live_end,
        )
        print(f"  manifest written to {manifest_path}")
    else:
        print("Skipping MTG ingest (--skip-mtg)")

    print("Fetching market series (S&P 500, VIX, gold, oil) via yfinance...")
    market_series = build_market_series(end=as_of.isoformat())
    market_df = _series_list_to_frame(market_series)
    market_df.to_parquet(config.CACHE_DIR / "market_prices.parquet", index=False)
    print(
        f"  {len(market_series)} tickers, {market_df['date'].min()} .. {market_df['date'].max()}, "
        f"{len(market_series[0].values)} trading days each"
    )

    print("Fetching CPI YoY from FRED...")
    cpi = build_cpi_series(end=as_of)
    cpi_df = _series_list_to_frame([cpi])
    cpi_df.to_parquet(config.CACHE_DIR / "cpi_yoy.parquet", index=False)
    print(f"  {len(cpi.values)} months, {cpi_df['date'].min()} .. {cpi_df['date'].max()}")

    print(f"\nAll caches written to {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_data_cli.py -v`
Expected: PASS.

Run: `uv run pytest -v` (full suite)
Expected: PASS — every test in `tests/`, offline. Live-gated tests (`test_mtg_live.py`, `test_data_live.py`, `test_model_smoke.py`) remain skipped by default.

Run: `uv run ruff check src tests scripts`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/01_fetch_data.py tests/test_fetch_data_cli.py
git commit -m "feat: require explicit --as-of (or --allow-live-end) for MTG/market/CPI fetch"
```

---

## Final Report (write after Task 7's commit, do not commit it as part of this plan)

At the end of execution, write a short report (chat message, not a new file unless asked) covering:
- Files changed (the list above).
- Tests run and their result (`uv run pytest -v`, `uv run ruff check src tests scripts` — paste the summary line of each).
- Residual risks: `results/mtg_prices.parquet` committed today predates `as_of`/manifest provenance and is not regenerated by this plan (no live fetch performed); `scripts/02`-`05` don't yet consume a manifest (out of scope, `manifest.write_manifest`'s core is ready for them); the one real `.7z`-fixture integration test in Task 2 covers the happy path + one ambiguity path through `fetch_daily_prices`, not every combination (unit tests on `_resolve_subtype_row`/`_price_from_row` cover the rest).
- Migration notes for old results: none required for `results/mtg_prices.parquet`'s schema (unchanged); a real re-fetch (opt-in, live) would be needed to get `as_of`/manifest provenance for the currently committed cache.
