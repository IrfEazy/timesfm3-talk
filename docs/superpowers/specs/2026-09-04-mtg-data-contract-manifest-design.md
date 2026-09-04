# MTG data contract + reproducible run manifest — design

Status: approved for planning (2026-09-04). Base commit: d6714b7.
Source: `openai.review.md`, "Prompt 2 — Data contract e riproducibilità dei dati MTG".

## Problem

`src/tfm3lab/data/mtg.py` ingestion has three implicit-contract risks that make a
full MTG fetch non-reproducible and silently lossy:

1. `build_card_series(end=None)` defaults to `dt.date.today()` — a full-history
   fetch run today and the same fetch run tomorrow silently cover different
   date ranges, and nothing records which range a given `results/` artifact
   used.
2. `fetch_daily_prices()` catches every `requests.HTTPError` the same way and
   treats it as "day missing" — a transient 429/5xx becomes indistinguishable
   from a legitimate 404 (archive genuinely never published for that date),
   and both become forward-filled missingness with no trace of *why*.
3. Price lookup keys on `productId` alone. TCGCSV's daily archive can carry
   multiple rows for one `productId` (distinct `subTypeName`, e.g. Normal vs
   Foil bundled under a shared id in some sets); today the last row read from
   the JSON wins, silently, and `subTypeName`/`price_field_used` are never
   persisted per point.

None of this is visible from `results/mtg_prices.parquet` alone — there is no
manifest tying a cached parquet back to the archive hashes, git SHA, or `as_of`
date that produced it.

## Scope

In scope: `src/tfm3lab/data/mtg.py`, a new `src/tfm3lab/manifest.py`, CLI
changes to `scripts/01_fetch_data.py` (and the `--as-of`/`--allow-live-end`
plumbing into `market.py`'s `end` param and a post-fetch trim in `macro.py`),
a new data-quality table/figure pair in `figdata.py`/`plots.py`.

Out of scope: `scripts/02`–`05` (model-run manifests — hardware/inference
flags). `write_manifest()`'s core is built generic enough for those scripts to
adopt later without rework, but they are not touched by this change.

## Architecture

```
scripts/01_fetch_data.py
  --as-of YYYY-MM-DD  XOR  --allow-live-end
        |
        v
  tfm3lab.data.mtg.build_card_series(end=as_of, price_policy=..., session=...)
        |                                   \
        v                                    v
  per-day: fetch_daily_prices()        tfm3lab.manifest.build_fetch_manifest(...)
    - retrying session (429/5xx)             |
    - 404 -> ArchiveNotAvailableError         v
    - atomic .part + sha256 download    results/manifests/fetch-<as_of>-<ts>.json
    - dedup (productId, subTypeName)
        |
        v
  results/mtg_prices.parquet
  results/mtg_data_quality.parquet     (figdata.data_quality_table)
  results/figures/mtg_data_quality.png (plots.plot_data_quality)
```

## Components

### 1. `--as-of` / `--allow-live-end` (`scripts/01_fetch_data.py`)

- `argparse` mutually-exclusive group, one of the two required whenever the
  script is not run with `--skip-mtg` *and* market/CPI fetch is happening
  (i.e. required for any full run). `--allow-live-end` prints a `WARNING:`
  line to stderr and sets `as_of = date.today()`, `live_end = True`.
- `as_of` is threaded into all three fetches:
  - MTG: `build_card_series(end=as_of, ...)`.
  - Market: `build_market_series(end=as_of.isoformat(), ...)` (already accepts
    a string `end`).
  - CPI: FRED has no query-side end date; fetch as today, then
    `yoy = yoy[yoy.index.date <= as_of]` before building `SeriesData`.
- `--mtg-start` is unchanged, orthogonal (start-side dev override).
- Library functions (`build_card_series`, `build_market_series`) keep
  `end: date | None = None` defaulting to "today" — the *policy* that a full
  experiment run must not do that lives in the script/CLI layer, not the
  library, so tests and ad-hoc notebook use aren't forced through the flag.

### 2. HTTP hardening (`mtg.py`)

- New `_session_with_retries() -> requests.Session`: mounts
  `HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET", "HEAD"})))`
  on `http://` and `https://`. `build_card_series` uses this by default when
  no `session` is injected (tests keep injecting fakes, unaffected).
- New `class ArchiveNotAvailableError(Exception)`: raised by
  `fetch_daily_prices` specifically on a 404 from the archive URL — a legit
  "TCGCSV never published this date" case, caught by `build_card_series` and
  left as `NaN` (existing forward-fill behavior).
- Any other `requests.HTTPError` (i.e. retries exhausted on 429/5xx, or an
  unexpected 4xx) propagates out of `build_card_series` uncaught — the script
  aborts with a clear traceback instead of manufacturing missingness.
- Atomic download in `fetch_daily_prices`: write archive bytes to
  `archive_path.with_suffix(archive_path.suffix + ".part")` while streaming
  and hashing (`hashlib.sha256`), then `os.replace(part, archive_path)`. The
  hash is returned alongside the parsed prices (function return type grows to
  a small dataclass or tuple — see Data flow) so callers can put it in the
  manifest without re-reading the file.

### 3. Price-selection policy (`mtg.py`)

- New `class PriceSelectionPolicy(enum.Enum)`: `MARKET_THEN_MID` (default,
  today's behavior), `MARKET_ONLY`, `MID_ONLY`.
- `_price_from_row(row, policy)` gains the `policy` parameter; `MARKET_ONLY`/
  `MID_ONLY` return `None` if their field is null rather than falling back.
- Grouping key for one day's archive payload becomes `(productId, subTypeName)`
  instead of bare `productId`. Resolution per `productId`:
  - 0 or 1 row → use it directly.
  - 2+ rows, exactly one has `subTypeName in ("Normal", None)` → use that one,
    others discarded (documented as the printing this project tracks; foil
    variants are a different product line, not a duplicate observation of
    the same price).
  - 2+ rows and the "Normal" resolution above is not unique (two Normal rows,
    or no Normal row among several non-Normal ones) → raise `ValueError`
    naming `productId`, `date`, and the `subTypeName`s seen — no silent
    last-one-wins.
- `subTypeName` and which field (`market`/`mid`) was actually used are kept
  per point (see Data flow) so they can feed the manifest's
  `price_field_used` percentage.

### 4. Manifest (`src/tfm3lab/manifest.py`, JSON)

- `write_manifest(payload: dict, path: Path) -> Path`: adds a common core
  (`git_sha`, `as_of`, `live_end`, `written_at_utc`, `package_versions` via
  `importlib.metadata.version(...)` for `requests`/`py7zr`/`pandas`) under a
  `"_meta"` key, merges the caller's `payload`, writes indented JSON.
  `git_sha` resolution: `git rev-parse HEAD` via subprocess, falling back to
  `"unknown"` (never fails the run) if not in a git checkout (e.g. Colab
  clone without `.git`, or `git` unavailable).
- `build_fetch_manifest(resolved_cards, archive_hashes, price_field_counts, coverage_stats, as_of, live_end) -> dict`:
  MTG-specific payload — date range, `{date: sha256}` archive map, resolved
  `CardSpec` rows (label/group_id/product_id/group_abbreviation/product_name),
  `price_field_used` % (market vs mid), and per-card observed/forward-filled
  %.
- `scripts/01_fetch_data.py` calls `write_manifest(build_fetch_manifest(...), config.RESULTS_DIR / "manifests" / f"fetch-{as_of}-{utc_ts}.json")` after
  the MTG fetch. `results/manifests/` is created on demand (mirrors the
  `config.py` `_d.mkdir` pattern for the other results dirs, but scoped local
  to this call so `manifest.py` doesn't import `config`).

### 5. Data-quality table + figure (`figdata.py`, `plots.py`)

- `figdata.data_quality_table(series_list: list[SeriesData]) -> pd.DataFrame`:
  one row per `SeriesData`, columns `series`, `observed_rate`,
  `fallback_rate` (`1 - observed_rate`, restricted to in-range points —
  matches `README.md`'s existing framing), `max_gap_days` (longest run of
  consecutive unobserved days), `glitch_count` (reuses `find_glitches` against
  a `series/index/date/value` frame built from the raw, not truth-reconstructed,
  prices), `price_min`, `price_max`, `log_return_volatility` (std of
  observed-only log returns).
- `plots.plot_data_quality(table: pd.DataFrame, axes=None)`: one panel figure
  (small-multiple bars: observed rate, fallback rate, glitch count — same
  `PALETTE` used by the rest of `plots.py`).
- `scripts/01_fetch_data.py` calls both for the MTG series only (right after
  building `mtg_series`) and writes
  `results/mtg_data_quality.parquet` / `results/figures/mtg_data_quality.png`.
  Not wired into market/CPI: those series are always 100% observed by
  construction (intersection-only, no fill), so the table would carry no
  information for them.

## Data flow (per-day fetch, MTG)

```
fetch_daily_prices(date, group_ids, policy, session)
  -> download archive atomically (.part -> sha256 -> rename)   [or ArchiveNotAvailableError on 404]
  -> parse payload rows, group by (productId, subTypeName)
  -> resolve each productId to one row per policy (raise on ambiguity)
  -> _price_from_row(row, policy)
  -> returns {group_id: {product_id: PricePoint(price, field_used, subtype)}}, archive_sha256

build_card_series(...)
  -> per date: try fetch_daily_prices; on ArchiveNotAvailableError, leave NaN
  -> accumulate: raw prices frame, price_field_used counts, archive hash map
  -> ffill(limit=max_ffill_days), observed mask (unchanged from today)
  -> returns list[SeriesData], plus a small "ingest report" object carrying
     the archive hash map + price_field_used counts + resolved CardSpec rows
     for the caller (scripts/01_fetch_data.py) to hand to build_fetch_manifest
```

`PricePoint` is a tiny `NamedTuple`/dataclass (`price: float`,
`field_used: Literal["market", "mid"]`, `subtype: str | None`) — internal to
`mtg.py`, not part of `SeriesData` (which stays as-is: `name`/`values`/
`dates`/`observed`, per `windows.py`/`backtest.py`'s existing contract — no
change to that dataclass in this piece of work).

## Error handling

- 404 on an archive → `ArchiveNotAvailableError`, caught, day left `NaN`
  (unchanged end-user behavior, now explicit and distinguishable in logs).
- 429/5xx exhausting retries, or any other unexpected HTTP status → raises,
  script aborts. No metric is silently degraded into missingness.
- Ambiguous `(productId, subTypeName)` resolution → `ValueError` with
  productId/date/subtypes-seen, matching `resolve_card_specs`'s existing
  "fail loud with detail" style in the same file.
- `--as-of`/`--allow-live-end` both given or neither given (on a full run) →
  `argparse` error before any network call.
- Manifest git-SHA resolution failure → degrades to `"unknown"`, never blocks
  the run (a manifest missing one field is better than no manifest).

## Testing

All offline, extending the existing `_FakeSession`/`_FakeResponse` pattern in
`tests/test_mtg.py`:

- Retry-then-raise: fake session returns 429/500 repeatedly → asserts the
  retry count and that the final exception propagates (not swallowed into
  missingness).
- 404 → `ArchiveNotAvailableError`, caught by `build_card_series`, day is NaN.
- `(productId, subTypeName)` ambiguity: fixture with two Normal rows for one
  productId → `ValueError`; fixture with one Normal + one Foil → resolves to
  Normal silently (no error) — both cases get a test.
- `PriceSelectionPolicy` variants: `MARKET_ONLY`/`MID_ONLY` behavior on rows
  missing their preferred field.
- Atomic download: fake session, assert `.part` file never left behind after
  a successful fetch, and that a mid-download failure doesn't leave a
  half-written final-named file.
- New `tests/test_manifest.py`: `write_manifest`/`build_fetch_manifest` as
  pure functions against fixture inputs — git SHA resolution mocked/skippable
  when not in a git checkout, package version lookups don't hit network.
- New `tests/test_fetch_data_cli.py` (or extend an existing CLI test if one
  exists): `--as-of`/`--allow-live-end` mutual exclusivity and requiredness,
  via `argparse` parsing only (no real fetch).
- `tests/test_figdata.py`: `data_quality_table` against hand-built
  `SeriesData` fixtures (mixed observed/ffill/glitch cases), asserting exact
  expected rates/counts.

No new test requires network; `TFM3LAB_RUN_LIVE_FETCH_SMOKE=1`-gated live
tests are unaffected (out of scope here — Prompt 1's convention, not touched).

## Migration notes for existing `results/`

- `results/mtg_prices.parquet` committed today was fetched with an implicit
  `end=date.today()` and no manifest — it predates this change and has no
  `as_of` provenance. Not regenerated as part of this work (no live fetch
  here); flagged as a residual risk in the final report.
- No schema change to `results/mtg_prices.parquet` itself (`date`, `value`,
  `observed`, `series` columns unchanged) — the new `subTypeName`/
  `price_field_used` detail lives only in the manifest and the data-quality
  table, not retrofitted into the cached parquet's columns, to avoid a
  breaking schema change to a committed artifact other scripts already read.
