# MTG Benchmark Configurability & Preregistered Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Experiment A (MTG card prices) configurable as a preregistered
evaluation: a declarative context/horizon/ablation grid, a multi-baseline
leaderboard, a documented median+weighted-mean metric aggregation, and a
TimesFM-2.5 zero-shot adapter — all additive, offline-testable, and without
touching `02_exp_mtg.py`'s existing output or any number in
`docs/talk-outline.md`.

**Architecture:** New `src/tfm3lab/benchmark_config.py` (JSON-loaded grid
schema) and `src/tfm3lab/benchmark.py` (shared origin set + ablation combo
enumeration + placebo-panel sampling) sit alongside the existing
`backtest.py`/`summarize.py`, which get small additive extensions
(`make_positive` recorded per row; a multi-baseline leaderboard + weighted
aggregation). A new CLI, `scripts/02b_exp_mtg_benchmark.py`, wires it all
together with `--dry-run` support. `src/tfm3lab/model_2p5.py` adapts the
already-installed `timesfm` distribution's bundled TimesFM-2.5 implementation
to the existing `Forecaster` protocol.

**Tech Stack:** Python 3.12, pandas, numpy, the already-installed `timesfm`
package (bundles both `timesfm3` and legacy `timesfm.TimesFM_2p5_200M_torch`
— no new dependency), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-04-mtg-benchmark-preregistration-design.md`

## Global Constraints

- Usa esclusivamente `uv` — mai `pip` o `uv pip`.
- Non modificare manualmente risultati parquet o numeri nelle slide —
  `results/exp_mtg_*.parquet` e `docs/talk-outline.md` restano intoccati da
  questo branch.
- Non inventare risultati, metriche, fonti o dati — nessuna selezione reale
  di 30+ carte benchmark; solo formato + criteri documentati.
- Test offline per default; nessuna chiamata live, nessun download HF, nessuna
  GPU in questo branch — solo CLI, dry-run, test, documentazione.
- Non introdurre leakage — `windows.py` resta esattamente com'è; ogni cella
  della griglia riusa lo stesso origin set condiviso.
- Ogni metrica deve rispettare `observed=True` — già garantito da
  `summarize.py`'s existing filtering, riusato senza modifiche.
- Preserva la convenzione di `windows.py`: origin è il primo indice predetto
  — `windows.py` non viene toccato da nessun task di questo piano.
- Aggiungi un test unitario per ogni bug corretto.
- Alla fine: report — file cambiati, test eseguiti, rischi residui,
  migrazioni necessarie.

---

## Task 1: `BenchmarkConfig` — declarative grid schema + JSON loader

**Files:**
- Create: `src/tfm3lab/benchmark_config.py`
- Test: `tests/test_benchmark_config.py`

**Interfaces:**
- Produces: `BenchmarkConfig` (frozen dataclass: `config_id: str`,
  `context_lengths: tuple[int, ...]`, `horizons: tuple[int, ...]`,
  `primary_horizons: tuple[int, ...] = ()`, `origin_stride: int = 1`,
  `max_origins: int | None = None`, `cards: str = "showcase"`,
  `transforms: tuple[str, ...] = ("raw", "log1p")`,
  `make_positive: tuple[bool, ...] = (True, False)`,
  `modes: tuple[str, ...] = ("univariate", "multivariate")`,
  `placebo_panel_size: int = 7`, `placebo_seed: int = 42`,
  `season_length: int | None = None`,
  `baselines: tuple[str, ...] = ("naive", "seasonal_naive", "drift", "ets")`,
  `description: str = ""`), `load_benchmark_config(path) -> BenchmarkConfig`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_benchmark_config.py
"""Unit tests for tfm3lab.benchmark_config -- pure dataclass validation and
JSON loading, no network, no other tfm3lab modules involved."""

from __future__ import annotations

import json

import pytest

from tfm3lab.benchmark_config import BenchmarkConfig, load_benchmark_config


def _write(tmp_path, payload, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload(**overrides):
    payload = {
        "config_id": "test_grid",
        "context_lengths": [64, 128],
        "horizons": [1, 7],
    }
    payload.update(overrides)
    return payload


def test_load_benchmark_config_minimal_valid_file(tmp_path):
    path = _write(tmp_path, _valid_payload())
    cfg = load_benchmark_config(path)
    assert cfg.config_id == "test_grid"
    assert cfg.context_lengths == (64, 128)
    assert cfg.horizons == (1, 7)
    # defaults
    assert cfg.transforms == ("raw", "log1p")
    assert cfg.make_positive == (True, False)
    assert cfg.modes == ("univariate", "multivariate")
    assert cfg.origin_stride == 1


def test_load_benchmark_config_full_override(tmp_path):
    path = _write(
        tmp_path,
        _valid_payload(
            primary_horizons=[7],
            origin_stride=2,
            max_origins=50,
            cards="benchmark",
            transforms=["log1p"],
            make_positive=[False],
            modes=["multivariate_placebo"],
            placebo_panel_size=10,
            placebo_seed=7,
            season_length=7,
            baselines=["naive"],
            description="override test",
        ),
    )
    cfg = load_benchmark_config(path)
    assert cfg.primary_horizons == (7,)
    assert cfg.origin_stride == 2
    assert cfg.max_origins == 50
    assert cfg.cards == "benchmark"
    assert cfg.transforms == ("log1p",)
    assert cfg.make_positive == (False,)
    assert cfg.modes == ("multivariate_placebo",)
    assert cfg.placebo_panel_size == 10
    assert cfg.placebo_seed == 7
    assert cfg.season_length == 7
    assert cfg.baselines == ("naive",)
    assert cfg.description == "override test"


@pytest.mark.parametrize("missing_field", ["config_id", "context_lengths", "horizons"])
def test_load_benchmark_config_missing_required_field_raises(tmp_path, missing_field):
    payload = _valid_payload()
    del payload[missing_field]
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=missing_field):
        load_benchmark_config(path)


def test_benchmark_config_rejects_empty_config_id():
    with pytest.raises(ValueError, match="config_id"):
        BenchmarkConfig(config_id="", context_lengths=(64,), horizons=(1,))


def test_benchmark_config_rejects_empty_context_lengths():
    with pytest.raises(ValueError, match="context_lengths"):
        BenchmarkConfig(config_id="t", context_lengths=(), horizons=(1,))


def test_benchmark_config_rejects_empty_horizons():
    with pytest.raises(ValueError, match="horizons"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=())


def test_benchmark_config_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="horizons"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(0,))


def test_benchmark_config_rejects_non_positive_context_length():
    with pytest.raises(ValueError, match="context_lengths"):
        BenchmarkConfig(config_id="t", context_lengths=(-1,), horizons=(1,))


def test_benchmark_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown modes"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), modes=("bogus",))


def test_benchmark_config_rejects_unknown_transform():
    with pytest.raises(ValueError, match="unknown transforms"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), transforms=("bogus",))


def test_benchmark_config_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="unknown baselines"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), baselines=("bogus",))


def test_benchmark_config_rejects_placebo_panel_size_below_one():
    with pytest.raises(ValueError, match="placebo_panel_size"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), placebo_panel_size=0)


def test_benchmark_config_rejects_origin_stride_below_one():
    with pytest.raises(ValueError, match="origin_stride"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), origin_stride=0)


def test_benchmark_config_rejects_primary_horizon_not_in_horizons():
    with pytest.raises(ValueError, match="primary_horizons"):
        BenchmarkConfig(
            config_id="t", context_lengths=(64,), horizons=(1, 7), primary_horizons=(28,)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tfm3lab.benchmark_config'`

- [ ] **Step 3: Write the implementation**

```python
# src/tfm3lab/benchmark_config.py
"""Declarative benchmark configuration: the JSON grid a preregistered MTG
benchmark run is defined by (context lengths, horizons, ablation
dimensions). Loaded once per invocation of
scripts/02b_exp_mtg_benchmark.py -- describes *which* combos to run, never
touches windows.py's origin-index math itself (see tfm3lab.benchmark for
that).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_KNOWN_MODES = frozenset({"univariate", "multivariate", "multivariate_placebo"})
_KNOWN_TRANSFORMS = frozenset({"raw", "log1p"})
_KNOWN_BASELINES = frozenset({"naive", "seasonal_naive", "drift", "ets"})

_REQUIRED_FIELDS = ("config_id", "context_lengths", "horizons")
_TUPLE_FIELDS = (
    "context_lengths",
    "horizons",
    "primary_horizons",
    "transforms",
    "make_positive",
    "modes",
    "baselines",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """One declarative benchmark grid. `config_id` is required and never
    auto-generated -- two configs may share every other field but must be
    named distinctly by the author, so a run's provenance is always
    traceable to an explicit name, not a content hash."""

    config_id: str
    context_lengths: tuple[int, ...]
    horizons: tuple[int, ...]
    primary_horizons: tuple[int, ...] = ()
    origin_stride: int = 1
    max_origins: int | None = None
    cards: str = "showcase"
    transforms: tuple[str, ...] = ("raw", "log1p")
    make_positive: tuple[bool, ...] = (True, False)
    modes: tuple[str, ...] = ("univariate", "multivariate")
    placebo_panel_size: int = 7
    placebo_seed: int = 42
    season_length: int | None = None
    baselines: tuple[str, ...] = ("naive", "seasonal_naive", "drift", "ets")
    description: str = ""

    def __post_init__(self):
        if not self.config_id:
            raise ValueError("config_id is required and must be non-empty")
        if not self.context_lengths:
            raise ValueError("context_lengths must be non-empty")
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if any(c <= 0 for c in self.context_lengths):
            raise ValueError(f"context_lengths must all be positive, got {self.context_lengths}")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must all be positive, got {self.horizons}")
        unknown_modes = set(self.modes) - _KNOWN_MODES
        if unknown_modes:
            raise ValueError(
                f"unknown modes {sorted(unknown_modes)}, expected subset of {sorted(_KNOWN_MODES)}"
            )
        unknown_transforms = set(self.transforms) - _KNOWN_TRANSFORMS
        if unknown_transforms:
            raise ValueError(
                f"unknown transforms {sorted(unknown_transforms)}, "
                f"expected subset of {sorted(_KNOWN_TRANSFORMS)}"
            )
        unknown_baselines = set(self.baselines) - _KNOWN_BASELINES
        if unknown_baselines:
            raise ValueError(
                f"unknown baselines {sorted(unknown_baselines)}, "
                f"expected subset of {sorted(_KNOWN_BASELINES)}"
            )
        if self.placebo_panel_size < 1:
            raise ValueError(f"placebo_panel_size must be >= 1, got {self.placebo_panel_size}")
        if self.origin_stride < 1:
            raise ValueError(f"origin_stride must be >= 1, got {self.origin_stride}")
        unknown_primary = set(self.primary_horizons) - set(self.horizons)
        if unknown_primary:
            raise ValueError(
                f"primary_horizons {sorted(unknown_primary)} not present in horizons {self.horizons}"
            )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Loads and validates a BenchmarkConfig from a JSON file. A missing
    required field (config_id, context_lengths, horizons) raises ValueError
    naming exactly what's missing -- never silently defaulted."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in _REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"{path}: missing required field(s) {missing}")

    kwargs = dict(payload)
    for key in _TUPLE_FIELDS:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])
    return BenchmarkConfig(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_config.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/benchmark_config.py tests/test_benchmark_config.py
git commit -m "feat: add declarative BenchmarkConfig schema + JSON loader"
```

---

## Task 2: `benchmark.py` — shared origin set, ablation combos, placebo panel, dry-run report

**Files:**
- Create: `src/tfm3lab/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `BenchmarkConfig` (Task 1); `windows.valid_origins` (existing,
  unmodified: `valid_origins(n, context_len, horizon, max_origins=None) -> np.ndarray`).
- Produces: `common_origin_set(n, context_lengths, horizons, origin_stride=1, max_origins=None) -> np.ndarray`;
  `AblationCombo` (frozen dataclass: `context_len, horizon, transform, make_positive, mode`);
  `iter_ablation_combos(cfg, card_pool_size) -> list[AblationCombo]`;
  `select_placebo_panel(pool, panel_size, seed) -> tuple`;
  `dry_run_report(cfg, n_days, card_pool_size) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_benchmark.py
"""Unit tests for tfm3lab.benchmark -- pure grid/index math, no forecaster,
no network."""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab.benchmark import (
    AblationCombo,
    common_origin_set,
    dry_run_report,
    iter_ablation_combos,
    select_placebo_panel,
)
from tfm3lab.benchmark_config import BenchmarkConfig
from tfm3lab.windows import valid_origins


def _cfg(**overrides) -> BenchmarkConfig:
    fields = dict(
        config_id="t",
        context_lengths=(4, 8),
        horizons=(1, 2),
        modes=("univariate", "multivariate"),
        transforms=("raw",),
        make_positive=(True,),
    )
    fields.update(overrides)
    return BenchmarkConfig(**fields)


def test_common_origin_set_matches_valid_origins_at_widest_cell():
    got = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2))
    expected = valid_origins(n=30, context_len=8, horizon=2)
    np.testing.assert_array_equal(got, expected)


def test_common_origin_set_thins_by_stride():
    full = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2), origin_stride=1)
    thinned = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2), origin_stride=3)
    np.testing.assert_array_equal(thinned, full[::3])


def test_common_origin_set_is_valid_for_every_smaller_cell():
    origins = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2))
    origins_set = set(origins.tolist())
    for context_len in (4, 8):
        for horizon in (1, 2):
            smaller_valid = set(valid_origins(n=30, context_len=context_len, horizon=horizon).tolist())
            assert origins_set <= smaller_valid


def test_common_origin_set_respects_max_origins():
    got = common_origin_set(n=30, context_lengths=(4,), horizons=(1,), max_origins=3)
    assert len(got) == 3


def test_iter_ablation_combos_full_cartesian_product():
    cfg = _cfg(
        context_lengths=(4, 8),
        horizons=(1, 2),
        transforms=("raw", "log1p"),
        make_positive=(True, False),
        modes=("univariate", "multivariate"),
    )
    combos = iter_ablation_combos(cfg, card_pool_size=7)
    assert len(combos) == 2 * 2 * 2 * 2 * 2
    assert all(isinstance(c, AblationCombo) for c in combos)


def test_iter_ablation_combos_skips_placebo_when_pool_too_small():
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    combos = iter_ablation_combos(cfg, card_pool_size=3)
    assert all(c.mode != "multivariate_placebo" for c in combos)
    assert any(c.mode == "univariate" for c in combos)


def test_iter_ablation_combos_keeps_placebo_when_pool_large_enough():
    cfg = _cfg(modes=("multivariate_placebo",), placebo_panel_size=3)
    combos = iter_ablation_combos(cfg, card_pool_size=5)
    assert len(combos) > 0
    assert all(c.mode == "multivariate_placebo" for c in combos)


def test_select_placebo_panel_deterministic_under_same_seed():
    pool = tuple(f"card_{i}" for i in range(10))
    a = select_placebo_panel(pool, panel_size=4, seed=1)
    b = select_placebo_panel(pool, panel_size=4, seed=1)
    assert a == b
    assert len(a) == 4
    assert set(a) <= set(pool)


def test_select_placebo_panel_different_seeds_can_differ():
    pool = tuple(f"card_{i}" for i in range(20))
    a = select_placebo_panel(pool, panel_size=5, seed=1)
    b = select_placebo_panel(pool, panel_size=5, seed=2)
    assert a != b


def test_select_placebo_panel_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="need >="):
        select_placebo_panel(("a", "b"), panel_size=5, seed=1)


def test_dry_run_report_zero_days_reports_zero_origins_without_raising():
    cfg = _cfg()
    report = dry_run_report(cfg, n_days=0, card_pool_size=7)
    assert report["n_origins"] == 0
    assert report["n_combos"] > 0  # combos are still enumerated, just not runnable yet


def test_dry_run_report_combo_count_and_skip_accounting():
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    report = dry_run_report(cfg, n_days=30, card_pool_size=3)  # pool too small for placebo
    assert report["n_combos_skipped_placebo_pool_too_small"] > 0
    assert all(c["mode"] != "multivariate_placebo" for c in report["combos"])


def test_dry_run_report_estimated_calls_scale_with_card_pool_for_univariate():
    cfg = _cfg(modes=("univariate",), context_lengths=(4,), horizons=(1,))
    small = dry_run_report(cfg, n_days=30, card_pool_size=2)
    large = dry_run_report(cfg, n_days=30, card_pool_size=8)
    assert large["estimated_predict_batch_calls"] == 4 * small["estimated_predict_batch_calls"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tfm3lab.benchmark'`

- [ ] **Step 3: Write the implementation**

```python
# src/tfm3lab/benchmark.py
"""Grid orchestration for the declarative MTG benchmark: one shared origin
set across the whole context/horizon grid, ablation-combo enumeration, and
placebo-panel card sampling. Origin *index* math itself stays entirely in
windows.py (untouched) -- this module only decides which combos to run and
reuses windows.valid_origins for the actual computation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .benchmark_config import BenchmarkConfig
from .windows import valid_origins


def common_origin_set(
    n: int,
    context_lengths: tuple[int, ...],
    horizons: tuple[int, ...],
    origin_stride: int = 1,
    max_origins: int | None = None,
) -> np.ndarray:
    """One origin array valid for the grid's *widest* cell
    (max(context_lengths), max(horizons)) -- every smaller cell's own valid
    range is a superset of this (windows.valid_origins' start/end bounds
    are monotonic in context_len/horizon), so reusing this one array for
    every combo satisfies "all configurations must be compared on the same
    origins" exactly, with no per-combo recomputation.
    """
    origins = valid_origins(
        n=n,
        context_len=max(context_lengths),
        horizon=max(horizons),
        max_origins=max_origins,
    )
    if origin_stride > 1:
        origins = origins[::origin_stride]
    return origins


@dataclass(frozen=True)
class AblationCombo:
    """One cell of the ablation grid -- what run_univariate_backtest /
    run_multivariate_backtest (backtest.py, unmodified) get called with."""

    context_len: int
    horizon: int
    transform: str
    make_positive: bool
    mode: str


def iter_ablation_combos(cfg: BenchmarkConfig, card_pool_size: int) -> list[AblationCombo]:
    """Full cartesian product of the grid's ablation dimensions, in a fixed
    deterministic order (context_lengths x horizons x transforms x
    make_positive x modes, each in the order given in the config).

    `multivariate_placebo` combos are dropped when `card_pool_size` is
    smaller than `cfg.placebo_panel_size` -- the caller (dry_run_report /
    the CLI) is responsible for reporting that skip; this function just
    doesn't emit an unrunnable combo.
    """
    combos = []
    for context_len in cfg.context_lengths:
        for horizon in cfg.horizons:
            for transform in cfg.transforms:
                for make_positive in cfg.make_positive:
                    for mode in cfg.modes:
                        if mode == "multivariate_placebo" and card_pool_size < cfg.placebo_panel_size:
                            continue
                        combos.append(
                            AblationCombo(context_len, horizon, transform, make_positive, mode)
                        )
    return combos


def select_placebo_panel(pool: tuple, panel_size: int, seed: int) -> tuple:
    """Deterministic random sample of `panel_size` items from `pool`
    (typically CardSpec instances), seeded so the same config always
    selects the same placebo panel. Raises if the pool is smaller than
    `panel_size` -- the caller must have already checked this via
    iter_ablation_combos' skip logic; this function fails loudly rather
    than silently sampling with replacement or truncating.
    """
    if len(pool) < panel_size:
        raise ValueError(f"card pool has {len(pool)} cards, need >= {panel_size} for a placebo panel")
    rng = random.Random(seed)
    return tuple(rng.sample(list(pool), panel_size))


def dry_run_report(cfg: BenchmarkConfig, n_days: int, card_pool_size: int) -> dict:
    """Everything --dry-run needs to report, with zero forecaster calls: the
    shared origin set's size (0 if n_days <= 0 -- e.g. no cached data yet),
    every combo that would run, how many were skipped (and why), and a rough
    predict_batch call-count estimate (one call per univariate series-origin
    pair, one call per multivariate origin, per combo).
    """
    if n_days > 0:
        origins = common_origin_set(
            n=n_days,
            context_lengths=cfg.context_lengths,
            horizons=cfg.horizons,
            origin_stride=cfg.origin_stride,
            max_origins=cfg.max_origins,
        )
    else:
        origins = np.array([], dtype=int)

    combos = iter_ablation_combos(cfg, card_pool_size)
    full_count = (
        len(cfg.context_lengths)
        * len(cfg.horizons)
        * len(cfg.transforms)
        * len(cfg.make_positive)
        * len(cfg.modes)
    )
    n_origins = len(origins)
    estimated_calls = sum(
        n_origins * card_pool_size if c.mode == "univariate" else n_origins for c in combos
    )

    return {
        "config_id": cfg.config_id,
        "n_days": n_days,
        "n_origins": n_origins,
        "n_combos": len(combos),
        "n_combos_skipped_placebo_pool_too_small": full_count - len(combos),
        "card_pool_size": card_pool_size,
        "estimated_predict_batch_calls": estimated_calls,
        "combos": [
            {
                "context_len": c.context_len,
                "horizon": c.horizon,
                "transform": c.transform,
                "make_positive": c.make_positive,
                "mode": c.mode,
            }
            for c in combos
        ],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/benchmark.py tests/test_benchmark.py
git commit -m "feat: add shared origin set, ablation combo enumeration, placebo panel sampling"
```

---

## Task 3: Wider card manifest loader (`load_card_manifest`)

**Files:**
- Modify: `src/tfm3lab/data/mtg.py` (add function near `DEFAULT_CARDS`, after line 177)
- Test: `tests/test_mtg.py` (extend)

**Interfaces:**
- Consumes: `CardSpec` (existing, `src/tfm3lab/data/mtg.py`).
- Produces: `load_card_manifest(path: str | Path) -> tuple[CardSpec, ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mtg.py` (add `load_card_manifest` to the existing
import block at the top, and `import json` alongside the existing imports):

```python
def test_load_card_manifest_csv(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text(
        "# comment line describing selection criteria\n"
        "label,group_abbreviation,product_name\n"
        "Foo [ABC],ABC,Foo Card\n",
        encoding="utf-8",
    )
    cards = load_card_manifest(path)
    assert cards == (CardSpec("Foo [ABC]", "ABC", "Foo Card"),)


def test_load_card_manifest_csv_multiple_rows_preserve_order(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text(
        "label,group_abbreviation,product_name\n"
        "First [A],A,First Card\n"
        "Second [B],B,Second Card\n",
        encoding="utf-8",
    )
    cards = load_card_manifest(path)
    assert cards == (
        CardSpec("First [A]", "A", "First Card"),
        CardSpec("Second [B]", "B", "Second Card"),
    )


def test_load_card_manifest_json(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(
        json.dumps(
            {"cards": [{"label": "Foo [ABC]", "group_abbreviation": "ABC", "product_name": "Foo Card"}]}
        ),
        encoding="utf-8",
    )
    cards = load_card_manifest(path)
    assert cards == (CardSpec("Foo [ABC]", "ABC", "Foo Card"),)


def test_load_card_manifest_json_bare_list(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(
        json.dumps([{"label": "Foo [ABC]", "group_abbreviation": "ABC", "product_name": "Foo Card"}]),
        encoding="utf-8",
    )
    cards = load_card_manifest(path)
    assert cards == (CardSpec("Foo [ABC]", "ABC", "Foo Card"),)


def test_load_card_manifest_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "cards.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_card_manifest(path)


def test_load_card_manifest_rejects_empty_manifest(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("label,group_abbreviation,product_name\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cards found"):
        load_card_manifest(path)
```

Also update the import block at the top of `tests/test_mtg.py`:

```python
from tfm3lab.data.mtg import (
    ArchiveNotAvailableError,
    CardSpec,
    PriceSelectionPolicy,
    _download_archive_atomic,
    _price_from_row,
    _resolve_subtype_row,
    _session_with_retries,
    build_card_series,
    fetch_daily_prices,
    load_card_manifest,
    resolve_card_specs,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mtg.py -k load_card_manifest -v`
Expected: FAIL with `ImportError: cannot import name 'load_card_manifest'`

- [ ] **Step 3: Write the implementation**

Insert into `src/tfm3lab/data/mtg.py` immediately after the `DEFAULT_CARDS`
tuple's closing `)` (after line 177):

```python
def load_card_manifest(path: str | Path) -> tuple[CardSpec, ...]:
    """Loads a wider card pool from CSV or JSON (extension-dispatched), same
    shape as DEFAULT_CARDS: label, group_abbreviation, product_name.

    Used for the optional benchmark card pool (as opposed to the fixed
    7-card showcase) -- see configs/benchmark_cards.example.csv for the
    format and the selection criteria a real wider manifest should satisfy.
    A `#`-prefixed line in a CSV is treated as a comment (documenting those
    criteria inline), not a data row.
    """
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["cards"] if isinstance(payload, dict) else payload
    elif path.suffix == ".csv":
        df = pd.read_csv(path, comment="#")
        rows = df.to_dict(orient="records")
    else:
        raise ValueError(f"unsupported card manifest format: {path.suffix} (expected .csv or .json)")

    cards = tuple(
        CardSpec(
            label=str(row["label"]),
            group_abbreviation=str(row["group_abbreviation"]),
            product_name=str(row["product_name"]),
        )
        for row in rows
    )
    if not cards:
        raise ValueError(f"{path}: no cards found in manifest")
    return cards
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mtg.py -v`
Expected: PASS, all tests green (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/data/mtg.py tests/test_mtg.py
git commit -m "feat: add load_card_manifest for the optional wider benchmark card pool"
```

---

## Task 4: Record `make_positive` per backtest row

**Files:**
- Modify: `src/tfm3lab/backtest.py`
- Test: `tests/test_backtest.py` (extend)

**Interfaces:**
- Consumes: nothing new — `make_positive` is already a parameter of
  `run_univariate_backtest`/`run_multivariate_backtest` (computed, passed to
  `forecast_batch`, but never recorded in the output row).
- Produces: every row of `run_univariate_backtest`/`run_multivariate_backtest`'s
  output DataFrame now has a `make_positive: bool` column.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
def test_make_positive_recorded_true_in_univariate_backtest():
    a = _make_series("a", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=4, max_horizon=2, make_positive=True
    )
    assert df["make_positive"].eq(True).all()


def test_make_positive_recorded_false_in_univariate_backtest():
    a = _make_series("a", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=4, max_horizon=2, make_positive=False
    )
    assert df["make_positive"].eq(False).all()


def test_make_positive_recorded_in_multivariate_backtest():
    a = _make_series("a", np.arange(20.0))
    b = _make_series("b", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_multivariate_backtest(
        FakeForecaster(), [a, b], origins, context_len=4, max_horizon=2, make_positive=False
    )
    assert df["make_positive"].eq(False).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py -k make_positive -v`
Expected: FAIL with `KeyError: 'make_positive'`

- [ ] **Step 3: Write the implementation**

In `src/tfm3lab/backtest.py`, modify `_rows_for_one_series_forecast`'s
signature (currently ends `mode_label: str, transform: ValueTransform = IDENTITY_TRANSFORM,`)
to add a `make_positive: bool = True` parameter, and record it in the row
dict:

```python
def _rows_for_one_series_forecast(
    s: SeriesData,
    origin: int,
    point: np.ndarray,
    quantiles: np.ndarray,
    context_len: int,
    max_horizon: int,
    season_length: int | None,
    mode_label: str,
    transform: ValueTransform = IDENTITY_TRANSFORM,
    make_positive: bool = True,
) -> list[dict]:
```

Inside the function, in the `row = {...}` dict literal, add one entry (right
after `"transform": transform.name,` — insert `"make_positive": make_positive,`
as the next key):

```python
        row = {
            "mode": mode_label,
            "transform": transform.name,
            "make_positive": make_positive,
            "series": s.name,
            ...
```

Then update both call sites to pass it through. In `run_univariate_backtest`,
inside the `for i, ts_id in enumerate(batch.ts_ids):` loop, the call becomes:

```python
        rows.extend(
            _rows_for_one_series_forecast(
                s,
                origin,
                batch.forecast[i],
                batch.quantiles[i],
                context_len,
                max_horizon,
                season_length,
                mode_label,
                transform,
                make_positive,
            )
        )
```

In `run_multivariate_backtest`, inside the nested `for j, s in enumerate(series_list):`
loop, the call becomes:

```python
            rows.extend(
                _rows_for_one_series_forecast(
                    s,
                    origin,
                    batch.forecast[i, j],
                    batch.quantiles[i, j],
                    context_len,
                    max_horizon,
                    season_length,
                    mode_label,
                    transform,
                    make_positive,
                )
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS, all tests green (existing + new) — this is an additive
column, so every pre-existing assertion on specific column sets must still
pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/backtest.py tests/test_backtest.py
git commit -m "feat: record make_positive per row in backtest output"
```

---

## Task 5: Multi-baseline leaderboard + documented aggregation

**Files:**
- Modify: `src/tfm3lab/summarize.py`
- Test: `tests/test_summarize.py` (extend)

**Interfaces:**
- Consumes: `summarize_accuracy` (existing, unmodified signature).
- Produces: `summarize_accuracy` output gains a `skill_vs_baseline` column
  (additive); `DEFAULT_BASELINE_COLUMNS: tuple[str, ...]`;
  `summarize_leaderboard(df, mase_scales, group_cols=(...), baseline_cols=DEFAULT_BASELINE_COLUMNS) -> pd.DataFrame`;
  `aggregate_leaderboard(leaderboard_df, weight_col="n") -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `tests/test_summarize.py`:

```python
from tfm3lab.summarize import (
    MIN_OBSERVATIONS_FOR_DM_TEST,
    QUANTILE_COLUMNS,
    aggregate_leaderboard,
    compute_mase_scales,
    summarize_accuracy,
    summarize_calibration,
    summarize_leaderboard,
)
```

Append to `tests/test_summarize.py`:

```python
def _row_with_extra_baseline(baseline_drift=None, **kwargs) -> dict:
    row = _row(**kwargs)
    if baseline_drift is not None:
        row["baseline_drift"] = baseline_drift
    return row


def test_summarize_accuracy_includes_skill_column():
    rows = [_row(actual=10.0, forecast=9.0, baseline_naive=8.0)]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    row = summary.iloc[0]
    assert row["skill_vs_baseline"] == pytest.approx(1.0 - row["relative_mae_vs_baseline"])


def test_summarize_leaderboard_multiple_baseline_methods():
    rows = [
        _row_with_extra_baseline(actual=10.0, forecast=9.0, baseline_naive=8.0, baseline_drift=11.0),
        _row_with_extra_baseline(actual=12.0, forecast=11.0, baseline_naive=10.0, baseline_drift=13.0),
    ]
    df = pd.DataFrame(rows)
    leaderboard = summarize_leaderboard(
        df, mase_scales={"a": 1.0}, baseline_cols=("baseline_naive", "baseline_drift")
    )
    assert set(leaderboard["baseline_method"]) == {"naive", "drift"}
    assert "mae_baseline" in leaderboard.columns
    assert "mase_baseline" in leaderboard.columns
    assert len(leaderboard) == 2  # one group x 2 methods


def test_summarize_leaderboard_filters_nan_rows_per_baseline_independently():
    rows = [
        _row_with_extra_baseline(actual=10.0, forecast=9.0, baseline_naive=8.0, baseline_drift=None),
        _row_with_extra_baseline(actual=12.0, forecast=11.0, baseline_naive=10.0, baseline_drift=13.0),
    ]
    df = pd.DataFrame(rows)
    leaderboard = summarize_leaderboard(
        df, mase_scales={"a": 1.0}, baseline_cols=("baseline_naive", "baseline_drift")
    )
    naive_row = leaderboard[leaderboard["baseline_method"] == "naive"].iloc[0]
    drift_row = leaderboard[leaderboard["baseline_method"] == "drift"].iloc[0]
    assert naive_row["n"] == 2
    assert drift_row["n"] == 1  # the None-baseline_drift row excluded for drift only


def test_summarize_leaderboard_rejects_when_no_baseline_columns_present():
    df = pd.DataFrame([_row()])
    with pytest.raises(ValueError, match="none of"):
        summarize_leaderboard(df, mase_scales={"a": 1.0}, baseline_cols=("baseline_ets",))


def test_aggregate_leaderboard_weighted_mean_documents_the_weighting():
    df = pd.DataFrame(
        [
            {
                "mode": "m",
                "baseline_method": "naive",
                "horizon_step": 1,
                "series": "a",
                "n": 10,
                "relative_mae_vs_baseline": 0.5,
            },
            {
                "mode": "m",
                "baseline_method": "naive",
                "horizon_step": 1,
                "series": "b",
                "n": 1000,
                "relative_mae_vs_baseline": 1.5,
            },
        ]
    )
    agg = aggregate_leaderboard(df)
    row = agg.iloc[0]
    naive_mean = (0.5 + 1.5) / 2  # what a naive unweighted mean-of-ratios would give
    expected_weighted = (0.5 * 10 + 1.5 * 1000) / 1010
    assert row["relative_mae_mean_weighted"] == pytest.approx(expected_weighted)
    assert row["relative_mae_mean_weighted"] != pytest.approx(naive_mean)
    assert row["relative_mae_median"] == pytest.approx(1.0)
    assert row["skill_mean_weighted"] == pytest.approx(1.0 - expected_weighted)
    assert row["skill_median"] == pytest.approx(0.0)
    assert row["n_cards"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_summarize.py -k "skill or leaderboard" -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_leaderboard'`

- [ ] **Step 3: Write the implementation**

In `src/tfm3lab/summarize.py`, add near the top (after `QUANTILE_COLUMNS`):

```python
DEFAULT_BASELINE_COLUMNS = ("baseline_naive", "baseline_seasonal_naive", "baseline_drift", "baseline_ets")
```

In `summarize_accuracy`, immediately after the existing
`row.update({...})` block that sets `"relative_mae_vs_baseline": relative_mae_val,`
(before the Diebold-Mariano block that follows it), add:

```python
        row["skill_vs_baseline"] = 1.0 - relative_mae_val
```

At the end of the file, after `summarize_calibration`, add:

```python
def summarize_leaderboard(
    df: pd.DataFrame,
    mase_scales: dict[str, float],
    group_cols: tuple[str, ...] = ("mode", "series", "horizon_step"),
    baseline_cols: tuple[str, ...] = DEFAULT_BASELINE_COLUMNS,
) -> pd.DataFrame:
    """Runs summarize_accuracy once per baseline column present in `df`,
    stacking the results with a `baseline_method` column -- the
    "leaderboard per method" this project's plan requires instead of a
    naive-only headline.

    A baseline column entirely absent from `df` (e.g. no `baseline_ets`
    because season_length was never set) is skipped. A column that's
    present but NaN on some rows has those specific rows filtered out
    before computing THAT baseline's metrics -- its `n` may differ from
    another baseline's `n` in the same group; that's expected (see
    docstring on backtest._baseline_forecasts), not a bug.
    """
    present = [c for c in baseline_cols if c in df.columns]
    if not present:
        raise ValueError(f"none of {baseline_cols} present in df.columns")

    tables = []
    for col in present:
        method = col.removeprefix("baseline_")
        rows_for_col = df[df[col].notna()] if df[col].isna().any() else df
        summary = summarize_accuracy(rows_for_col, mase_scales, baseline_col=col, group_cols=group_cols)
        summary = summary.rename(columns={f"mae_{col}": "mae_baseline", f"mase_{col}": "mase_baseline"})
        summary.insert(0, "baseline_method", method)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def aggregate_leaderboard(leaderboard_df: pd.DataFrame, weight_col: str = "n") -> pd.DataFrame:
    """Collapses the `series` dimension out of a summarize_leaderboard table
    into two documented cross-card statistics per remaining group:

    - `relative_mae_median` / `skill_median`: the median across cards --
      robust to one outlier card dominating the picture.
    - `relative_mae_mean_weighted` / `skill_mean_weighted`: the mean across
      cards, weighted by each card's `n` (observed row count) -- a
      pooled/micro-average: a card with more observed (origin, horizon)
      rows contributes proportionally more than a thinly-observed one.
      This is deliberately NOT a naive unweighted mean of per-card ratios
      (which would let a 5-observation card move the aggregate as much as
      a 5000-observation one).
    """
    per_series_metric_cols = {
        "mae_model",
        "rmse_model",
        "smape_model",
        "mase_model",
        "mae_baseline",
        "mase_baseline",
        "dm_stat",
        "dm_pvalue",
    }
    group_cols = [
        c
        for c in leaderboard_df.columns
        if c not in {"series", weight_col, "relative_mae_vs_baseline", "skill_vs_baseline", *per_series_metric_cols}
    ]

    rows = []
    for keys, group in leaderboard_df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        weights = group[weight_col].to_numpy(dtype=float)
        ratios = group["relative_mae_vs_baseline"].to_numpy(dtype=float)
        row = dict(zip(group_cols, keys, strict=True))
        row["n_cards"] = len(group)
        row["relative_mae_median"] = float(np.median(ratios))
        row["relative_mae_mean_weighted"] = float(np.average(ratios, weights=weights))
        row["skill_median"] = 1.0 - row["relative_mae_median"]
        row["skill_mean_weighted"] = 1.0 - row["relative_mae_mean_weighted"]
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: PASS, all tests green (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/summarize.py tests/test_summarize.py
git commit -m "feat: add multi-baseline leaderboard and documented weighted aggregation"
```

---

## Task 6: TimesFM-2.5 adapter (`model_2p5.py`)

**Files:**
- Create: `src/tfm3lab/model_2p5.py`
- Test: `tests/test_model_2p5.py`

**Interfaces:**
- Consumes: `model.forecast_batch` (existing, unmodified — the adapter is
  designed to be a drop-in `Forecaster` for it).
- Produces: `TimesFM2p5Adapter` (class implementing `Forecaster.predict_batch`);
  `load_forecaster_2p5(repo_id=DEFAULT_REPO_ID, **kwargs) -> TimesFM2p5Adapter`;
  `DEFAULT_REPO_ID = "google/timesfm-2.5-200m-pytorch"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_2p5.py
"""Unit tests for tfm3lab.model_2p5's TimesFM2p5Adapter using a fake
underlying 2.5 model -- no real checkpoint, no network, no torch download.
The real checkpoint is never loaded by this branch (no GPU run performed) --
see load_forecaster_2p5's docstring for the live path, exercised only by a
future opt-in smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab import config
from tfm3lab.model import forecast_batch
from tfm3lab.model_2p5 import TimesFM2p5Adapter


class _FakeTimesFM25:
    """Mimics TimesFM_2p5_200M_torch.forecast(horizon, inputs) -> (point, quantiles):
    repeats each input's last value; quantiles = point +/- fixed offsets,
    `n_levels` configurable so tests can prove the adapter itself doesn't
    hardcode TimesFM-3's 9-level grid (config.N_QUANTILES).
    """

    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels

    def forecast(self, horizon, inputs):
        points, quants = [], []
        levels = np.linspace(-0.2, 0.2, self.n_levels)
        for ctx in inputs:
            ctx = np.asarray(ctx, dtype=float)
            point = np.full(horizon, ctx[-1])
            quant = point[:, None] + levels
            points.append(point)
            quants.append(quant)
        return np.stack(points, axis=0), np.stack(quants, axis=0)


def test_predict_batch_shapes_and_ts_id_order():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=5))
    contexts = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])]
    outputs = adapter.predict_batch(contexts, horizon=4, ts_ids=["a", "b"])
    assert [o.ts_id for o in outputs] == ["a", "b"]
    assert outputs[0].forecast.shape == (4,)
    np.testing.assert_allclose(outputs[0].forecast, 3.0)
    np.testing.assert_allclose(outputs[1].forecast, 20.0)
    assert outputs[1].quantiles.shape == (4, 5)


def test_predict_batch_generates_ts_ids_when_not_given():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    outputs = adapter.predict_batch([np.array([1.0]), np.array([2.0])], horizon=1)
    assert [o.ts_id for o in outputs] == ["0", "1"]


def test_predict_batch_rejects_covariates():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    with pytest.raises(NotImplementedError, match="covariates"):
        adapter.predict_batch(
            [np.array([1.0])], horizon=1, past_only_covariates=[np.array([0.0])]
        )


def test_adapter_with_matching_quantile_grid_works_through_model_forecast_batch():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=config.N_QUANTILES))
    result = forecast_batch(adapter, [np.array([1.0, 2.0, 5.0])], max_horizon=3, ts_ids=["only"])
    assert result.ts_ids == ["only"]
    assert result.forecast.shape == (1, 3)


def test_adapter_with_mismatched_quantile_grid_raises_loudly_through_model_forecast_batch():
    # Documents the open/unverified risk from model_2p5.py's module
    # docstring: if TimesFM-2.5's real quantile grid differs from
    # TimesFM-3's, routing it through the shared model.forecast_batch fails
    # loudly at assert_quantile_shape rather than silently producing a
    # misaligned results table.
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=5))
    with pytest.raises(AssertionError, match="expected 9 quantiles"):
        forecast_batch(adapter, [np.array([1.0])], max_horizon=2, ts_ids=["only"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_2p5.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tfm3lab.model_2p5'`

- [ ] **Step 3: Write the implementation**

```python
# src/tfm3lab/model_2p5.py
"""Adapter making TimesFM-2.5 satisfy the same `Forecaster` protocol
model.py's `forecast_batch` already consumes -- so it works as a drop-in
zero-shot historical baseline alongside TimesFM-3, without backtest.py
needing to know which model it's talking to.

Confirmed (see docs/superpowers/specs/2026-09-04-mtg-benchmark-preregistration-design.md):
no new/conflicting dependency -- the already-pinned `timesfm>=3.0.0`
distribution bundles TimesFM-2.5's own legacy-API implementation
(`timesfm.TimesFM_2p5_200M_torch`). Its default checkpoint
(google/timesfm-2.5-200m-pytorch) is Apache-2.0 and ungated, unlike the v3
checkpoint -- no HF login required even to load it for real.

Open, UNVERIFIED risks (no live inference run performed by this branch):
  - the exact quantile grid TimesFM-2.5 returns has not been checked
    against a real checkpoint load -- this adapter does NOT assume it
    matches config.N_QUANTILES/config.QUANTILE_LEVELS; if it differs,
    routing this adapter through model.forecast_batch raises loudly at
    assert_quantile_shape (see tests/test_model_2p5.py), it does not
    silently misalign the results table.
  - the assumed (n_series, horizon[, n_quantiles]) batch-array shape
    TimesFM_2p5_200M_torch.forecast() returns is inferred from its type
    signature (`-> tuple[np.ndarray, np.ndarray]`), not verified against a
    real call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_REPO_ID = "google/timesfm-2.5-200m-pytorch"


@dataclasses.dataclass
class _Output:
    ts_id: str
    forecast: np.ndarray
    quantiles: np.ndarray


class TimesFM2p5Adapter:
    """Wraps a loaded TimesFM_2p5_200M_torch instance to satisfy
    model.Forecaster's predict_batch shape.

    `use_symmetric_averaging`/`use_znorm`/`sort_quantiles`/`padding_mode`
    are accepted for protocol compatibility but have no equivalent in
    TimesFM_2p5_200M_torch.forecast() and are silently unused -- they are
    implementation/performance details, not semantic content, so dropping
    them cannot silently change a forecast's meaning the way dropping
    covariates would. `past_only_covariates`/`past_future_covariates` DO
    change a forecast's meaning if silently dropped, so a non-None value
    for either raises instead.
    """

    def __init__(self, model: Any):
        self._model = model

    def predict_batch(
        self,
        contexts: Sequence[np.ndarray],
        horizon: int,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids: list[str] | None = None,
        return_quantiles: bool = True,
        use_symmetric_averaging: bool = True,
        make_positive: bool = True,
        sort_quantiles: bool = True,
        use_znorm: bool = False,
        padding_mode: str = "none",
    ) -> list[_Output]:
        if past_only_covariates is not None or past_future_covariates is not None:
            raise NotImplementedError(
                "TimesFM2p5Adapter does not support covariates -- the underlying "
                "TimesFM_2p5_200M_torch.forecast() call has no covariate parameters"
            )
        ts_ids = list(ts_ids) if ts_ids is not None else [str(i) for i in range(len(contexts))]
        point, quantiles = self._model.forecast(
            horizon=horizon, inputs=[np.asarray(c, dtype=float) for c in contexts]
        )
        return [
            _Output(ts_id=ts_id, forecast=np.asarray(point[i]), quantiles=np.asarray(quantiles[i]))
            for i, ts_id in enumerate(ts_ids)
        ]


def load_forecaster_2p5(repo_id: str = DEFAULT_REPO_ID, **kwargs: Any) -> TimesFM2p5Adapter:
    """Loads the real TimesFM-2.5 checkpoint via the bundled legacy API.
    `timesfm` is imported inside this function, not at module load, so
    importing tfm3lab.model_2p5 (and running its unit tests against a fake
    model) never requires torch -- mirrors model.py's load_forecaster.
    Not called by this branch (no GPU/live-checkpoint run performed).
    """
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch(**kwargs)
    model.load_checkpoint(repo_id)
    return TimesFM2p5Adapter(model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_2p5.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/model_2p5.py tests/test_model_2p5.py
git commit -m "feat: add TimesFM-2.5 zero-shot adapter (bundled dependency, no new package)"
```

---

## Task 7: Example benchmark config + card manifest files

**Files:**
- Create: `configs/benchmark_preregistered.example.json`
- Create: `configs/benchmark_cards.example.csv`
- Test: `tests/test_example_configs.py`

**Interfaces:**
- Consumes: `load_benchmark_config` (Task 1), `load_card_manifest` (Task 3).
- Produces: two committed example files a real run can point `--config`/
  `--cards` at, regression-tested against the loaders that must accept them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_example_configs.py
"""Regression guard: the shipped example config/card-manifest files must
stay loadable by the schemas they document. No network."""

from __future__ import annotations

from pathlib import Path

from tfm3lab.benchmark_config import load_benchmark_config
from tfm3lab.data.mtg import load_card_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_benchmark_config_loads_and_matches_the_analysis_plan():
    cfg = load_benchmark_config(REPO_ROOT / "configs" / "benchmark_preregistered.example.json")
    assert cfg.context_lengths == (64, 128, 256, 512)
    assert cfg.horizons == (1, 7, 28, 56, 64)
    assert cfg.primary_horizons == (28, 56)


def test_example_card_manifest_loads():
    cards = load_card_manifest(REPO_ROOT / "configs" / "benchmark_cards.example.csv")
    assert len(cards) == 7
    labels = {c.label for c in cards}
    assert "Ragavan [MH2]" in labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_example_configs.py -v`
Expected: FAIL — `FileNotFoundError` (neither example file exists yet)

- [ ] **Step 3: Write the implementation**

Create `configs/benchmark_preregistered.example.json`:

```json
{
  "config_id": "preregistered_v1_example",
  "description": "Worked example of the full preregistered grid from docs/analysis-plan.md. No results/exp_mtg_benchmark_*.parquet exist for this config_id until someone runs scripts/02b_exp_mtg_benchmark.py against real cached data on a GPU (see README's hybrid-execution section) -- not done as part of this branch.",
  "context_lengths": [64, 128, 256, 512],
  "horizons": [1, 7, 28, 56, 64],
  "primary_horizons": [28, 56],
  "origin_stride": 1,
  "max_origins": null,
  "cards": "showcase",
  "transforms": ["raw", "log1p"],
  "make_positive": [true, false],
  "modes": ["univariate", "multivariate"],
  "placebo_panel_size": 7,
  "placebo_seed": 42,
  "season_length": 7,
  "baselines": ["naive", "seasonal_naive", "drift", "ets"]
}
```

Create `configs/benchmark_cards.example.csv`:

```
# Wider MTG benchmark card manifest -- FORMAT + a 7-row WORKED EXAMPLE.
#
# This is NOT a real 30+-card benchmark selection. Picking one requires a
# live TCGCSV catalog crawl (network, out of scope for this branch) to check
# each candidate against the criteria below -- populating a real list is
# future, separately-reviewed work, not invented here.
#
# A real wider manifest's cards should each satisfy:
#   - continuous TCGCSV daily coverage since TCGCSV_ARCHIVE_START
#     (2024-02-08) -- no card that only appears mid-history
#   - an observed_rate (post-ingestion -- see figdata.data_quality_table)
#     above some floor (e.g. >= 0.90) -- thin/delisted cards make weak
#     evaluation targets
#   - no two rows resolving to the same underlying printing (distinct
#     product_name/group_abbreviation pairs)
#   - a mix of price regimes (chase mythic, mid-tier playable, reprint
#     staple, budget common), not all high-volatility chase cards -- see
#     DEFAULT_CARDS in src/tfm3lab/data/mtg.py for how the 7-card showcase
#     already documents this mix
#
# Columns: label,group_abbreviation,product_name (same shape as
# tfm3lab.data.mtg.CardSpec -- resolved the same way via resolve_card_specs).
label,group_abbreviation,product_name
Ragavan [MH2],MH2,"Ragavan, Nimble Pilferer"
Urza's Saga [MH2],MH2,Urza's Saga
Sheoldred [DMU],DMU,"Sheoldred, the Apocalypse"
The One Ring [LTR],LTR,The One Ring
Orcish Bowmasters [LTR],LTR,Orcish Bowmasters
Chatterfang [MH2],MH2,"Chatterfang, Squirrel General"
Mishra's Factory [MH2],MH2,Mishra's Factory
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_example_configs.py -v`
Expected: PASS, both tests green

- [ ] **Step 5: Commit**

```bash
git add configs/benchmark_preregistered.example.json configs/benchmark_cards.example.csv tests/test_example_configs.py
git commit -m "feat: add worked-example benchmark config + card manifest (format only, no invented 30+ selection)"
```

---

## Task 8: CLI — `scripts/02b_exp_mtg_benchmark.py`

**Files:**
- Create: `scripts/02b_exp_mtg_benchmark.py`
- Test: `tests/test_exp_mtg_benchmark_cli.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`/`load_benchmark_config` (Task 1),
  `common_origin_set`/`iter_ablation_combos`/`select_placebo_panel`/`dry_run_report`
  (Task 2), `load_card_manifest`/`DEFAULT_CARDS`/`CardSpec` (Task 3, existing),
  `run_univariate_backtest`/`run_multivariate_backtest`/`SeriesData`/
  `IDENTITY_TRANSFORM`/`LOG1P_TRANSFORM` (existing, `backtest.py`),
  `summarize_leaderboard`/`aggregate_leaderboard`/`compute_mase_scales`
  (Task 5, existing), `TimesFM2p5Adapter`/`load_forecaster_2p5` (Task 6),
  `manifest.write_manifest` (existing, unmodified).
- Produces: `build_parser() -> argparse.ArgumentParser`;
  `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_exp_mtg_benchmark_cli.py
"""CLI tests for scripts/02b_exp_mtg_benchmark.py -- --dry-run and argument
validation only, no forecaster load, no network. Mirrors
tests/test_fetch_data_cli.py's importlib-loading pattern (scripts/ are thin
CLIs, not part of the tfm3lab package).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02b_exp_mtg_benchmark.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp_mtg_benchmark_02b", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench02b():
    return _load_script_module()


def _write_config(tmp_path, **overrides):
    payload = {
        "config_id": "test_grid",
        "context_lengths": [4, 8],
        "horizons": [1, 2],
        "modes": ["univariate", "multivariate"],
        "transforms": ["raw"],
        "make_positive": [True],
        "cards": "showcase",
    }
    payload.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_is_required(bench02b):
    parser = bench02b.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run"])


def test_dry_run_reports_expected_combo_count(bench02b, tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path)
    # No data/cache/mtg_prices.parquet in this tmp environment -> dry-run
    # must still succeed, with n_days=0 (a NOTE printed to stderr, not a crash).
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    bench02b.main(["--config", str(cfg_path), "--dry-run"])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report["config_id"] == "test_grid"
    # 2 context_lengths x 2 horizons x 1 transform x 1 make_positive x 2 modes
    assert report["n_combos"] == 8
    assert "NOTE" in out.err


def test_dry_run_out_writes_file_instead_of_stdout(bench02b, tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    out_path = tmp_path / "report.json"
    bench02b.main(["--config", str(cfg_path), "--dry-run", "--dry-run-out", str(out_path)])
    report = json.loads(out_path.read_text())
    assert report["config_id"] == "test_grid"


def test_cards_benchmark_without_path_exits(bench02b, tmp_path):
    cfg_path = _write_config(tmp_path)
    with pytest.raises(SystemExit, match="requires a manifest path"):
        bench02b.main(["--config", str(cfg_path), "--dry-run", "--cards", "benchmark"])


def test_cards_showcase_dry_run_uses_seven_card_pool(bench02b, tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    bench02b.main(["--config", str(cfg_path), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["card_pool_size"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exp_mtg_benchmark_cli.py -v`
Expected: FAIL — script file doesn't exist yet (`FileNotFoundError` /
`spec is None` from `spec_from_file_location`)

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
# scripts/02b_exp_mtg_benchmark.py
"""Experiment A, configurable: declarative context/horizon/ablation grid
over the MTG card pool, multi-baseline leaderboard, TimesFM-3 or TimesFM-2.5.

This is Experiment A's PREREGISTERED evaluation entry point -- distinct from
scripts/02_exp_mtg.py, whose fixed single-config output
(results/exp_mtg_*.parquet) backs the numbers already committed in
docs/talk-outline.md. This script's own outputs are prefixed
exp_mtg_benchmark_* and are never read by 02_exp_mtg.py, 04-07, or the
existing slides -- running this script does not change any existing number.

Usage:
    uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --dry-run
    uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --as-of 2026-09-01

Requires (non-dry-run only):
  - scripts/01_fetch_data.py already run for the cards this config needs
  - a loaded forecaster: TimesFM-3 (default, gated HF checkpoint) or
    TimesFM-2.5 (--adapter timesfm2.5, Apache-2.0, ungated)

Writes to results/ (non-dry-run only):
    exp_mtg_benchmark_raw_predictions.parquet
    exp_mtg_benchmark_leaderboard.parquet
    exp_mtg_benchmark_leaderboard_aggregate.parquet
    manifests/benchmark-<config_id>-<run_id>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import sys
from pathlib import Path

import pandas as pd

from tfm3lab import config, manifest
from tfm3lab.backtest import (
    IDENTITY_TRANSFORM,
    LOG1P_TRANSFORM,
    SeriesData,
    run_multivariate_backtest,
    run_univariate_backtest,
)
from tfm3lab.benchmark import common_origin_set, dry_run_report, iter_ablation_combos, select_placebo_panel
from tfm3lab.benchmark_config import BenchmarkConfig, load_benchmark_config
from tfm3lab.data.mtg import DEFAULT_CARDS, CardSpec, load_card_manifest
from tfm3lab.summarize import aggregate_leaderboard, compute_mase_scales, summarize_leaderboard

_TRANSFORMS = {"raw": IDENTITY_TRANSFORM, "log1p": LOG1P_TRANSFORM}


def _resolve_card_pool(cfg: BenchmarkConfig, cards_override: str | None) -> tuple[CardSpec, ...]:
    spec = cards_override or cfg.cards
    if spec == "showcase":
        return DEFAULT_CARDS
    if spec == "benchmark":
        raise SystemExit(
            "--cards benchmark requires a manifest path -- there is no built-in "
            "wider card list (see configs/benchmark_cards.example.csv's header "
            "for why, and the criteria a real one must satisfy)"
        )
    return load_card_manifest(spec)


def load_cached_series(cards: tuple[CardSpec, ...]) -> list[SeriesData]:
    path = config.CACHE_DIR / "mtg_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/01_fetch_data.py first")
    df = pd.read_parquet(path)
    wanted = {c.label for c in cards}
    df = df[df["series"].isin(wanted)]
    missing = wanted - set(df["series"].unique())
    if missing:
        raise ValueError(
            f"{sorted(missing)} not found in cached series at {path} -- "
            "re-run scripts/01_fetch_data.py with these cards included"
        )
    series_list = [
        SeriesData(
            name=name,
            values=group.sort_values("date")["value"].to_numpy(dtype=float),
            dates=group.sort_values("date")["date"].to_numpy(),
            observed=group.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, group in df.groupby("series")
    ]
    lengths = {s.name: len(s.values) for s in series_list}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"  trimming all series to the shortest common length ({n} days): {lengths}")
    return [SeriesData(s.name, s.values[-n:], s.dates[-n:], s.observed[-n:]) for s in series_list]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="path to a BenchmarkConfig JSON file")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the grid, no forecaster calls, no results/ writes"
    )
    parser.add_argument(
        "--dry-run-out", type=Path, default=None, help="write the dry-run report JSON here instead of stdout"
    )
    parser.add_argument(
        "--cards", default=None, help="override the config's 'cards' field: showcase | a manifest path"
    )
    parser.add_argument("--adapter", choices=("timesfm3", "timesfm2.5"), default="timesfm3")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_benchmark_config(args.config)
    cards = _resolve_card_pool(cfg, args.cards)

    if args.dry_run:
        cache_path = config.CACHE_DIR / "mtg_prices.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            wanted = {c.label for c in cards}
            present = df[df["series"].isin(wanted)]
            n_days = int(present.groupby("series").size().min()) if len(present) else 0
        else:
            n_days = 0
            print(
                f"NOTE: {cache_path} not found -- dry-run report uses n_days=0 "
                "(run scripts/01_fetch_data.py first for a realistic origin count)",
                file=sys.stderr,
            )
        report = dry_run_report(cfg, n_days=n_days, card_pool_size=len(cards))
        text = json.dumps(report, indent=2)
        if args.dry_run_out:
            args.dry_run_out.write_text(text, encoding="utf-8")
            print(f"wrote dry-run report to {args.dry_run_out}")
        else:
            print(text)
        return

    series_list = load_cached_series(cards)
    n = len(series_list[0].values)
    origins = common_origin_set(
        n=n,
        context_lengths=cfg.context_lengths,
        horizons=cfg.horizons,
        origin_stride=cfg.origin_stride,
        max_origins=cfg.max_origins,
    )
    if len(origins) == 0:
        raise RuntimeError(
            f"no valid origins: {n} days too short for context_lengths={cfg.context_lengths} "
            f"+ horizons={cfg.horizons}"
        )

    if args.adapter == "timesfm3":
        from tfm3lab.model import load_forecaster

        forecaster = load_forecaster()
        use_symmetric_averaging = True
    else:
        from tfm3lab.model_2p5 import load_forecaster_2p5

        forecaster = load_forecaster_2p5()
        use_symmetric_averaging = False  # not supported by TimesFM2p5Adapter

    combos = iter_ablation_combos(cfg, card_pool_size=len(cards))
    placebo_series: list[SeriesData] = []
    if any(c.mode == "multivariate_placebo" for c in combos):
        placebo_panel = select_placebo_panel(cards, cfg.placebo_panel_size, cfg.placebo_seed)
        placebo_labels = {c.label for c in placebo_panel}
        placebo_series = [s for s in series_list if s.name in placebo_labels]

    run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"

    all_results = []
    for combo in combos:
        transform = _TRANSFORMS[combo.transform]
        combo_series = placebo_series if combo.mode == "multivariate_placebo" else series_list
        run_fn = run_univariate_backtest if combo.mode == "univariate" else run_multivariate_backtest
        adapter_label = "timesfm3" if args.adapter == "timesfm3" else "timesfm2p5"
        mode_label = f"{adapter_label}_{combo.mode}"
        df = run_fn(
            forecaster,
            combo_series,
            origins,
            combo.context_len,
            combo.horizon,
            season_length=cfg.season_length,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=combo.make_positive,
            mode_label=mode_label,
            transform=transform,
        )
        df = df.assign(
            run_id=run_id,
            config_id=cfg.config_id,
            context_len=combo.context_len,
            requested_horizon=combo.horizon,
        )
        all_results.append(df)

    raw_df = pd.concat(all_results, ignore_index=True)
    raw_df = raw_df.rename(columns={"origin_index": "origin"})
    raw_df.to_parquet(
        config.RESULTS_DIR / "exp_mtg_benchmark_raw_predictions.parquet", index=False, compression="zstd"
    )

    mase_scales = compute_mase_scales(series_list, boundary_index=int(origins.min()))
    group_cols = (
        "mode",
        "transform",
        "make_positive",
        "context_len",
        "requested_horizon",
        "series",
        "horizon_step",
    )
    leaderboard = summarize_leaderboard(raw_df, mase_scales, group_cols=group_cols)
    leaderboard.to_parquet(config.RESULTS_DIR / "exp_mtg_benchmark_leaderboard.parquet", index=False)

    aggregate = aggregate_leaderboard(leaderboard)
    aggregate.to_parquet(config.RESULTS_DIR / "exp_mtg_benchmark_leaderboard_aggregate.parquet", index=False)

    manifest_payload = {
        "config_id": cfg.config_id,
        "run_id": run_id,
        "adapter": args.adapter,
        "resolved_cards": [c.label for c in cards],
        "grid": {
            "context_lengths": list(cfg.context_lengths),
            "horizons": list(cfg.horizons),
            "n_combos": len(combos),
            "n_origins": len(origins),
        },
        "use_symmetric_averaging": use_symmetric_averaging,
    }
    manifest.write_manifest(
        manifest_payload,
        config.RESULTS_DIR / "manifests" / f"benchmark-{cfg.config_id}-{run_id}.json",
        as_of=dt.date.today(),
    )

    print(
        f"Wrote {len(raw_df)} prediction rows, {len(leaderboard)} leaderboard rows, "
        f"{len(aggregate)} aggregate rows to {config.RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exp_mtg_benchmark_cli.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add scripts/02b_exp_mtg_benchmark.py tests/test_exp_mtg_benchmark_cli.py
git commit -m "feat: add scripts/02b_exp_mtg_benchmark.py CLI with --dry-run"
```

---

## Task 9: `docs/analysis-plan.md` expansion + README pointers

**Files:**
- Modify: `docs/analysis-plan.md`
- Modify: `README.md`

**Interfaces:** none (documentation only) — self-review is the check, not a
pytest run, though ruff/pytest are re-run at the end of this task to confirm
nothing else broke.

- [ ] **Step 1: Rewrite `docs/analysis-plan.md`**

Replace the entire file content with:

```markdown
Primary target:
  prezzo marketplace giornaliero, non consiglio di investimento.

Primary horizons:
  h=28 e h=56 (dentro la griglia completa h in {1, 7, 28, 56, 64} -- vedi
  configs/benchmark_preregistered.example.json). h=28/56 sono i primary
  perche' coprono l'orizzonte "trading window" tipico per una carta
  collezionabile (mese/bimestre), non il one-step banale ne' il tail 64
  dominato dallo stitching multi-pass di TimesFM-3 oltre OUTPUT_PATCH_LENGTH.

Primary metric:
  skill = 1 - MAE_model / MAE_naive (relative MAE = MAE_model / MAE_naive;
  skill ne e' il complemento a 1 -- skill > 0 vuol dire il modello batte la
  naive, skill < 0 il contrario).

  Aggregazione fra carte: mediana (robusta a una carta outlier) E media
  pesata per numero di osservazioni per carta (n) -- mai una media ingenua
  di rapporti (una carta con 5 osservazioni peserebbe come una con 5000).
  Vedi tfm3lab.summarize.aggregate_leaderboard.

Primary comparison:
  TimesFM-3 univariato vs naive.

Secondary comparisons (leaderboard multi-baseline, non solo naive):
  vs seasonal_naive, drift, ets (quando converge) --
    tfm3lab.summarize.summarize_leaderboard;
  multivariato vs univariato;
  multivariato panel vero vs panel placebo (carte random dal pool piu'
    ampio, seed configurabile) -- tfm3lab.benchmark.select_placebo_panel;
  covariata lecita vs nessuna;
  raw vs log1p;
  make_positive True vs False;
  TimesFM-2.5 (zero-shot storico) vs TimesFM-3, quando l'adapter e'
    disponibile (tfm3lab.model_2p5) -- nessun run GPU reale eseguito in
    questo branch.

Ablation grid:
  context_lengths in {64, 128, 256, 512};
  horizons in {1, 7, 28, 56, 64};
  origin set condiviso -- calcolato UNA volta su
  (max(context_lengths)=512, max(horizons)=64), thinnato da origin_stride,
  poi riusato identico per ogni cella della griglia (vedi
  tfm3lab.benchmark.common_origin_set) -- requisito esplicito: "tutte le
  configurazioni devono essere confrontate sulle stesse origini".

Card pool:
  showcase (7 carte, DEFAULT_CARDS) per il talk;
  benchmark manifest piu' ampio (30+) OPZIONALE -- formato + criteri
  documentati in configs/benchmark_cards.example.csv, nessuna selezione
  reale di 30+ carte inventata in questo branch (richiederebbe un crawl
  TCGCSV live, fuori scope).

Exclusion rules:
  righe con observed=False escluse da ogni metrica (forward-fill, non
    un'osservazione reale -- gia' applicato in summarize.py, non toccato
    da questo lavoro);
  ETS escluso dal confronto quando non converge sul contesto dato
    (eccezione soppressa per QUELLA riga in backtest._baseline_forecasts,
    non l'intero run);
  combinazione multivariate_placebo esclusa quando il pool di carte
    disponibili e' piu' piccolo di placebo_panel_size (skip esplicito,
    contato nel dry-run report, mai un gap silenzioso);
  Diebold-Mariano non calcolato sotto MIN_OBSERVATIONS_FOR_DM_TEST
    osservazioni per gruppo (dm_stat/dm_pvalue = NaN, non un numero
    inventato).

Uncertainty:
  paired moving-block bootstrap;
  block length >= horizon;
  correzione BH per p-value multipli.

Claim rule:
  "migliora" solo se CI del delta e' coerente e preregistrata.
```

- [ ] **Step 2: Update `README.md`**

Add a paragraph after the "What's here" table (after line 21, before the
`## Setup` heading):

```markdown
`scripts/02b_exp_mtg_benchmark.py` runs Experiment A as a declarative,
preregistered grid (context lengths x horizons x transform x make_positive x
univariate/multivariate/placebo-panel) instead of the single fixed config
above — see `docs/analysis-plan.md` and
`configs/benchmark_preregistered.example.json`. Its outputs
(`exp_mtg_benchmark_*.parquet`) are separate from `02_exp_mtg.py`'s and never
feed the numbers already committed in `docs/talk-outline.md` or the slides.
```

In the "Running it" code block, after the existing `# 2-5.` block (after the
line `uv run scripts/05_exp_covariates.py`), add:

```bash
# 2b. Preregistered benchmark grid (optional, separate from 2-5 above)
uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --dry-run
uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --as-of <YYYY-MM-DD>
```

In the "Project layout" tree, inside `src/tfm3lab/`, after the
`manifest.py` line, add three lines:

```
  benchmark_config.py  declarative benchmark grid (context/horizon/ablation) schema + loader
  benchmark.py          shared origin set, ablation combo enumeration, placebo panel sampling
  model_2p5.py           TimesFM-2.5 zero-shot adapter (bundled dependency, no new package)
```

Change the `scripts/` line from:
```
scripts/          00-07, numbered in dependency order (see "Running it")
```
to:
```
scripts/          00-07 + 02b, numbered/lettered in dependency order (see "Running it")
```

Add one new line right after the `scripts/` line:
```
configs/          declarative benchmark configs + card manifests (examples, no invented 30+ card selection)
```

- [ ] **Step 3: Run the full suite to confirm nothing broke**

Run: `uv run pytest -q`
Expected: PASS — every test from Tasks 1-8 plus the full pre-existing suite

Run: `uv run ruff check src tests scripts`
Expected: no findings

- [ ] **Step 4: Commit**

```bash
git add docs/analysis-plan.md README.md
git commit -m "docs: expand analysis-plan.md and README for the preregistered MTG benchmark"
```

---

## Final Report (write at the end, in chat)

After all 9 tasks are complete and the final whole-branch review is clean,
write a brief report covering: files changed, tests run (paste the final
`uv run pytest -q` summary line), residual risks (the TimesFM-2.5 quantile
grid is unverified; no real 30+-card benchmark manifest exists yet; this
branch performs no GPU/live run), and migrations needed for old results
(none — `02_exp_mtg.py`'s outputs are untouched).
