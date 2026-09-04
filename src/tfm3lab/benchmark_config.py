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
                f"primary_horizons {sorted(unknown_primary)} not present in horizons "
                f"{self.horizons}"
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
