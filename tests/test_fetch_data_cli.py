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


def test_skip_mtg_manifest_records_skip_and_outputs(fetch01, tmp_path):
    import json

    market = tmp_path / "market_prices.parquet"
    cpi = tmp_path / "cpi_yoy.parquet"
    market.write_bytes(b"m")
    cpi.write_bytes(b"c")
    path = fetch01.write_skip_mtg_manifest(
        dt.date(2026, 9, 1), False, [market, cpi], manifest_dir=tmp_path / "man"
    )
    data = json.loads(path.read_text())
    assert data["mtg"] == "skipped"
    assert [o["path"] for o in data["outputs"]] == [str(market), str(cpi)]
    assert data["_meta"]["as_of"] == "2026-09-01"


def _fake_ingest(captured):
    import numpy as np
    import pandas as pd

    from tfm3lab.backtest import SeriesData
    from tfm3lab.data.mtg import IngestReport

    def fake(cards, start, end, price_policy, cached_only=False):
        captured["cards"] = tuple(cards)
        captured["cached_only"] = cached_only
        dates = pd.date_range("2026-08-01", periods=5).to_numpy()
        series = [SeriesData(c.label, np.ones(5), dates, np.ones(5, dtype=bool)) for c in cards]
        report = IngestReport(
            resolved_cards=pd.DataFrame({"label": [c.label for c in cards]}),
            archive_hashes={},
            price_field_counts={},
            subtype_counts={},
            missing_dates=["2026-08-03"] if cached_only else [],
            cached_archives_only=cached_only,
            catalog={"dir": "catalog-x", "sha256": "0" * 64} if cached_only else None,
        )
        return series, report

    return fake


def test_cards_default_is_showcase(fetch01):
    from tfm3lab.data.mtg import DEFAULT_CARDS

    cards, records = fetch01.resolve_card_sources(None)
    assert cards == DEFAULT_CARDS
    assert [r["source"] for r in records] == ["showcase"] and len(records[0]["sha256"]) == 64


def test_cards_union_dedups_and_hashes_manifest(fetch01, tmp_path):
    import hashlib

    from tfm3lab.data.mtg import DEFAULT_CARDS

    csv = tmp_path / "pool.csv"
    first = DEFAULT_CARDS[0]
    csv.write_text(
        "label,group_abbreviation,product_name,product_id\n"
        f'{first.label},{first.group_abbreviation},"{first.product_name}",\n'
        "Other Card,ZZZ,Other Card,4242\n"
    )
    cards, records = fetch01.resolve_card_sources(["showcase", str(csv)])
    assert len(cards) == len(DEFAULT_CARDS) + 1
    other = cards[-1]
    assert other.label == "Other Card" and other.product_id == 4242
    assert records[1]["sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert records[1]["n_cards"] == 2


def test_conflicting_card_label_is_an_error(fetch01, tmp_path):
    from tfm3lab.data.mtg import DEFAULT_CARDS

    csv = tmp_path / "pool.csv"
    csv.write_text(
        "label,group_abbreviation,product_name,product_id\n"
        f"{DEFAULT_CARDS[0].label},ZZZ,Different,99\n"
    )
    with pytest.raises(SystemExit, match="defined differently"):
        fetch01.resolve_card_sources(["showcase", str(csv)])


def test_main_builds_union_cache_and_records_card_sources(fetch01, tmp_path, monkeypatch):
    import json

    import pandas as pd

    csv = tmp_path / "pool.csv"
    csv.write_text(
        "label,group_abbreviation,product_name,product_id\nPool Card,ZZZ,Pool Card,4242\n"
    )
    captured: dict = {}
    monkeypatch.setattr(fetch01, "build_card_series", _fake_ingest(captured))
    monkeypatch.setattr(fetch01.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(fetch01.config, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "cache").mkdir()
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(fetch01.plots, "save", lambda *a, **k: None)
    monkeypatch.setattr(fetch01.plots, "plot_data_quality", lambda *a, **k: None)
    monkeypatch.setattr(fetch01, "build_market_series", lambda end: [])
    monkeypatch.setattr(fetch01, "build_cpi_series", lambda end: None)
    monkeypatch.setattr(fetch01, "_series_list_to_frame", _frame_stub(fetch01))
    monkeypatch.setattr(
        sys, "argv", ["01", "--as-of", "2026-09-04", "--cards", "showcase", "--cards", str(csv)]
    )
    try:  # noqa: SIM105
        fetch01.main()
    except IndexError:  # the empty market stub ends the run after the MTG part
        pass
    labels = [c.label for c in captured["cards"]]
    assert "Pool Card" in labels and len(labels) == 8
    assert [c for c in captured["cards"] if c.label == "Pool Card"][0].product_id == 4242
    (man,) = (tmp_path / "results" / "manifests").glob("fetch-2026-09-04-*.json")
    sources = json.loads(man.read_text())["card_sources"]
    assert [s["source"] for s in sources] == ["showcase", str(csv)]
    assert all(len(s["sha256"]) == 64 for s in sources)
    assert isinstance(pd.read_parquet(tmp_path / "cache" / "mtg_prices.parquet"), pd.DataFrame)


def _frame_stub(fetch01):
    original = fetch01._series_list_to_frame

    def stub(series_list):
        if not series_list:
            import pandas as pd

            return pd.DataFrame({"date": [], "value": [], "observed": [], "series": []})
        return original(series_list)

    return stub


def test_cached_archives_only_flag_parses(fetch01):
    parser = fetch01.build_parser()
    assert parser.parse_args(["--as-of", "2026-09-04"]).cached_archives_only is False
    args = parser.parse_args(["--as-of", "2026-09-04", "--cached-archives-only"])
    assert args.cached_archives_only is True


def test_main_cached_archives_only_records_manifest(fetch01, tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(fetch01, "build_card_series", _fake_ingest(captured))
    monkeypatch.setattr(fetch01.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(fetch01.config, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "cache").mkdir()
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(fetch01.plots, "save", lambda *a, **k: None)
    monkeypatch.setattr(fetch01.plots, "plot_data_quality", lambda *a, **k: None)
    monkeypatch.setattr(fetch01, "build_market_series", lambda end: [])
    monkeypatch.setattr(fetch01, "build_cpi_series", lambda end: None)
    monkeypatch.setattr(fetch01, "_series_list_to_frame", _frame_stub(fetch01))
    monkeypatch.setattr(sys, "argv", ["01", "--as-of", "2026-09-04", "--cached-archives-only"])
    try:  # noqa: SIM105
        fetch01.main()
    except IndexError:  # the empty market stub ends the run after the MTG part
        pass
    assert captured["cached_only"] is True
    (man,) = (tmp_path / "results" / "manifests").glob("fetch-2026-09-04-*.json")
    import json

    payload = json.loads(man.read_text())
    assert payload["cached_archives_only"] is True
    assert payload["missing_dates"] == ["2026-08-03"]
    assert payload["missing_dates_count"] == 1
    assert payload["catalog_snapshot"]["dir"] == "catalog-x"
