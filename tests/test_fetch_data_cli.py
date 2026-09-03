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
