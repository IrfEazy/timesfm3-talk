"""scripts/07_build_slides.py is a thin wrapper over the publication gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_slides", REPO / "scripts/07_build_slides.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_has_no_v1_parquet_reads():
    src = (REPO / "scripts/07_build_slides.py").read_text()
    assert "read_parquet" not in src and "figdata" not in src and "pandas" not in src


def test_wrapper_renders_same_output_and_prints_marp_command(capsys):
    out = REPO / "slides/talk.md"
    before = out.read_bytes()
    mod = _load()
    assert [d.output for d in mod.SLIDES] == ["slides/talk.md"]
    assert mod.main() == 0
    assert out.read_bytes() == before  # committed slides are fresh
    printed = capsys.readouterr().out
    assert "npx @marp-team/marp-cli slides/talk.md" in printed
