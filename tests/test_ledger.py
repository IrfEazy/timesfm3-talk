"""Offline tests for tfm3lab.ledger / tfm3lab.claims (synthetic tables and manifests)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tfm3lab import ledger as L
from tfm3lab.claims import EXEMPTIONS, SPECS
from tfm3lab.ledger import (
    AllowedLiteral,
    B,
    ClaimSpec,
    F,
    I,
    LedgerError,
    T,
    find_untraced_numbers,
    format_decimal,
    render,
    when,
)

REPO = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_root(tmp_path: Path, df: pd.DataFrame | None = None, layout: str = "list") -> Path:
    if df is None:
        df = pd.DataFrame(
            {
                "grp": ["x", "x", "y"],
                "h": [28, 56, 28],
                "est": [0.0125, -0.0004, 1.5],
                "lo": [0.0, -0.1, 1.0],
                "hi": [0.0375, 0.1, 2.0],
                "n": [40, 40, 41],
                "verdict": ["a|b", "inconclusive", "pass"],
            }
        )
    (tmp_path / "results/manifests").mkdir(parents=True)
    src = tmp_path / "results/t.parquet"
    df.to_parquet(src)
    write_manifest(tmp_path, src, layout)
    return tmp_path


def write_manifest(root: Path, src: Path, layout: str = "list", name: str = "m1.json") -> None:
    entry = {"path": f"/kaggle/tmp/{src.name}", "sha256": sha(src)}
    if layout == "list":
        data = {"outputs": [entry]}
    elif layout == "dict":
        data = {"outputs": {"t": entry}}
    else:
        data = {"output": f"results/{src.name}", "output_sha256": sha(src)}
    (root / "results/manifests" / name).write_text(json.dumps(data))


SPEC = ClaimSpec(
    id="t.{grp}.h{h}",
    source="results/t.parquet",
    group="T: test",
    expand=("grp", "h"),
    expected_claims=3,
    fields={
        "value": F("est"),
        "ci_low": F("lo"),
        "ci_high": F("hi"),
        "n": I("n"),
        "verdict": T("verdict"),
    },
)


# ------------------------------------------------------------------ formatting


def test_format_ties_half_even_and_negative_zero():
    assert format_decimal(0.0125, 3) == "0.012"
    assert format_decimal(0.0135, 3) == "0.014"
    assert format_decimal(-0.0004, 3) == "0.000"
    assert format_decimal(-0.0, 3) == "0.000"
    assert format_decimal(-0.0, 0) == "0"
    assert format_decimal(2, 2) == "2.00"
    assert format_decimal(-1.2345, 3) == "-1.234"


def test_non_finite_and_integer_policy():
    with pytest.raises(LedgerError):
        format_decimal(float("inf"), 3)
    assert I("c").format(40.0) == (40, "40")
    with pytest.raises(LedgerError):
        I("c").format(40.5)
    with pytest.raises(LedgerError):
        F("c").format(float("nan"))
    cond = when(s="bad")
    assert F("c", nan_if=cond).format(float("nan"), {"s": "bad"}) == (None, "n/a")
    with pytest.raises(LedgerError, match="nan_if"):
        F("c", nan_if=cond).format(float("nan"), {"s": "ok"})
    with pytest.raises(LedgerError):
        F("c", nan_if=cond).format(float("nan"))  # no row
    with pytest.raises(LedgerError):
        F("c", nan_if=cond).format(float("inf"), {"s": "bad"})
    assert T("c", nan_if=[{"s": ["x", "bad"]}]).format(None, {"s": "bad"}) == (None, "n/a")
    either = F("c", nan_if=(when(n=0), when(s="bad")))
    assert either.format(float("nan"), {"n": 0, "s": "ok"}) == (None, "n/a")
    assert either.format(float("nan"), {"n": 5, "s": "bad"}) == (None, "n/a")
    with pytest.raises(LedgerError):
        either.format(float("nan"), {"n": 5, "s": "ok"})
    assert B("c").format(True) == (True, "true")


# ------------------------------------------------------------------ build


def test_build_values_ci_and_manifest(tmp_path):
    root = make_root(tmp_path)
    led = L.build_ledger([SPEC], root)
    c = {c["id"]: c for c in led["claims"]}
    f = c["t.x.h28"]["fields"]
    assert f["value"]["text"] == "0.012"
    assert f["ci"]["text"] == "[0.000, 0.038]"  # 0.0375 -> half-even 0.038
    assert f["n"]["raw"] == 40
    assert c["t.x.h28"]["manifest"] == "m1.json"
    assert c["t.y.h28"]["source_sha256"] == sha(root / "results/t.parquet")
    assert list(c) == sorted(c)


@pytest.mark.parametrize("layout", ["list", "dict", "single"])
def test_manifest_layouts(tmp_path, layout):
    root = make_root(tmp_path, layout=layout)
    assert L.build_ledger([SPEC], root)["claims"][0]["manifest"] == "m1.json"


def test_latest_manifest_by_filename(tmp_path):
    root = make_root(tmp_path)
    write_manifest(root, root / "results/t.parquet", "dict", name="m2.json")
    assert L.build_ledger([SPEC], root)["claims"][0]["manifest"] == "m2.json"


def _stamp(root, name, ts):
    p = root / "results/manifests" / name
    d = json.loads(p.read_text())
    d["_meta"] = {"written_at_utc": ts}
    p.write_text(json.dumps(d))


def test_newest_manifest_must_match_older_match_insufficient(tmp_path):
    root = make_root(tmp_path)
    src = root / "results/t.parquet"
    write_manifest(root, src, "list", name="m2.json")
    m2 = root / "results/manifests/m2.json"
    d = json.loads(m2.read_text())
    d["outputs"][0]["sha256"] = "0" * 64  # newer manifest records different bytes
    m2.write_text(json.dumps(d))
    # by filename order: m2 is newest and disagrees
    with pytest.raises(LedgerError, match="newest manifest m2.json"):
        L.build_ledger([SPEC], root)
    # by recorded timestamp: m1 (matching) is older -> still an error
    _stamp(root, "m1.json", "2026-10-01T00:00:00+00:00")
    _stamp(root, "m2.json", "2026-10-02T00:00:00+00:00")
    with pytest.raises(LedgerError, match="newest manifest m2.json"):
        L.build_ledger([SPEC], root)
    # timestamps beat filename order: m1 stamped newer than m2 -> m1 is newest and matches
    _stamp(root, "m1.json", "2026-10-03T00:00:00+00:00")
    assert L.build_ledger([SPEC], root)["claims"][0]["manifest"] == "m1.json"


def test_parquet_changed_but_manifest_unchanged(tmp_path):
    root = make_root(tmp_path)
    df = pd.read_parquet(root / "results/t.parquet")
    df.loc[0, "est"] = 0.5
    df.to_parquet(root / "results/t.parquet")
    with pytest.raises(LedgerError, match="without a manifest"):
        L.build_ledger([SPEC], root)


def test_no_manifest_error_unless_exempt(tmp_path):
    root = make_root(tmp_path)
    (root / "results/manifests/m1.json").unlink()
    with pytest.raises(LedgerError, match="no manifest"):
        L.build_ledger([SPEC], root)
    led = L.build_ledger([SPEC], root, {"results/t.parquet": "legacy output"})
    assert led["claims"][0]["manifest"] is None
    assert led["claims"][0]["exemption"] == "legacy output"
    with pytest.raises(LedgerError, match="reason"):
        L.build_ledger([SPEC], root, {"results/t.parquet": " "})


def test_selector_matching_two_rows(tmp_path):
    df = pd.DataFrame({"g": ["x", "x"], "est": [1.0, 2.0]})
    root = make_root(tmp_path, df)
    spec = ClaimSpec(
        "t.dup", "results/t.parquet", "T", select={"g": "x"}, fields={"value": F("est")}
    )
    with pytest.raises(LedgerError, match="t.dup.*2 rows"):
        L.build_ledger([spec], root)
    spec0 = ClaimSpec(
        "t.none", "results/t.parquet", "T", select={"g": "z"}, fields={"value": F("est")}
    )
    with pytest.raises(LedgerError, match="t.none.*0 rows"):
        L.build_ledger([spec0], root)


def test_expand_with_duplicate_rows_errors(tmp_path):
    df = pd.DataFrame({"g": ["x", "x"], "est": [1.0, 2.0]})
    root = make_root(tmp_path, df)
    spec = ClaimSpec(
        "t.{g}",
        "results/t.parquet",
        "T",
        expand=("g",),
        expected_claims=1,
        fields={"value": F("est")},
    )
    with pytest.raises(LedgerError, match="t.x"):
        L.build_ledger([spec], root)


def test_missing_column_nan_and_duplicate_ids(tmp_path):
    df = pd.DataFrame({"g": ["a", "A"], "est": [1.0, float("nan")]})
    root = make_root(tmp_path, df)
    bad_col = ClaimSpec(
        "t.c", "results/t.parquet", "T", select={"g": "a"}, fields={"value": F("nope")}
    )
    with pytest.raises(LedgerError, match="nope"):
        L.build_ledger([bad_col], root)
    nan = ClaimSpec("t.n", "results/t.parquet", "T", select={"g": "A"}, fields={"value": F("est")})
    with pytest.raises(LedgerError, match="t.n.value.*missing"):
        L.build_ledger([nan], root)
    bad_cond = ClaimSpec(
        "t.n",
        "results/t.parquet",
        "T",
        select={"g": "A"},
        fields={"value": F("est", nan_if=when(g="B"))},
    )
    with pytest.raises(LedgerError, match="t.n.value.*nan_if"):
        L.build_ledger([bad_cond], root)
    ok = ClaimSpec(
        "t.n",
        "results/t.parquet",
        "T",
        select={"g": "A"},
        fields={"value": F("est", nan_if=when(g="A"))},
    )
    assert L.build_ledger([ok], root)["claims"][0]["fields"]["value"] == {
        "raw": None,
        "text": "n/a",
    }
    # ids lower-case: "a" and "A" collide
    coll = ClaimSpec(
        "t.{g}",
        "results/t.parquet",
        "T",
        expand=("g",),
        expected_claims=2,
        fields={"value": F("est", nan_if=when(g=["A", "a"]))},
    )
    with pytest.raises(LedgerError, match="duplicate claim id 't.a'"):
        L.build_ledger([coll], root)
    with pytest.raises(LedgerError, match="duplicate claim id"):
        L.build_ledger([ok, ok], root)


def test_expected_claims_mismatch_and_required(tmp_path):
    root = make_root(tmp_path)
    wrong = ClaimSpec(
        "t.{grp}.h{h}",
        "results/t.parquet",
        "T",
        expand=("grp", "h"),
        expected_claims=4,
        fields={"value": F("est")},
    )
    with pytest.raises(LedgerError, match="expected 4 claims, expanded to 3"):
        L.build_ledger([wrong], root)
    with pytest.raises(LedgerError, match="expected_claims"):
        ClaimSpec("t.{grp}", "results/t.parquet", "T", expand=("grp",), fields={})


def test_bad_id_characters(tmp_path):
    root = make_root(tmp_path)
    spec = ClaimSpec(
        "Bad Id", "results/t.parquet", "T", select={"grp": "y"}, fields={"value": F("est")}
    )
    with pytest.raises(LedgerError, match="a-z0-9"):
        L.build_ledger([spec], root)


def test_count_claims(tmp_path):
    df = pd.DataFrame(
        {
            "fam": ["a"] * 4 + ["b"] * 2,
            "rej": pd.array([True, False, None, True, False, False], dtype="boolean"),
            "size": [10] * 4 + [7, 8],
        }
    )
    root = make_root(tmp_path, df)
    spec = ClaimSpec(
        "t.{fam}",
        "results/t.parquet",
        "T",
        expand=("fam",),
        expected_claims=2,
        counts={"rejections": {"rej": True}, "n": {}},
        unique={"family_size": I("size")},
    )
    with pytest.raises(LedgerError, match="t.b.family_size.*2 distinct"):
        L.build_ledger([spec], root)
    ok = ClaimSpec(
        "t.a",
        "results/t.parquet",
        "T",
        select={"fam": "a"},
        counts={"rejections": {"rej": True}, "n": {}},
        unique={"family_size": I("size")},
    )
    f = L.build_ledger([ok], root)["claims"][0]["fields"]
    assert (f["rejections"]["raw"], f["n"]["raw"], f["family_size"]["text"]) == (2, 4, "10")


def test_deterministic_regeneration(tmp_path):
    root = make_root(tmp_path)
    a = L.generate(root, [SPEC])
    b = L.generate(root, list(reversed([SPEC])))
    assert a == b
    assert L._cell("a|b\nc") == "a\\|b c"  # pipes escaped in Markdown
    assert "| id | value | CI | n | source | manifest |" in a[1]


def test_run_write_and_check_fails_on_hand_edit(tmp_path):
    root = make_root(tmp_path)
    msgs: list[str] = []
    assert L.run(root, [SPEC], out=msgs.append) == 0
    assert L.run(root, [SPEC], check=True, out=msgs.append) == 0
    p = root / L.LEDGER_JSON
    p.write_text(p.read_text().replace('"0.012"', '"0.999"', 1))
    msgs.clear()
    assert L.run(root, [SPEC], check=True, out=msgs.append) == 1
    assert any("STALE results/evidence_ledger.json" in m and "0.999" in m for m in msgs)
    L.run(root, [SPEC], out=msgs.append)
    md = root / L.LEDGER_MD
    md.write_text(md.read_text() + "extra\n")
    assert L.run(root, [SPEC], check=True, out=msgs.append) == 1
    assert L.run(root, [SPEC], out=msgs.append) == 0
    assert L.run(root, [SPEC], check=True, out=msgs.append) == 0


def test_run_reports_ledger_error(tmp_path):
    root = make_root(tmp_path)
    (root / "results/t.parquet").write_bytes(b"x")
    msgs: list[str] = []
    assert L.run(root, [SPEC], out=msgs.append) == 1


# ------------------------------------------------------------------ renderer + guard


@pytest.fixture
def led(tmp_path):
    return L.build_ledger([SPEC], make_root(tmp_path))


def test_render_fields(led):
    t = "est {{t.x.h28.value}} ci {{ t.x.h28.ci }} n={{t.x.h28.n}} {{t.y.h28.verdict}}"
    assert render(t, led) == "est 0.012 ci [0.000, 0.038] n=40 pass"


@pytest.mark.parametrize(
    "tpl",
    [
        "{{t.zzz.value}}",
        "{{t.x.h28.nope}}",
        "{{value}}",
        "broken {{ t.x.h28.value",
        "{{t.x.h28.value}} {{",
    ],
)
def test_render_errors(led, tpl):
    with pytest.raises(LedgerError):
        render(tpl, led)


def test_guard_flags_untraced_even_if_equal_to_claim_value(led):
    tpl = "Skill is {{t.x.h28.value}} but also 0.012 and -3.5%, 1,234 and 2026-10-12 at h28."
    marked = render(tpl, led, mark=True)
    toks = [u.literal for u in find_untraced_numbers(marked, [])]
    assert toks == ["0.012", "-3.5%", "1,234", "2026", "10", "12", "28"]
    assert L.strip_marks(marked) == render(tpl, led)


def test_guard_ignores_placeholder_output_but_flags_adjacent(led):
    marked = render("{{t.x.h28.value}}5 {{t.x.h28.ci}}", led, mark=True)
    assert [u.literal for u in find_untraced_numbers(marked, [])] == ["5"]


def test_guard_does_not_exempt_links_or_code(led):
    text = "[a](https://x.org/v2/3) and\n```\nrun 42\n```\n"
    toks = [u.literal for u in find_untraced_numbers(render(text, led, mark=True), [])]
    assert toks == ["2", "3", "42"]


def test_allowlist_scoped_and_counted(led):
    entry = AllowedLiteral("README.md", "2026", "As of 2026-10-12 we", "date of data snapshot", 1)
    ok = render("As of 2026-10-12 we ran.", led, mark=True)
    left = find_untraced_numbers(ok, [entry], doc="README.md")
    assert [u.literal for u in left] == ["10", "12"]
    # other doc: not exempt
    assert "2026" in [u.literal for u in find_untraced_numbers(ok, [entry], doc="other.md")]
    # same literal in a different context stays flagged
    two = render("As of 2026-10-12 we ran in 2026.", led, mark=True)
    assert "2026" in [u.literal for u in find_untraced_numbers(two, [entry], doc="README.md")]
    # an extra occurrence of the allowlisted context is a count mismatch
    extra = render("As of 2026-10-12 we ran. As of 2026-10-12 we ran.", led, mark=True)
    with pytest.raises(LedgerError, match="expected 1.*found 2"):
        find_untraced_numbers(extra, [entry], doc="README.md")
    with pytest.raises(LedgerError, match="expected 1.*found 0"):
        find_untraced_numbers("nothing", [entry], doc="README.md")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("x \u22123.5 y", ["\u22123.5"]),
        ("range 10\u201320", ["10", "20"]),
        ("1,234,567", ["1,234,567"]),
        ("1 234 567.5", ["1 234 567.5"]),
        ("1\u2009234", ["1\u2009234"]),
        ("1\u00a0234", ["1\u00a0234"]),
        ("1e-3 and 1.2E+04", ["1e-3", "1.2E+04"]),
        ("a 12.5% b", ["12.5%"]),
        ("is .5 ok", [".5"]),
        ("-.5e-3", ["-.5e-3"]),
        ("v1.5", ["1.5"]),
    ],
)
def test_guard_single_tokens(text, expected):
    assert [u.literal for u in find_untraced_numbers(text, [])] == expected


def test_guard_provenance_is_per_occurrence(led):
    marked = render("{{t.x.h28.value}} then typed 0.012 and {{t.x.h28.value}}", led, mark=True)
    hits = find_untraced_numbers(marked, [])
    assert [u.literal for u in hits] == ["0.012"]
    assert "typed 0.012" in hits[0].context
    with pytest.raises(LedgerError, match="reserved"):
        render("forged \ue0000.012\ue001", led)


def test_unused_allowlist_entry_is_error():
    entry = AllowedLiteral("d.md", "7", "has 7 items", "why", 1)
    with pytest.raises(LedgerError, match="unused allowlist entry"):
        find_untraced_numbers("no numbers here", [entry], doc="d.md")


def test_allowlist_entry_validation():
    with pytest.raises(LedgerError):
        AllowedLiteral("d", "5", "5 and 5", "r", 1)
    with pytest.raises(LedgerError):
        AllowedLiteral("d", "5", "a 5", " ", 1)


# ------------------------------------------------------------------ committed ledger


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "ledger_script", REPO / "scripts/10_evidence_ledger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_ledger_is_fresh():
    json_text, md_text = L.generate(REPO, SPECS, EXEMPTIONS)
    assert (REPO / L.LEDGER_JSON).read_text() == json_text
    assert (REPO / L.LEDGER_MD).read_text() == md_text
    assert _load_script().main(["--check"]) == 0


def test_committed_ledger_covers_required_tables():
    data = json.loads((REPO / L.LEDGER_JSON).read_text())
    ids = [c["id"] for c in data["claims"]]
    assert len(ids) == len(set(ids))
    for prefix in (
        "a.verdict.", "a.secondary.", "a.sig.", "c.coverage.", "c.conformal.",
        "b.tfm3.", "b.tfm25.", "d.cov.", "l.tfm3.", "l.tfm25.", "s.selfcheck.",
    ):  # fmt: skip
        assert any(i.startswith(prefix) for i in ids), prefix
    assert all(c["manifest"] or c.get("exemption") for c in data["claims"])


def test_metadata_prefixes_present():
    data = json.loads((REPO / L.LEDGER_JSON).read_text())
    ids = {c["id"] for c in data["claims"]}
    assert {"m.code", "m.grid", "m.fetch", "m.events", "m.inference", "m.selfcheck"} <= ids
    assert any(i.startswith("a.lb.") for i in ids) and any(i.startswith("a.lb25.") for i in ids)


# ------------------------------------------------------------------ json / python sources


def _meta_root(tmp_path: Path) -> Path:
    (tmp_path / "results/manifests").mkdir(parents=True)
    (tmp_path / "cfg.json").write_text(
        json.dumps(
            {
                "as_of": "2026-09-04",
                "horizons": [1, 7, 28],
                "nested": {"seed": 7, "band": [0.92, 0.98], "flag": True},
                "records": [{"x": 1}],
            }
        )
    )
    (tmp_path / "consts.py").write_text(
        "from datetime import date\n"
        "CUTOFF = date(2023, 11, 30)\n"
        "NAME = 'google/x'\n"
        "NEG = -2\n"
        "DYN = some_call()\n"
        "OTHER = 1 + 2\n"
    )
    return tmp_path


def test_json_and_python_sources(tmp_path):
    root = _meta_root(tmp_path)
    specs = [
        ClaimSpec(
            id="m.cfg",
            source="cfg.json",
            group="M",
            fields={
                "as_of": T("as_of"),
                "hs": L.LST("horizons"),
                "h2": I("horizons.1"),
                "seed": I("nested.seed"),
                "lo": F("nested.band.0", 2),
                "flag": B("nested.flag"),
            },
        ),
        ClaimSpec(
            id="m.py",
            source="consts.py",
            group="M",
            fields={"cutoff": L.D("CUTOFF"), "name": T("NAME"), "neg": I("NEG")},
        ),
    ]
    ex = {"cfg.json": "committed config", "consts.py": "committed constants"}
    ledger = L.build_ledger(specs, root, ex)
    c = {x["id"]: x["fields"] for x in ledger["claims"]}
    assert c["m.cfg"]["hs"]["text"] == "1, 7, 28" and c["m.cfg"]["hs"]["raw"] == [1, 7, 28]
    assert c["m.cfg"]["h2"]["text"] == "7" and c["m.cfg"]["lo"]["text"] == "0.92"
    assert c["m.cfg"]["flag"]["text"] == "true"
    assert c["m.py"]["cutoff"]["text"] == "2023-11-30" and c["m.py"]["neg"]["text"] == "-2"
    # sha is pinned even for exempt sources, and an unlisted source needs an exemption
    assert ledger["sources"]["cfg.json"]["exemption"] == "committed config"
    with pytest.raises(LedgerError, match="no manifest records"):
        L.build_ledger(specs, root, {})
    # dynamic constants are skipped, not evaluated
    bad = ClaimSpec(id="m.bad", source="consts.py", group="M", fields={"x": I("DYN")})
    with pytest.raises(LedgerError, match="missing column"):
        L.build_ledger([bad], root, ex)


def test_date_and_list_kinds_reject_bad_values():
    with pytest.raises(LedgerError, match="not a date"):
        L.D("c").format("not-a-date")
    assert L.D("c").format(pd.Timestamp("2026-01-02")) == ("2026-01-02", "2026-01-02")
    with pytest.raises(LedgerError, match="non-empty list"):
        L.LST("c").format([])
    with pytest.raises(LedgerError, match="not str/int"):
        L.LST("c").format([1.5])
