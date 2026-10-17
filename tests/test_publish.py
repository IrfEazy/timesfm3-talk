"""Offline tests for the publication gate (tfm3lab.publish, scripts/11_render_docs.py)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from tfm3lab import ledger as L
from tfm3lab import publish as P

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "render_script", REPO / "scripts/11_render_docs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sandbox(tmp_path):
    """A temp copy of everything the gate reads (templates, outputs, ledger, claims' sources)."""
    from tfm3lab.claims import SPECS

    paths = {P.ALLOWLIST_PATH, L.LEDGER_JSON, L.LEDGER_MD}
    for d in P.PUBLIC_DOCS:
        paths |= {d.template, d.output}
    paths |= {s.source for s in SPECS}
    for rel in paths:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    for m in (REPO / L.MANIFEST_DIR).glob("*.json"):
        (tmp_path / L.MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(m, tmp_path / L.MANIFEST_DIR / m.name)
    return tmp_path


def _check(root, capsys=None):
    lines: list[str] = []
    rc = P.check_docs(root, out=lines.append)
    return rc, "\n".join(lines)


def test_gate_passes_on_committed_docs():
    lines: list[str] = []
    assert P.check_docs(REPO, out=lines.append) == 0, "\n".join(lines)
    assert _load_script().main(["--check"]) == 0


def test_registry_and_header():
    assert [(d.template, d.output) for d in P.PUBLIC_DOCS] == [
        ("docs/report.md.template", "docs/report.md"),
    ]
    text = (REPO / "docs/report.md").read_text()
    first = text.splitlines()[0]
    assert first == (
        "<!-- generated from docs/report.md.template by scripts/11_render_docs.py; do not edit -->"
    )
    assert "{{" not in text and "" not in text


def test_typed_number_in_template_fails(sandbox):
    tpl = sandbox / "docs/report.md.template"
    tpl.write_text(tpl.read_text() + "\nThe skill is 0.123 here.\n")
    P.write_docs(sandbox, out=lambda *_: None)  # outputs fresh: only the guard can object
    rc, log = _check(sandbox)
    assert rc == 1 and "UNTRACED NUMBER" in log and "'0.123'" in log


def test_stale_rendered_doc_fails(sandbox):
    out = sandbox / "docs/report.md"
    out.write_text(out.read_text().replace("loses to naive", "beats naive", 1))
    rc, log = _check(sandbox)
    assert rc == 1 and "STALE docs/report.md" in log


def test_unknown_placeholder_fails(sandbox):
    tpl = sandbox / "docs/report.md.template"
    tpl.write_text(tpl.read_text() + "\n{{no.such.claim.value}}\n")
    rc, log = _check(sandbox)
    assert rc == 1 and "unknown claim id" in log


def test_unused_allowlist_entry_fails(sandbox):
    p = sandbox / P.ALLOWLIST_PATH
    data = json.loads(p.read_text())
    data["entries"].append(
        {
            "doc": "docs/report.md",
            "literal": "42",
            "context": "answer 42 here",
            "reason": "test",
            "expected_count": 1,
        }
    )
    p.write_text(json.dumps(data))
    rc, log = _check(sandbox)
    assert rc == 1 and "unused allowlist entry" in log


def test_stale_ledger_fails_before_rendering(sandbox):
    p = sandbox / L.LEDGER_JSON
    p.write_text(p.read_text().replace('"inconclusive"', '"beats naive"', 1))
    rc, log = _check(sandbox)
    assert rc == 1 and "ledger is stale" in log


def test_write_mode_regenerates_outputs(sandbox):
    out = sandbox / "docs/report.md"
    out.write_text("tampered")
    assert P.write_docs(sandbox, out=lambda *_: None) == 0
    assert out.read_text() == (REPO / "docs/report.md").read_text()


def test_allowlist_entries_are_documented():
    data = json.loads((REPO / P.ALLOWLIST_PATH).read_text())
    for e in data["entries"]:
        assert e["reason"].strip() and e["expected_count"] >= 1
        assert e["doc"] in {d.output for d in P.PUBLIC_DOCS}


# ---- the narrative must follow the ledger: anchors for sentences that state a result ----------


def _claims():
    data = json.loads((REPO / L.LEDGER_JSON).read_text())
    return {c["id"]: {k: v["raw"] for k, v in c["fields"].items()} for c in data["claims"]}


def test_report_quotes_ledger_verdicts_verbatim():
    c, text = _claims(), (REPO / "docs/report.md").read_text()
    for h in (28, 56):
        v = c[f"a.verdict.pool.h{h}.median"]["verdict"]
        assert f"**{v}**" in text
    assert "inconclusive" in text and "loses to naive" in text
    assert c["b.tfm3.rule.timesfm3_univariate.none.promotion"]["verdict"] == "appendix"
    assert "**Promotion verdict: appendix.**" in text


def test_narrative_assumptions_hold_in_ledger():
    c = _claims()
    # "none was drawn": the sensitivity without showcase cards equals the pool
    for h in (28, 56):
        a, b = c[f"a.verdict.pool.h{h}.median"], c[f"a.verdict.excl_showcase.h{h}.median"]
        assert (a["value"], a["ci"], a["verdict"]) == (b["value"], b["ci"], b["verdict"])
    # forward-fill is uninformative: the only non-observed day is the as-of day
    m = c["m.fetch"]
    assert m["missing_date"] == m["as_of"] and m["missing_count"] == 1
    for k in ("ffill0", "ffill1"):
        r = c[f"a.secondary.{k}_vs_ffill3.h28.median"]
        assert r["value"] == 0 and r["ci"] == [0, 0]
    # MTG preregistered conformal is not evaluable; market h=28 shock has no evaluation rows
    for h in (1, 7, 28, 56):
        assert c[f"c.conformal.preregistered.mtg.h{h}"]["status"] == "no_calibration_rows"
    assert c["c.conformal.preregistered.market_shock.h28"]["n"] == 0
    # h=1 exploratory: not under-covering before recalibration
    assert c["c.conformal.exploratory_post_hoc_resplit.mtg.h1"]["under_covers_before"] is False
    # covariates: all inconclusive; market legit vs no covariate (all horizons) excludes 0,
    # but the placebo comparison does not
    for k, v in c.items():
        if k.startswith("d.cov."):
            assert v["verdict"] == "inconclusive"
    nocov = c["d.cov.market.legit_vs_no_covariate.hall.weighted_mean"]
    placebo = c["d.cov.market.legit_vs_placebo.hall.weighted_mean"]
    assert nocov["ci_low"] > 0 and placebo["ci_low"] <= 0
    # stress scenario under-covers at the acceptance cluster count
    stress = c["s.selfcheck.stress.g40.median"]
    assert stress["mc_ci_high"] < c["m.covariates"]["ci_level"]
    assert c["m.selfcheck"]["verdict"] == "PASS"


def test_template_leading_comment_digit_is_flagged(sandbox):
    tpl = sandbox / "docs/report.md.template"
    tpl.write_text("<!-- draft 7 of the report -->\n" + tpl.read_text())
    P.write_docs(sandbox, out=lambda *_: None)
    rc, log = _check(sandbox)
    assert rc == 1 and "UNTRACED NUMBER" in log and "'7'" in log


def test_v3_claims_match_confirmatory_verdicts_and_not_exploratory_intervals():
    claims = _claims()
    verdicts = json.loads((REPO / "results/v3_verdicts.json").read_text())
    report = (REPO / "docs/report.md").read_text()
    for verdict in verdicts:
        c = claims[f"v3.{verdict['study'].lower()}"]
        assert c["verdict"] == verdict["verdict"]
        assert c["value"] == verdict["point"]
        assert c["n"] == verdict["n"]
        assert c["status"] == verdict["status"] == "complete"
        interval = verdict["intervals"].get("t", verdict["intervals"].get("cluster"))
        assert [c["ci_low"], c["ci_high"]] == interval
        if "block" in verdict["intervals"]:
            assert [c["cal_ci_low"], c["cal_ci_high"]] == verdict["intervals"]["block"]
        assert f"**{c['verdict']}**" in report
    # This narrative must continue to match the committed census/evaluation.
    s3 = json.loads((REPO / "results/v3_s3_summary.json").read_text())
    assert s3["confirmatory"] is True
    s1 = claims["v3.s1"]
    assert s1["sensitivity_low"] > 0
    assert "does not replace the primary verdict" in report
    # Rare floor hits must not round to zero in the public table.
    ledger = json.loads((REPO / L.LEDGER_JSON).read_text())
    floor = next(c for c in ledger["claims"] if c["id"] == "v3.floor.s2")
    assert float(floor["fields"]["median_steps"]["text"]) > 0


def test_changed_v3_results_fail_manifest_validation(sandbox):
    source = sandbox / "results/v3_s2_summary.json"
    data = json.loads(source.read_text())
    data["primary"]["mean"] = 0.5
    source.write_text(json.dumps(data))
    rc, log = _check(sandbox)
    assert rc == 1 and "does not match the newest manifest" in log
