"""Kaggle run helpers (Preregistration v2 GPU steps) and the runtime-probe maths.

Everything the Kaggle notebook (notebooks/run_on_kaggle.ipynb) does beyond
"run this script" lives here so it is unit-testable offline: the ordered step
table, a subprocess runner that records exit status and elapsed time, cache
verification against the committed manifests, restore from a previous
session's output, the two-archive packing, and the throughput projection used
by scripts/00b_runtime_probe.py.

The module imports only the standard library at import time (the notebook
kernel is not the uv environment). pandas, numpy and the model code are
imported lazily inside the functions that need them. No runtime is ever
hard-coded: every projected duration is derived from throughput measured by
the probe on the GPU actually attached.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- step table


@dataclasses.dataclass(frozen=True)
class Step:
    """One GPU step. `args` are the script arguments before `--as-of`."""

    step_id: str
    script: str
    args: tuple[str, ...] = ()
    adapter: str | None = None  # "timesfm3" | "timesfm2.5" | None
    title: str = ""


# Order matters: the primary preregistered claim first, then the secondary
# configs, then the shock / covariate / latency experiments.
STEPS: tuple[Step, ...] = (
    Step("probe", "scripts/00b_runtime_probe.py", (), None, "runtime probe"),
    Step(
        "02b_preregistered_v2",
        "scripts/02b_exp_mtg_benchmark.py",
        ("--config", "configs/benchmark_preregistered_v2.json", "--resume"),
        "timesfm3",
        "Experiment A, preregistered v2 grid (primary claim)",
    ),
    Step(
        "02b_timesfm2p5",
        "scripts/02b_exp_mtg_benchmark.py",
        ("--config", "configs/benchmark_v2_timesfm2p5.json", "--adapter", "timesfm2.5", "--resume"),
        "timesfm2.5",
        "Experiment A, TimesFM-2.5 comparison grid",
    ),
    Step(
        "02b_showcase",
        "scripts/02b_exp_mtg_benchmark.py",
        ("--config", "configs/benchmark_v2_showcase.json", "--resume"),
        "timesfm3",
        "Experiment A, showcase cards (descriptive)",
    ),
    Step(
        "02b_ffill0",
        "scripts/02b_exp_mtg_benchmark.py",
        ("--config", "configs/benchmark_v2_ffill0.json", "--resume"),
        "timesfm3",
        "Experiment A, forward-fill sensitivity (0 days)",
    ),
    Step(
        "02b_ffill1",
        "scripts/02b_exp_mtg_benchmark.py",
        ("--config", "configs/benchmark_v2_ffill1.json", "--resume"),
        "timesfm3",
        "Experiment A, forward-fill sensitivity (1 day)",
    ),
    Step("03_timesfm3", "scripts/03_exp_shock.py", (), "timesfm3", "Experiment B, shock events"),
    Step(
        "03_timesfm2p5",
        "scripts/03_exp_shock.py",
        ("--adapter", "timesfm2.5"),
        "timesfm2.5",
        "Experiment B, shock events with TimesFM-2.5",
    ),
    Step(
        "05_covariates", "scripts/05_exp_covariates.py", (), "timesfm3", "Experiment C, covariates"
    ),
    Step("09_latency_timesfm3", "scripts/09_latency.py", (), "timesfm3", "latency, TimesFM-3"),
    Step(
        "09_latency_timesfm2p5",
        "scripts/09_latency.py",
        ("--adapter", "timesfm2.5"),
        "timesfm2.5",
        "latency, TimesFM-2.5",
    ),
    # v3 studies: appended last so the ids and order above stay unchanged. It reads
    # data/cache/v3_requests.parquet (scripts/14_v3_forecast.py default), which the
    # attached PRIVATE cache dataset provides: copy_dataset_into_cache copies it with
    # the other parquet files and verify_cache checks it against the committed
    # cache manifest like any other cache file (rebuild that manifest after adding it).
    Step(
        "v3_forecasts",
        "scripts/14_v3_forecast.py",
        ("--resume",),
        "timesfm3",
        "v3 studies: S1/S2/S3 forecasts",
    ),
)

STEP_BY_ID: dict[str, Step] = {s.step_id: s for s in STEPS}
# Steps that never run in the default (Preregistration v2) session: they need inputs
# the v2 cache dataset does not hold, so a session opts in by naming them in EXPERIMENTS.
OPT_IN_STEPS: frozenset[str] = frozenset({"v3_forecasts"})
DEFAULT_EXPERIMENTS: tuple[str, ...] = tuple(
    s.step_id for s in STEPS if s.step_id not in OPT_IN_STEPS
)
PROBE_ID = "probe"


def uv_argv(*args: str) -> list[str]:
    """uv is installed with pip into the kernel's interpreter, so it is invoked
    as a module of that interpreter (no PATH lookup of a `uv` executable)."""
    return [sys.executable, "-m", "uv", *args]


def step_argv(
    step: Step, as_of: str, *, allow_cpu: bool = False, extra: Sequence[str] = ()
) -> list[str]:
    """The exact `uv run --no-sync ...` command line for a step. --no-sync keeps
    the environment built by `uv sync --extra cuda --frozen` (a bare `uv run`
    re-resolves and replaced the locked cu124 torch with cu130 on Kaggle)."""
    argv = uv_argv("run", "--no-sync", step.script, *step.args, "--as-of", as_of, *extra)
    if allow_cpu:
        argv.append("--allow-cpu")
    return argv


def validate_experiments(experiments: Iterable[str]) -> list[str]:
    ids = list(experiments)
    unknown = [e for e in ids if e not in STEP_BY_ID]
    if unknown:
        raise ValueError(f"unknown step id(s) {unknown}; valid: {list(STEP_BY_ID)}")
    return ids


# ---------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series_sha256(dates: Iterable, values: Iterable) -> str:
    """Same recipe as scripts/03a_select_events.py series_sha256 (date,value lines)."""
    text = "\n".join(f"{d.date().isoformat()},{v:.10g}" for d, v in zip(dates, values, strict=True))
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- cache


class CacheMismatch(RuntimeError):
    """The attached dataset does not match the committed manifests."""


def copy_dataset_into_cache(dataset_dir: Path, cache_dir: Path) -> list[Path]:
    """Copy every *.parquet found under dataset_dir (any depth) flat into cache_dir."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise CacheMismatch(f"DATASET_DIR {dataset_dir} does not exist (attach the dataset)")
    files = sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise CacheMismatch(f"no *.parquet files found under {dataset_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        target = cache_dir / f.name
        shutil.copy2(f, target)
        out.append(target)
    return out


def latest_manifest(manifests_dir: Path, pattern: str) -> Path | None:
    found = sorted(Path(manifests_dir).glob(pattern))
    return found[-1] if found else None


def cache_manifest_path(manifests_dir: Path, as_of: str) -> Path:
    return Path(manifests_dir) / f"cache-{as_of}.json"


def build_cache_files(cache_dir: Path) -> dict[str, dict]:
    """{filename: {"sha256", "bytes"}} for every *.parquet in cache_dir."""
    files = sorted(Path(cache_dir).glob("*.parquet"))
    if not files:
        raise CacheMismatch(f"no *.parquet files in {cache_dir}")
    return {f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files}


def write_cache_manifest(cache_dir: Path, manifests_dir: Path, as_of: str) -> Path:
    """Write results/manifests/cache-<as_of>.json (via tfm3lab.manifest) with
    the sha256 and size of every data/cache/*.parquet. Run it locally right
    after the fetch and commit the result: the Kaggle notebook refuses a cache
    that does not match it file by file."""
    from tfm3lab.manifest import write_manifest

    files = build_cache_files(cache_dir)
    return write_manifest(
        {"files": files, "n_files": len(files)},
        cache_manifest_path(manifests_dir, as_of),
        as_of=dt.date.fromisoformat(as_of),
    )


def verify_cache(cache_dir: Path, manifests_dir: Path, as_of: str) -> dict:
    """Verify the copied cache against the committed manifests; raise
    CacheMismatch on the first inconsistency. Returns a report with the sha256
    of every cache file.

    Checks:
    - cache manifest cache-<as_of>.json (REQUIRED): every listed file is present
      with the same sha256 and size, and the cache holds no unlisted parquet;
    - shock-events manifest: sha256 of every shock_v2_<label>_<as_of>.parquet
      close series equals manifest['data_sha256'][label];
    - fetch manifest (latest fetch-<as_of>-*.json): the set of series in
      mtg_prices.parquet equals the resolved_cards labels, the last date equals
      date_range.end, and an optional 'cache_sha256' map ({file: sha256}) is
      honoured when present.
    """
    import pandas as pd

    cache_dir, manifests_dir = Path(cache_dir), Path(manifests_dir)
    report: dict = {"files": {}, "checked": []}
    for f in sorted(cache_dir.glob("*.parquet")):
        report["files"][f.name] = sha256_file(f)

    cman = cache_manifest_path(manifests_dir, as_of)
    if not cman.is_file():
        raise CacheMismatch(
            f"missing committed cache manifest {cman}. Locally, right after the fetch, run "
            f"`uv run python -m tfm3lab.kaggle cache-manifest --as-of {as_of}`, commit the file "
            "and push GIT_REF"
        )
    listed = json.loads(cman.read_text(encoding="utf-8")).get("files") or {}
    if not listed:
        raise CacheMismatch(f"{cman.name} lists no files")
    for name, meta in listed.items():
        actual = report["files"].get(name)
        if actual is None:
            raise CacheMismatch(f"cache is missing {name} (listed in {cman.name})")
        if actual != meta["sha256"]:
            raise CacheMismatch(
                f"{name}: sha256 {actual} != {cman.name} {meta['sha256']}; the attached dataset "
                "is not the cache that was committed"
            )
        size = (cache_dir / name).stat().st_size
        if meta.get("bytes") is not None and size != meta["bytes"]:
            raise CacheMismatch(f"{name}: {size} bytes != {cman.name} {meta['bytes']}")
    extra = sorted(set(report["files"]) - set(listed))
    if extra:
        raise CacheMismatch(f"cache has files not in {cman.name}: {extra[:5]}")
    report["checked"].append(f"cache_manifest:{len(listed)}_files")
    report["cache_manifest"] = cman.name

    # shock events
    shock_manifest = manifests_dir / f"shock-events-v2-{as_of}.json"
    if not shock_manifest.is_file():
        raise CacheMismatch(f"missing committed manifest {shock_manifest}")
    shock = json.loads(shock_manifest.read_text(encoding="utf-8"))
    for label, expected in shock["data_sha256"].items():
        path = cache_dir / f"shock_v2_{label}_{as_of}.parquet"
        if not path.is_file():
            raise CacheMismatch(f"cache is missing {path.name}")
        df = pd.read_parquet(path)
        actual = series_sha256(pd.to_datetime(df["date"]), df["close"].to_numpy())
        if actual != expected:
            raise CacheMismatch(
                f"{path.name}: series sha256 {actual} != shock manifest {expected}; the attached "
                "dataset was built from different data"
            )
        report["checked"].append(f"shock:{label}")

    # fetch manifest
    fetch = latest_manifest(manifests_dir, f"fetch-{as_of}-*.json")
    if fetch is None:
        raise CacheMismatch(f"no committed fetch manifest fetch-{as_of}-*.json in {manifests_dir}")
    fm = json.loads(fetch.read_text(encoding="utf-8"))
    report["fetch_manifest"] = fetch.name
    mtg_path = cache_dir / "mtg_prices.parquet"
    if not mtg_path.is_file():
        raise CacheMismatch("cache is missing mtg_prices.parquet")
    mtg = pd.read_parquet(mtg_path, columns=["series", "date"])
    have = set(mtg["series"].unique())
    want = {c["label"] for c in fm.get("resolved_cards", [])}
    if want and have != want:
        raise CacheMismatch(
            f"mtg_prices.parquet series differ from {fetch.name}: "
            f"missing {sorted(want - have)[:5]}, extra {sorted(have - want)[:5]}"
        )
    end = fm.get("date_range", {}).get("end")
    if end and str(pd.to_datetime(mtg["date"]).max().date()) != end:
        raise CacheMismatch(
            f"mtg_prices.parquet ends {pd.to_datetime(mtg['date']).max().date()}, "
            f"fetch manifest says {end}"
        )
    report["checked"].append("fetch:mtg_series")
    for name, expected in (fm.get("cache_sha256") or {}).items():
        actual = report["files"].get(name)
        if actual != expected:
            raise CacheMismatch(f"{name}: sha256 {actual} != fetch manifest {expected}")
        report["checked"].append(f"fetch:sha:{name}")
    for required in ("market_prices.parquet", "cpi_yoy.parquet"):
        if required not in report["files"]:
            raise CacheMismatch(f"cache is missing {required}")
    return report


# ---------------------------------------------------------------- resume


class ResumeCommitMismatch(RuntimeError):
    """The restored run_status was produced on a different commit."""


def _commit_mismatch_message(previous: str, current: str) -> str:
    return (
        f"The previous session ran commit {previous} but this clone is at {current}. "
        f"Stop: set GIT_REF = {previous!r} (the previous commit sha) and run again, so the "
        "02b run directories resume on identical code. Results of two commits must not be mixed."
    )


def check_resume_commit(previous_commit: str | None, current_commit: str | None) -> None:
    if previous_commit and current_commit and previous_commit != current_commit:
        raise ResumeCommitMismatch(_commit_mismatch_message(previous_commit, current_commit))


def restore_previous(
    previous_dir: Path | None, results_dir: Path, current_commit: str | None = None
) -> list[str]:
    """Copy a previous session's results (run dirs, partial outputs, status)
    into results_dir without overwriting newer files. Returns restored names.
    When current_commit is given, the restored run_status commit sha must equal
    it (checked before anything is copied): ResumeCommitMismatch otherwise."""
    if not previous_dir:
        return []
    previous_dir = Path(previous_dir)
    if not previous_dir.is_dir():
        raise FileNotFoundError(f"PREVIOUS_OUTPUT_DIR {previous_dir} does not exist")
    candidates = [previous_dir / "tfm3lab-data" / "results", previous_dir / "results", previous_dir]
    src = next(
        (c for c in candidates if (c / "benchmark").is_dir() or (c / "run_status.json").is_file()),
        None,
    )
    if src is None:
        raise FileNotFoundError(
            f"no tfm3lab-data/results (benchmark/ or run_status.json) under {previous_dir}"
        )
    prev_status = src / "run_status.json"
    if current_commit and prev_status.is_file():
        try:
            prev_sha = json.loads(prev_status.read_text(encoding="utf-8")).get("commit_sha")
        except json.JSONDecodeError:
            prev_sha = None
        check_resume_commit(prev_sha, current_commit)
    restored = []
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = results_dir / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        restored.append(str(rel))
    return restored


# ---------------------------------------------------------------- running


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict | None = None,
    echo: Callable[[str], None] | None = None,
) -> tuple[int, float]:
    """Run argv, streaming combined output to `echo` (default print) and a log
    file. Returns (exit_code, elapsed_seconds); an unlaunchable command is 127."""
    echo = echo or (lambda line: print(line, end="", flush=True))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n")
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=merged,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            log.write(f"launch failed: {exc}\n")
            echo(f"launch failed: {exc}\n")
            return 127, time.monotonic() - start
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            echo(line)
        code = proc.wait()
    return code, time.monotonic() - start


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def format_status_table(steps: dict[str, dict]) -> str:
    rows = [f"{'step':<24}{'status':<9}{'exit':>5}{'elapsed_min':>13}"]
    for sid in STEP_BY_ID:
        rec = steps.get(sid)
        if rec is None:
            continue
        code = "" if rec.get("exit_code") is None else str(rec["exit_code"])
        el = rec.get("elapsed_s")
        rows.append(
            f"{sid:<24}{rec['status']:<9}{code:>5}{'' if el is None else f'{el / 60:.1f}':>13}"
        )
    return "\n".join(rows)


class Session:
    """Holds the run parameters and the persisted status table."""

    def __init__(
        self,
        *,
        as_of: str,
        repo_dir: Path,
        data_root: Path,
        experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
        resume: bool = True,
        allow_cpu: bool = False,
        quota_hours: float | None = None,
        commit_sha: str | None = None,
        git_ref: str | None = None,
        results_dir: Path | None = None,
    ):
        self.as_of = as_of
        self.repo_dir = Path(repo_dir)
        self.data_root = Path(data_root)
        # results_dir may differ from data_root/results (a symlink target on Kaggle)
        self.results_dir = Path(results_dir) if results_dir else self.data_root / "results"
        self.experiments = validate_experiments(experiments)
        self.resume = resume
        self.allow_cpu = allow_cpu
        self.quota_hours = quota_hours
        self.commit_sha = commit_sha
        self.git_ref = git_ref
        self.status_path = self.results_dir / "run_status.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict:
        state: dict = {}
        if self.status_path.is_file():
            try:
                state = json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        state.setdefault("steps", {})
        check_resume_commit(state.get("commit_sha"), self.commit_sha)
        state["as_of"] = self.as_of
        state["commit_sha"] = self.commit_sha or state.get("commit_sha")
        state["git_ref"] = self.git_ref
        return state

    def save(self) -> Path:
        self.state["updated_utc"] = _utcnow()
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.status_path)
        return self.status_path

    def attach_preflight(self) -> None:
        """Copy the resolved checkpoint shas and environment from preflight.json."""
        path = self.results_dir / "preflight.json"
        if path.is_file():
            rep = json.loads(path.read_text(encoding="utf-8"))
            self.state["checkpoints"] = rep.get("checkpoints")
            self.state["environment"] = rep.get("environment")
            self.save()

    def probe_extra_args(self) -> list[str]:
        planned = [e for e in self.experiments if e != PROBE_ID]
        extra = ["--steps", ",".join(planned)] if planned else []
        if self.quota_hours is not None:
            extra += ["--quota-hours", str(self.quota_hours)]
        return extra

    def should_skip(self, step_id: str) -> str | None:
        """Reason to skip, or None. The probe is always re-run (it is cheap and
        quota changes between sessions)."""
        if step_id not in self.experiments:
            return "not in EXPERIMENTS"
        prev = self.state["steps"].get(step_id)
        if (
            self.resume
            and step_id != PROBE_ID
            and prev
            and prev.get("status") == "ok"
            and prev.get("commit_sha") == self.commit_sha
        ):
            return "already completed on this commit in a restored session (RESUME=True)"
        return None

    def run(self, step_id: str, *, echo: Callable[[str], None] | None = None) -> dict:
        step = STEP_BY_ID[step_id]
        reason = self.should_skip(step_id)
        if reason:
            print(f"[{step_id}] skipped: {reason}")
            if step_id not in self.state["steps"]:
                self.state["steps"][step_id] = {"status": "skipped", "reason": reason}
                self.save()
            return self.state["steps"][step_id]
        extra = self.probe_extra_args() if step_id == PROBE_ID else ()
        argv = step_argv(step, self.as_of, allow_cpu=self.allow_cpu, extra=extra)
        print(f"[{step_id}] {' '.join(argv)}")
        started = _utcnow()
        code, elapsed = run_command(
            argv,
            cwd=self.repo_dir,
            log_path=self.results_dir / "logs" / f"{step_id}.log",
            echo=echo,
        )
        rec = {
            "status": "ok" if code == 0 else "failed",
            "exit_code": code,
            "elapsed_s": round(elapsed, 1),
            "started_utc": started,
            "commit_sha": self.commit_sha,
            "argv": argv,
        }
        self.state["steps"][step_id] = rec
        self.save()
        print(f"[{step_id}] {rec['status']} (exit {code}, {elapsed / 60:.1f} min)")
        print(format_status_table(self.state["steps"]))
        return rec


# ---------------------------------------------------------------- packing


def _add_tree(tar: tarfile.TarFile, root: Path, arc_root: str, skip: Callable[[Path], bool]) -> int:
    n = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or skip(path.relative_to(root)):
            continue
        tar.add(path, arcname=f"{arc_root}/{path.relative_to(root).as_posix()}", recursive=False)
        n += 1
    return n


def pack_results(
    results_dir: Path,
    out_dir: Path,
    as_of: str,
    state: dict | None = None,
    *,
    include_heavy: bool = True,
) -> dict:
    """Write the light and heavy archives. The light archive holds everything
    under results_dir except benchmark/ (and includes run_status.json); the
    heavy archive holds only results/benchmark run dirs (actual prices: local
    use only, never committed)."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        (results_dir / "run_status.json").write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )
    light = out_dir / f"tfm3lab-results-{as_of}.tar.gz"
    heavy = out_dir / f"tfm3lab-heavy-{as_of}.tar.gz"
    info: dict = {}
    with tarfile.open(light, "w:gz") as tar:
        n = _add_tree(tar, results_dir, "results", lambda rel: rel.parts[0] == "benchmark")
    info["light"] = {"path": str(light), "files": n}
    bench = results_dir / "benchmark"
    if include_heavy and bench.is_dir() and any(p.is_file() for p in bench.rglob("*")):
        with tarfile.open(heavy, "w:gz") as tar:
            n = _add_tree(tar, bench, "results/benchmark", lambda rel: False)
        info["heavy"] = {"path": str(heavy), "files": n}
    else:
        info["heavy"] = None
    for key in ("light", "heavy"):
        if info[key]:
            p = Path(info[key]["path"])
            info[key]["bytes"] = p.stat().st_size
            info[key]["sha256"] = sha256_file(p)
    return info


# ---------------------------------------------------------------- environment


def _mem_gb() -> dict:
    out: dict = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[key] = round(int(rest.split()[0]) / 1024 / 1024, 2)
    except (OSError, ValueError):
        pass
    return out


def preflight_report(data_root: Path, *, allow_cpu: bool = False) -> dict:
    """GPU, CUDA, HF token, gated access to both checkpoints (resolved shas),
    free disk and RAM. Raises SystemExit (via tfm3lab.preflight) on failure."""
    from tfm3lab import config, preflight

    preflight.require_gpu(allow_cpu=allow_cpu)
    preflight.require_hf_token()
    env = preflight.run_environment()
    shas = {
        "timesfm3": {
            "repo_id": config.CHECKPOINT_ID,
            "revision_requested": config.CHECKPOINT_REVISION,
            "revision_resolved": preflight.require_checkpoint_access(
                config.CHECKPOINT_ID, config.CHECKPOINT_REVISION
            ),
        },
        "timesfm2.5": {
            "repo_id": config.CHECKPOINT_ID_2P5,
            "revision_requested": config.CHECKPOINT_REVISION_2P5,
            "revision_resolved": preflight.require_checkpoint_access(
                config.CHECKPOINT_ID_2P5, config.CHECKPOINT_REVISION_2P5
            ),
        },
    }
    usage = shutil.disk_usage(data_root)
    return {
        "environment": env,
        "hf_token_present": True,
        "checkpoints": shas,
        "disk_free_gb": round(usage.free / 1e9, 1),
        "ram_gb": _mem_gb(),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------- projection

LARGE = float("inf")


@dataclasses.dataclass(frozen=True)
class StepWork:
    """Work a step will issue: forecast requests as (context_len, n_series)."""

    step_id: str
    adapter: str
    items: tuple[tuple[int, int], ...]
    note: str = ""
    # (context, batch, n_calls): priced per call, see per_call_seconds
    cells: tuple[tuple[int, int, int], ...] = ()


def _interp(vals: dict[int, float], context: int) -> tuple[float, str]:
    ctxs = sorted(vals)
    if context in vals:
        return vals[context], "measured"
    if context < ctxs[0]:
        return vals[ctxs[0]], f"clamped to measured context {ctxs[0]}"
    if context > ctxs[-1]:
        return vals[ctxs[-1]] * context / ctxs[-1], f"scaled linearly from context {ctxs[-1]}"
    lo = max(c for c in ctxs if c < context)
    hi = min(c for c in ctxs if c > context)
    frac = (context - lo) / (hi - lo)
    return vals[lo] + frac * (vals[hi] - vals[lo]), f"interpolated {lo}..{hi}"


def per_call_seconds(measurements: Sequence[dict], adapter: str, context: int, batch: int):
    """Seconds for one forecast_batch call of `batch` series at `context`. Uses
    the points measured with this same batch (interpolated over context). If
    that batch was not measured (or failed), falls back to the largest measured
    batch's per-forecast time x batch and labels it 'lower bound'."""
    by_batch: dict[int, dict[int, float]] = {}
    for m in measurements:
        if m.get("adapter") == adapter and m.get("status", "ok") == "ok":
            by_batch.setdefault(int(m["batch"]), {})[int(m["context"])] = float(m["median_s"])
    if batch in by_batch:
        sec, basis = _interp(by_batch[batch], context)
        return sec, f"{basis} at batch {batch}"
    spf, basis = per_forecast_seconds(measurements, adapter, context)
    return spf * batch, f"lower bound (batch {batch} not measured; {basis})"


def per_forecast_seconds(measurements: Sequence[dict], adapter: str, context: int):
    """Seconds per single-series forecast at `context`, from the OK measurements
    of `adapter`, using the largest measured batch at each context. Linear
    interpolation between measured contexts; below the smallest the smallest
    measured value is used (conservative); above the largest the value scales
    linearly with context (the true growth is at least linear here, so this may
    under-estimate: flagged). Returns (seconds, basis)."""
    by_ctx: dict[int, tuple[int, float]] = {}
    for m in measurements:
        if m.get("adapter") != adapter or m.get("status", "ok") != "ok":
            continue
        c, b = int(m["context"]), int(m["batch"])
        if c not in by_ctx or b > by_ctx[c][0]:
            by_ctx[c] = (b, float(m["median_s"]) / b)
    if not by_ctx:
        msgs = [
            str(m["message"])
            for m in measurements
            if m.get("adapter") == adapter and m.get("message")
        ]
        why = f": {msgs[0]}" if msgs else ""
        raise ValueError(f"no usable measurement for adapter {adapter!r}{why}")
    return _interp({c: by_ctx[c][1] for c in by_ctx}, context)


def _not_projected(work: StepWork, load_seconds: dict[str, float], exc: Exception) -> dict:
    return {
        "step_id": work.step_id,
        "adapter": work.adapter,
        "n_forecasts": 0,
        "compute_s": 0.0,
        "load_s": float(load_seconds.get(work.adapter, 0.0)),
        "projected_s": 0.0,
        "basis": f"not projected (adapter failed: {exc})",
        "note": work.note,
        "projected": False,
    }


def project_step(
    work: StepWork, measurements: Sequence[dict], load_seconds: dict[str, float]
) -> dict:
    total = 0.0
    bases: set[str] = set()
    n_fc = 0
    for context, n in work.items:
        try:
            spf, basis = per_forecast_seconds(measurements, work.adapter, context)
        except ValueError as exc:
            return _not_projected(work, load_seconds, exc)
        total += spf * n
        n_fc += n
        bases.add(basis)
    for context, batch, calls in work.cells:
        try:
            sec, basis = per_call_seconds(measurements, work.adapter, context, batch)
        except ValueError as exc:
            return _not_projected(work, load_seconds, exc)
        total += sec * calls
        n_fc += batch * calls
        bases.add(basis)
    load = float(load_seconds.get(work.adapter, 0.0))
    return {
        "step_id": work.step_id,
        "adapter": work.adapter,
        "n_forecasts": n_fc,
        "compute_s": total,
        "load_s": load,
        "projected_s": total + load,
        "basis": "; ".join(sorted(bases)),
        "note": work.note,
    }


def project_plan(
    works: Sequence[StepWork],
    measurements: Sequence[dict],
    load_seconds: dict[str, float],
    quota_hours: float | None,
) -> list[dict]:
    """Per-step projections in order, with the cumulative total and, when a
    quota is given, whether the cumulative total still fits inside it."""
    rows = []
    cum = 0.0
    quota_s = None if quota_hours is None else quota_hours * 3600.0
    for w in works:
        row = project_step(w, measurements, load_seconds)
        cum += row["projected_s"]
        row["cumulative_s"] = cum
        row["quota_s"] = quota_s
        row["fits_quota"] = None if quota_s is None else cum <= quota_s
        rows.append(row)
    return rows


def work_from_dry_run(step_id: str, adapter: str, report: dict) -> StepWork:
    """02b dry-run report -> work. Every combo issues n_origins x card_pool
    series-forecasts (multivariate modes are priced per series as well)."""
    per_combo = int(report["n_origins"]) * int(report["card_pool_size"])
    items = tuple((int(c["context_len"]), per_combo) for c in report["combos"])
    note = (
        f"{len(items)} combos x {report['n_origins']} origins x {report['card_pool_size']} cards; "
        "CPU baselines and I/O not included"
    )
    return StepWork(step_id, adapter, items, note)


def work_from_shock(
    step_id: str,
    adapter: str,
    *,
    n_events: int,
    context_len: int,
    origins_per_event: int,
    n_perturbation_runs: int,
    panel_series: int,
) -> StepWork:
    per_event = origins_per_event * (1 + n_perturbation_runs + panel_series)
    return StepWork(
        step_id,
        adapter,
        ((context_len, n_events * per_event),),
        f"{n_events} events x {origins_per_event} origins x "
        f"(1 + {n_perturbation_runs} perturbation runs + {panel_series} panel series); upper bound",
    )


def work_from_covariates(
    step_id: str,
    adapter: str,
    *,
    context_len: int,
    market_origins: Sequence[int],
    mtg_origins: Sequence[int],
) -> StepWork:
    n = sum(market_origins) * 3 + sum(mtg_origins) * 4  # no_cov + legit + placebo (+ leaked, MTG)
    return StepWork(
        step_id,
        adapter,
        ((context_len, n),),
        "covariate arms priced like plain forecasts (lower bound; XReg adds time)",
    )


def work_from_latency(
    step_id: str, adapter: str, cells: Sequence[tuple[int, int, int]], repeats: int, warmup: int
) -> StepWork:
    cells_t = tuple((c, b, repeats + warmup) for b, c, _h in cells)
    return StepWork(
        step_id, adapter, (), "model calls only; CPU naive/ETS timing not included", cells_t
    )


def format_plan(rows: Sequence[dict]) -> str:
    head = f"{'step':<24}{'minutes':>10}{'cum_min':>10}  fits  basis"
    lines = [head]
    for r in rows:
        fits = "-" if r["fits_quota"] is None else ("yes" if r["fits_quota"] else "NO")
        lines.append(
            f"{r['step_id']:<24}{r['projected_s'] / 60:>10.1f}{r['cumulative_s'] / 60:>10.1f}"
            f"  {fits:<4}  {r['basis']}"
        )
    if rows:
        total_h = rows[-1]["cumulative_s"] / 3600
        q = rows[-1]["quota_s"]
        tail = f"projected total {total_h:.2f} h"
        if q is not None:
            gap = (rows[-1]["cumulative_s"] - q) / 3600
            tail += f" vs quota {q / 3600:.2f} h: " + (
                "fits" if gap <= 0 else f"EXCEEDS by {gap:.2f} h"
            )
        lines.append(tail)
    return "\n".join(lines)


def measure_points(
    forecaster,
    series: Sequence,
    adapter: str,
    points: Sequence[tuple[int, int, int]],
    *,
    repeats: int,
    warmup: int,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict]:
    """Time forecast_batch at (context, horizon, batch) points on real cached
    contexts, with warmup and `sync` around each timed call. An OOM or other
    failure at a point is recorded and the remaining points continue."""
    from tfm3lab import latency
    from tfm3lab.model import forecast_batch

    rows: list[dict] = []
    for context, horizon, batch in points:
        row = {
            "adapter": adapter,
            "context": context,
            "horizon": horizon,
            "batch": batch,
            "repeats": repeats,
            "warmup": warmup,
        }
        try:
            ctxs, source = latency.make_contexts(series, batch, context, allow_synthetic=False)
            ids = [str(i) for i in range(batch)]
            stats = latency.time_cell(
                lambda ctxs=ctxs, ids=ids, horizon=horizon: forecast_batch(
                    forecaster, ctxs, horizon, ts_ids=ids
                ),
                n_forecasts=batch,
                repeats=repeats,
                warmup=warmup,
                sync=sync,
                clock=clock,
            )
            rows.append({**row, **stats, "status": "ok", "data_source": source})
        except Exception as exc:  # noqa: BLE001 - a failed point must not end the probe
            status = "oom" if latency.is_oom(exc) else "error"
            rows.append({**row, "status": status, "message": f"{type(exc).__name__}: {exc}"})
    return rows


# ------------------------------------------------- code snapshot (no GitHub push)

INPUT_ROOT = Path("/kaggle/input")
SNAPSHOT_PREFIX = "tfm3lab-code-"
COMMIT_FILE = "COMMIT_SHA"


def _walk_dirs(root: Path, max_depth: int):
    """Yield (directory, depth) below root, breadth-first, never following symlinks loops."""
    queue = [(root, 0)]
    while queue:
        cur, depth = queue.pop(0)
        yield cur, depth
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in cur.iterdir() if p.is_dir())
        except OSError:
            continue
        queue.extend((c, depth + 1) for c in children)


def find_input_dir(slug: str, root: Path = INPUT_ROOT, max_depth: int = 6) -> Path:
    """Directory of an attached Kaggle dataset, found by its slug below `root`.

    Handles /kaggle/input/<slug> and nested mounts such as
    /kaggle/input/datasets/<owner>/<slug>. The match is on the directory name
    (case-insensitive)."""
    slug = slug.strip().strip("/").lower()
    if not slug:
        raise ValueError("dataset slug is empty")
    for cur, _ in _walk_dirs(Path(root), max_depth):
        if cur != Path(root) and cur.name.lower() == slug:
            return cur
    seen = [p.name for p, d in _walk_dirs(Path(root), 2) if d >= 1]
    raise FileNotFoundError(
        f"no attached dataset {slug!r} under {root}; attach it to the notebook "
        f"(found: {', '.join(seen) or 'nothing'})"
    )


def _is_code_dir(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "tfm3lab").is_dir()


def _snapshot_candidates(base: Path, max_depth: int) -> list[Path]:
    """Extracted snapshot directories and snapshot archives below `base`."""
    found: list[Path] = []
    for cur, _depth in _walk_dirs(base, max_depth):
        if any(cur.is_relative_to(f) for f in found if f.is_dir()):
            continue
        if _is_code_dir(cur):
            found.append(cur)
            continue
        try:
            found.extend(sorted(cur.glob(f"{SNAPSHOT_PREFIX}*.tar.gz")))
        except OSError:
            continue
    return found


def _candidate_matches(path: Path, sha12: str) -> bool:
    if sha12 in path.name.lower():
        return True
    commit = path / COMMIT_FILE if path.is_dir() else None
    if commit is not None and commit.is_file():
        first = commit.read_text(encoding="utf-8").strip().splitlines()[:1]
        return bool(first) and first[0].strip().lower().startswith(sha12)
    return False


def find_code_snapshot(
    slug: str | None = None, root: Path = INPUT_ROOT, max_depth: int = 6
) -> Path:
    """The code snapshot below `root`: an extracted directory (pyproject.toml and
    src/tfm3lab) or a tfm3lab-code-<sha>.tar.gz archive.

    With a `slug` the dataset of that name must be attached (FileNotFoundError
    otherwise, never a fallback to the whole tree) and is searched alone. When
    several snapshots are found, the one whose sha12 (from a `tfm3lab-code-<sha12>`
    slug) matches is chosen; if none or several do, fail listing them."""
    root = Path(root)
    base = find_input_dir(slug, root, max_depth) if slug else root
    found = _snapshot_candidates(base, max_depth)
    if not found:
        raise FileNotFoundError(
            f"no code snapshot under {base}: attach the dataset {slug!r} holding "
            f"{SNAPSHOT_PREFIX}<sha12>.tar.gz (or its extracted files)"
        )
    if len(found) == 1:
        return found[0]
    sha12 = ""
    if slug and slug.strip().lower().startswith(SNAPSHOT_PREFIX):
        sha12 = slug.strip().lower()[len(SNAPSHOT_PREFIX) :]
    matches = [p for p in found if sha12 and _candidate_matches(p, sha12)]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(
        f"{len(found)} code snapshots under {base}, "
        f"{'none or several match' if sha12 else 'no sha12 in the slug to choose'} "
        f"{sha12!r}: {', '.join(str(p) for p in found)}"
    )


def _make_writable(path: Path) -> None:
    for cur, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            p = Path(cur) / name
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | 0o200)
    path.chmod(path.stat().st_mode | 0o200)


def materialize_code_snapshot(source: Path, dest: Path) -> Path:
    """Copy (directory) or extract (tar.gz) the snapshot into `dest`, replacing it.
    The result is writable even when `source` is read-only (/kaggle/input)."""
    source, dest = Path(source), Path(dest)
    if dest.exists():
        _make_writable(dest)
        shutil.rmtree(dest)
    if source.is_dir():
        shutil.copytree(source, dest, copy_function=shutil.copyfile)
        _make_writable(dest)
    else:
        dest.mkdir(parents=True)
        with tarfile.open(source, "r:*") as tar:
            tar.extractall(dest, filter="data")
        _make_writable(dest)
    if not _is_code_dir(dest):
        raise RuntimeError(f"snapshot {source} has no pyproject.toml and src/tfm3lab")
    return dest


def read_snapshot_commit(snapshot_dir: Path) -> tuple[str, bool]:
    """(commit sha, declared clean) from the snapshot's COMMIT_SHA file: the first
    line is the sha, a later line `clean` declares a clean worktree."""
    path = Path(snapshot_dir) / COMMIT_FILE
    if not path.is_file():
        raise RuntimeError(f"{path} is missing: the snapshot was not made by code-snapshot")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sha = lines[0] if lines else ""
    if len(sha) < 12 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise RuntimeError(f"{path} does not start with a commit sha")
    return sha, "clean" in lines[1:]


def snapshot_environ(sha: str, clean: bool) -> dict[str, str]:
    """Environment that tfm3lab.manifest reads when the tree has no .git."""
    env = {"TFM3LAB_GIT_SHA": sha}
    if clean:
        env["TFM3LAB_GIT_DIRTY"] = "0"
    return env


def create_code_snapshot(out_dir: Path, repo: Path | None = None) -> tuple[Path, list[str]]:
    """git archive of HEAD plus a COMMIT_SHA file -> out_dir/tfm3lab-code-<sha12>.tar.gz.

    Refuses if tracked files are modified or staged. Untracked files are never
    included. Returns (archive path, sorted member names)."""
    import gzip
    import io

    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout

    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("worktree is dirty: commit or stash tracked changes first")
    sha = git("rev-parse", "HEAD").decode().strip()
    raw = git("archive", "--format=tar", "HEAD")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{SNAPSHOT_PREFIX}{sha[:12]}.tar.gz"
    names: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src:
        members = [m for m in src.getmembers() if m.name != COMMIT_FILE]
        mtime = int(members[0].mtime) if members else 0
        with (
            gzip.GzipFile(archive, "wb", mtime=0) as gz,
            tarfile.open(fileobj=gz, mode="w") as dst,
        ):
            for m in members:
                dst.addfile(m, src.extractfile(m) if m.isfile() else None)
                names.append(m.name)
            body = f"{sha}\nclean\n".encode()
            info = tarfile.TarInfo(COMMIT_FILE)
            info.size, info.mtime, info.mode = len(body), mtime, 0o644
            dst.addfile(info, io.BytesIO(body))
            names.append(COMMIT_FILE)
    return archive, sorted(names)


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m tfm3lab.kaggle preflight [--allow-cpu]` (Kaggle, under `uv run`):
    prints the report and writes results/preflight.json.
    `python -m tfm3lab.kaggle cache-manifest --as-of YYYY-MM-DD` (local, right after the
    fetch): writes results/manifests/cache-<as_of>.json with the sha256 and size of every
    data/cache/*.parquet; commit it.
    `python -m tfm3lab.kaggle code-snapshot --out DIR` (local, clean worktree): writes
    DIR/tfm3lab-code-<sha12>.tar.gz (git archive of HEAD plus COMMIT_SHA) and prints its files."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("preflight", "cache-manifest", "code-snapshot"))
    parser.add_argument("--out", type=Path, help="code-snapshot: output directory")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--as-of", type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    args = parser.parse_args(argv)

    if args.command == "code-snapshot":
        if args.out is None:
            parser.error("code-snapshot requires --out")
        try:
            archive, names = create_code_snapshot(args.out)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"code-snapshot: {exc}", file=sys.stderr)
            return 1
        print("\n".join(names))
        print(f"wrote {archive} ({len(names)} files)")
        return 0

    from tfm3lab import config

    if args.command == "cache-manifest":
        if args.as_of is None:
            parser.error("cache-manifest requires --as-of")
        out = write_cache_manifest(
            config.CACHE_DIR, config.RESULTS_DIR / "manifests", args.as_of.isoformat()
        )
        print(f"wrote {out}")
        return 0

    report = preflight_report(config.DATA_ROOT, allow_cpu=args.allow_cpu)
    out = config.RESULTS_DIR / "preflight.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
