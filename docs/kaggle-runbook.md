# Kaggle runbook (Phase 2, Preregistration v2)

This runs every GPU step of Preregistration v2 in `notebooks/run_on_kaggle.ipynb`
and returns one results archive. The notebook is parameterized, idempotent and
resumable. `notebooks/run_on_colab.ipynb` is **not updated for v2**; use the
Kaggle notebook.

Nothing here states how long a step takes. The runtime probe measures
throughput on the GPU you were given and projects the minutes from it.

## Prerequisites

1. **Commit and push the Git ref yourself.** The agent never pushes. The
   new files of this package (`notebooks/run_on_kaggle.ipynb`,
   `docs/kaggle-runbook.md`, `scripts/00b_runtime_probe.py`,
   `src/tfm3lab/kaggle.py`, the two tests) and the cache manifest of step 4
   must be committed **and pushed** so that they are inside `GIT_REF`; the
   notebook runs whatever the pushed ref contains. Set `GIT_REF` to the
   branch for session 1; the notebook clones `REPO_URL`, checks out that ref
   detached and prints the commit sha (also recorded in `run_status.json`).
   **For every later session set `GIT_REF` to that commit sha**, not the
   branch, so a push in between cannot change the code under a resumed run.
2. **Accept both gated licences on Hugging Face**, with the account that owns
   your token:
   - <https://huggingface.co/google/timesfm-3.0-pytorch>
   - <https://huggingface.co/google/timesfm-2.5-200m-pytorch>
3. **`HF_TOKEN` secret.** In the notebook: Add-ons, Secrets, add `HF_TOKEN`
   (a read token) and attach it to the notebook. The notebook stops with a
   clear error if it is missing.
4. **Build the cache locally and upload it as a private Kaggle dataset.**
   Card prices are not redistributed, so the dataset must be **Private**.
   Build `data/cache` locally with the exact fetch command from the
   preregistration (run it from the repository root):

   ```
   uv run scripts/01_fetch_data.py --as-of 2026-09-04 \
       --cards showcase --cards configs/benchmark_pool_v2.csv
   ```

   Then, **locally and right after the fetch** (and after
   `scripts/03a_select_events.py` has written the shock series), write the
   cache manifest and commit and push it:

   ```
   uv run python -m tfm3lab.kaggle cache-manifest --as-of 2026-09-04
   ```

   This writes `results/manifests/cache-2026-09-04.json` (sha256 and size of
   every `data/cache/*.parquet`). The notebook **requires** it and stops if it
   is missing or if any file of the attached dataset differs or is extra.
   Re-run it whenever the cache changes. The committed fetch manifest
   `results/manifests/fetch-2026-09-04-*.json` records the card sources used.
   The shock-event series come from `scripts/03a_select_events.py` (manifest
   `results/manifests/shock-events-v2-2026-09-04.json`). Upload the contents of
   `data/cache` (`mtg_prices.parquet`, `market_prices.parquet`,
   `cpi_yoy.parquet`, `shock_v2_*_2026-09-04.parquet`) as a new private
   dataset (the upload must match the manifest byte for byte) and attach it
   to the notebook. Files may sit in subfolders; the
   notebook finds every `*.parquet` and copies it flat into the cache.
5. In the notebook settings: Accelerator = a GPU, **Internet = On** (Hugging
   Face, and TCGCSV for step 05).

## Choosing a GPU

Pick any CUDA GPU with enough memory for the 200M-class models; the notebook
runs on one device. A larger or faster GPU shortens the run, a smaller one
may run out of memory in the larger batches (see troubleshooting). Do not
plan from rules of thumb: run the probe on the GPU you select and read its
projection. If the quota is tight, switch accelerators only between sessions,
because a different GPU means a different projection.

## Parameters (first code cell)

| Name | Meaning |
| --- | --- |
| `REPO_URL` | Repository to clone. |
| `GIT_REF` | Pushed branch, tag or commit sha (required). |
| `AS_OF` | Data freeze date, `2026-09-04`; must match the committed manifests. |
| `DATASET_DIR` | Mount path of the attached private cache dataset. |
| `PREVIOUS_OUTPUT_DIR` | Optional: a prior version's output, attached as input, to resume. |
| `EXPERIMENTS` | Ordered step ids; remove ids to skip steps. |
| `RESUME` | Skip steps already completed in a restored session (02b always resumes its run directories). |
| `ALLOW_CPU` | Smoke tests only. Never use for real results. |
| `QUOTA_HOURS` | Remaining weekly GPU quota; the probe compares its projection with it. |
| `PACK_HEAVY` | Also write the heavy archive. |

Step ids, in run order (the primary preregistered claim comes first after the
probe): `probe`, `02b_preregistered_v2`, `02b_timesfm2p5`, `02b_showcase`,
`02b_ffill0`, `02b_ffill1`, `03_timesfm3`, `03_timesfm2p5`, `05_covariates`,
`09_latency_timesfm3`, `09_latency_timesfm2p5`.

## What the notebook does

1. Installs `uv`, clones, pins `GIT_REF`, prints the commit sha, runs
   `uv sync --extra cuda --frozen`.
2. Loads `HF_TOKEN`. **Storage layout:** the clone, its `.venv`, the uv cache
   and the data cache all live on local disk under `/kaggle/tmp` (not saved,
   so the version output stays small); `TFM3LAB_DATA_ROOT=/kaggle/tmp/tfm3lab-data`.
   Its `results` entry is a symlink to the real directory
   **`/kaggle/working/tfm3lab-data/results`**, which is local disk and the
   path that is **saved** with the committed version: all results, the 02b run
   directories (`results/benchmark`), `run_status.json`, logs, the probe, and
   the two archives written to `/kaggle/working`.
3. If `PREVIOUS_OUTPUT_DIR` is set, restores its `results/benchmark` run
   directories, partial outputs and `run_status.json`. It first compares the
   commit sha recorded there with the sha of the current clone; on a mismatch
   it copies nothing and stops, telling you to set `GIT_REF` to the previous
   commit sha. Steps recorded as `ok` on another commit are never skipped.
4. Copies the dataset into the cache (local disk, outside `/kaggle/working`)
   and verifies it against the committed manifests; a mismatch stops the run
   before any GPU time is spent. The check **requires**
   `cache-2026-09-04.json` and compares the sha256 of every file with it,
   recomputes the shock-series hashes from `shock-events-v2-*.json`, and
   compares the card set and last date of `mtg_prices.parquet` with the fetch
   manifest.
5. Preflight: GPU name, CUDA, token, gated access to both checkpoints, the
   resolved checkpoint commit shas, free disk and RAM.
6. Runs each step in its own cell through `uv run`. Exit status and elapsed
   time go into a status table (`results/run_status.json`, per-step logs in
   `results/logs/`). A failing step does not block later steps.
7. Packs the archives.

## Reading the probe and planning against the weekly quota

The probe projects every step listed in `EXPERIMENTS` (except itself). To
plan before committing to a full run, run the notebook with the full
`EXPERIMENTS` list but stop after the probe cell, or run the probe in a short
first session. Set `QUOTA_HOURS` to what the Kaggle page shows as remaining.

The probe loads each adapter and times `forecast_batch` on real cached
contexts with warmup and CUDA sync at every context of the step-09 latency
grid (the largest included) and batch sizes 1, 8, 32 and 128 (at the longest
grid horizon). 02b, 03 and 05 are priced per forecast from the largest
measured batch; each step-09 latency cell is priced per call from the point
measured with the same batch. A cell whose batch could not be measured (for
example after an OOM) is projected from a smaller batch and labelled
`lower bound` in the basis column. The probe
prints one line per step: projected minutes, cumulative minutes, whether the
running total still fits the quota, and the basis (measured, interpolated or
scaled from the measured contexts). Work sizes come from the dry-run of each
02b config, the number of events and origins for 03, series x arms x origins
for 05, and the latency grid for 09. It writes `results/runtime_probe.parquet`
and a manifest.

Limits of the projection: it prices model time only (CPU baselines, disk I/O,
the covariate overhead of step 05 and the CPU timings of step 09 are not
included), and contexts or batches outside the measured points are clamped or
scaled. Treat it as a planning aid; leave margin. If the cumulative total
exceeds the quota, drop the lowest-priority steps from `EXPERIMENTS` for this
session (the order already puts the primary claim first) and run the rest in
a later session.

## Running headless and resuming

Use **Save & Run All (Commit)** so the run continues without the browser and
`/kaggle/working` is saved as the version output. The 02b steps write a crash-
safe per-cell store under `results/benchmark/`; they resume with `--resume`.

To continue in a second session (after a timeout, quota exhaustion or a
failed step): open the notebook again, attach the previous version's output
as an input, set `PREVIOUS_OUTPUT_DIR` to its mount path (shown in the right
sidebar), **set `GIT_REF` to the commit sha printed in session 1**, keep
`RESUME = True`, and run again. Completed steps are skipped, interrupted 02b
runs continue from the last committed cell, and failed steps are retried. The
restore cell checks that the previous session ran the same commit and stops
with a message naming the sha to use otherwise; it never overwrites the
recorded commit sha.

## What to bring back

Download from the notebook's Output tab:

- `tfm3lab-results-2026-09-04.tar.gz` (the **light archive**): every file
  under `results/` except `benchmark/`, the manifests, the logs, the probe and
  `run_status.json`. This is what the lead needs.
- `tfm3lab-heavy-2026-09-04.tar.gz` (optional, **local use only**): the
  `results/benchmark` run directories. It contains actual card prices and is
  never committed. It duplicates the run directories that are also still in
  the version output, so mind the output size limit; set `PACK_HEAVY = False`
  if you do not need it.

The console prints the size and sha256 of each archive.

If the session times out before the pack cell runs, there are no archives.
Download the version output's `tfm3lab-data/results` directory instead (it
holds the same files; `results/benchmark` is the heavy part) and use it in
place of the archives below, or resume in another session.

## Unpacking and offline follow-up

From the repository root, on a machine with the same commit checked out:

```
tar -xzf tfm3lab-results-2026-09-04.tar.gz        # creates results/...
tar -xzf tfm3lab-heavy-2026-09-04.tar.gz          # optional; results/benchmark/...
```

Then the lead runs the offline analysis steps (no GPU):

```
uv run scripts/04_exp_calibration.py --as-of 2026-09-04
uv run scripts/08b_exp_mtg_inference.py --as-of 2026-09-04
```

`04` needs `results/exp_shock_v2_predictions.parquet` (from step
`03_timesfm3`) and `results/exp_mtg_benchmark_primary.parquet` (from step
`02b_preregistered_v2`); `08b` needs the 02b primary and leaderboard outputs
plus `results/inference_selfcheck.parquet` (`scripts/08_inference_selfcheck.py`,
run locally). Both stop and name the missing producer otherwise. Review `results/run_status.json` first: it lists every step's
exit code and elapsed time, the commit sha and the resolved checkpoint shas.
Do not commit anything from the heavy archive.

## Troubleshooting

- **Out of memory (OOM).** The probe and 09 record an `oom` row and continue.
  For 02b, restart on a GPU with more memory; the run resumes from the last
  committed cell.
- **401 / 403 or "gated" at preflight.** The account behind `HF_TOKEN` has not
  accepted one of the two licences, or the secret is not attached to the
  notebook. Accept both licences and rerun the preflight cell.
- **Cannot reach Hugging Face / TCGCSV.** Internet is Off in the notebook
  settings. Turn it on (step 05 fetches set release dates from TCGCSV).
- **Cache mismatch or missing cache manifest.** The dataset differs from
  `cache-2026-09-04.json` (or the manifest is not in `GIT_REF`). Rebuild the
  cache with the exact command and `--as-of`, regenerate and push the
  manifest, and re-upload the dataset.
- **"Set GIT_REF to the previous commit sha".** The resumed output came from
  another commit; use the sha it names.
- **Session timeout or quota end.** Everything under `/kaggle/working` is
  saved with a committed version (results under
  `tfm3lab-data/results`); the clone, venv and cache under `/kaggle/tmp` are
  not and are rebuilt. Resume as described above, or bring back
  `tfm3lab-data/results` if packing never ran.
- **A step shows `failed`.** Read `results/logs/<step>.log`; later steps still
  ran. Fix and rerun only that step by setting `EXPERIMENTS` to its id.
