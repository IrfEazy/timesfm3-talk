"""v3 studies (S1/S2/S3): the file contract between the local request builder,
the GPU forecast runner and the offline analysis.

- `requests`: the frozen requests/forecasts parquet schemas, their IO and the
  validators. Requests hold scaled contexts only, never outcomes.
- `runner`: the resumable, block-prioritised forecast loop (scripts/14_v3_forecast.py).

Both modules import pandas/pyarrow/numpy only; the model is injected, so
everything here is testable without a GPU.
"""
