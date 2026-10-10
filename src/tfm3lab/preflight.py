"""Fail-fast environment checks (Preregistration v2 s7).

A multi-hour Kaggle run must not die at model load because the GPU is off,
the HF_TOKEN secret is missing, or the gated licence was never accepted.
Every check raises SystemExit with an actionable message. Heavy imports
(torch, huggingface_hub) are lazy so the module imports anywhere.
"""

from __future__ import annotations

import os

from . import config

LICENSE_URL_3 = "https://huggingface.co/google/timesfm-3.0-pytorch"


def _license_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require_gpu(allow_cpu: bool = False) -> None:
    """Exit unless a CUDA device is visible (or allow_cpu is set, for smoke runs)."""
    if _cuda_available():
        return
    if allow_cpu:
        print("WARNING: no GPU visible; continuing on CPU (--allow-cpu).")
        return
    raise SystemExit(
        "error: no CUDA GPU is available. On Kaggle enable an accelerator "
        "(Settings -> Accelerator -> GPU) and restart the session, or pass "
        "--allow-cpu for a slow smoke run."
    )


def require_hf_token() -> str:
    """Return the HF token from HF_TOKEN (or the cached `hf auth login` token)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None
    if token:
        return token
    raise SystemExit(
        "error: no Hugging Face token found. Add a Kaggle secret named HF_TOKEN "
        "(Add-ons -> Secrets, attach it to the notebook) or export HF_TOKEN / run "
        "`hf auth login` locally. The token must belong to an account that accepted "
        f"the model licence ({LICENSE_URL_3})."
    )


def require_checkpoint_access(repo_id: str, revision: str | None = None) -> str:
    """Resolve and return the commit sha of repo_id@revision (None = default
    branch head), mapping every hub failure to a clear SystemExit."""
    token = require_hf_token()
    url = _license_url(repo_id)
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
        raise SystemExit(f"error: huggingface_hub is not installed ({exc}).") from exc

    try:
        info = HfApi().model_info(repo_id, revision=revision, token=token)
    except GatedRepoError as exc:
        raise SystemExit(
            f"error: {repo_id} is gated and this token has no access. Accept the licence at "
            f"{url} with the account that owns the HF_TOKEN Kaggle secret, then rerun."
        ) from exc
    except RevisionNotFoundError as exc:
        raise SystemExit(
            f"error: revision {revision!r} does not exist for {repo_id}. Fix "
            "TFM3LAB_CHECKPOINT_REVISION (or config.CHECKPOINT_REVISION)."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            f"error: {repo_id} was not found or is not visible to this token. Check the repo id, "
            f"accept the licence at {url}, and verify the HF_TOKEN Kaggle secret is valid."
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise SystemExit(
                f"error: Hugging Face rejected the token for {repo_id} (HTTP {status}). Accept "
                f"the licence at {url} and set a valid HF_TOKEN Kaggle secret."
            ) from exc
        raise SystemExit(f"error: Hugging Face request for {repo_id} failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - connection errors differ across hub versions
        raise SystemExit(
            f"error: cannot reach Hugging Face to verify {repo_id} ({exc}). On Kaggle enable "
            "Internet in the notebook settings."
        ) from exc

    sha = getattr(info, "sha", None)
    if not sha:
        raise SystemExit(f"error: Hugging Face returned no commit sha for {repo_id}@{revision}.")
    return sha


def preflight_model(
    repo_id: str = config.CHECKPOINT_ID,
    revision: str | None = None,
    *,
    allow_cpu: bool = False,
) -> dict:
    """GPU + token + checkpoint access; returns the manifest `checkpoint` dict."""
    require_gpu(allow_cpu=allow_cpu)
    revision = revision or config.CHECKPOINT_REVISION
    resolved = require_checkpoint_access(repo_id, revision)
    return {"repo_id": repo_id, "revision_requested": revision, "revision_resolved": resolved}


def run_environment() -> dict:
    """GPU name, CUDA/torch versions and device count; 'cpu'/None when absent."""
    env: dict = {"gpu_name": "cpu", "cuda_version": None, "torch_version": None, "device_count": 0}
    try:
        import torch
    except ImportError:
        return env
    env["torch_version"] = getattr(torch, "__version__", None)
    try:
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = getattr(torch.version, "cuda", None)
            env["device_count"] = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - environment probing must never fail a run
        pass
    return env
