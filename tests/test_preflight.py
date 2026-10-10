"""Offline tests for tfm3lab.preflight: error mapping with a monkeypatched
huggingface_hub / torch. No network, no GPU."""

from __future__ import annotations

import sys
import types

import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from tfm3lab import preflight


def _make_error(cls, status=None):
    err = cls.__new__(cls)
    Exception.__init__(err, "boom")
    if status is not None:
        err.response = types.SimpleNamespace(status_code=status)
    return err


def _patch_model_info(monkeypatch, result=None, exc=None):
    def fake(self, repo_id, revision=None, token=None, **kw):
        if exc is not None:
            raise exc
        return types.SimpleNamespace(sha=result)

    monkeypatch.setattr(HfApi, "model_info", fake)


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake")


def test_require_hf_token_missing_names_kaggle_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        preflight.require_hf_token()


def test_require_hf_token_returns_env(monkeypatch):
    assert preflight.require_hf_token() == "hf_fake"


@pytest.mark.parametrize(
    ("exc", "needle"),
    [
        (_make_error(GatedRepoError, 403), "gated"),
        (_make_error(RepositoryNotFoundError, 404), "not found"),
        (_make_error(RevisionNotFoundError, 404), "revision"),
        (_make_error(HfHubHTTPError, 401), "HTTP 401"),
        (_make_error(HfHubHTTPError, 403), "HTTP 403"),
        (ConnectionError("down"), "Internet"),
    ],
)
def test_checkpoint_access_errors_map_to_systemexit(monkeypatch, exc, needle):
    _patch_model_info(monkeypatch, exc=exc)
    with pytest.raises(SystemExit) as info:
        preflight.require_checkpoint_access("google/timesfm-3.0-pytorch", "abc")
    msg = str(info.value)
    assert needle in msg
    hints = ("huggingface.co/google/timesfm-3.0-pytorch", "Internet", "revision")
    assert any(k in msg for k in hints)


def test_checkpoint_access_returns_sha(monkeypatch):
    _patch_model_info(monkeypatch, result="deadbeef")
    assert preflight.require_checkpoint_access("google/x") == "deadbeef"


def test_preflight_model_returns_checkpoint_dict(monkeypatch):
    _patch_model_info(monkeypatch, result="deadbeef")
    monkeypatch.setattr(preflight, "_cuda_available", lambda: False)
    out = preflight.preflight_model("google/x", "main", allow_cpu=True)
    assert out == {
        "repo_id": "google/x",
        "revision_requested": "main",
        "revision_resolved": "deadbeef",
    }


def test_require_gpu_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(preflight, "_cuda_available", lambda: False)
    with pytest.raises(SystemExit, match="GPU"):
        preflight.require_gpu()
    preflight.require_gpu(allow_cpu=True)  # must not raise


def test_run_environment_with_fake_torch(monkeypatch):
    fake = types.SimpleNamespace(
        __version__="2.9.0",
        version=types.SimpleNamespace(cuda="12.4"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda i: "Tesla T4", device_count=lambda: 2
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    env = preflight.run_environment()
    assert env == {
        "gpu_name": "Tesla T4",
        "cuda_version": "12.4",
        "torch_version": "2.9.0",
        "device_count": 2,
    }


def test_run_environment_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    env = preflight.run_environment()
    assert env["gpu_name"] == "cpu"
    assert env["torch_version"] is None
    assert env["device_count"] == 0


def test_pinned_checkpoint_revisions():
    import importlib

    from tfm3lab import config

    importlib.reload(config)
    assert config.CHECKPOINT_REVISION == "43046b85ec22d584a13f8098c2ed39c889e129c2"
    assert config.CHECKPOINT_REVISION_2P5 == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
