"""Tests for P2-2: squeue query failure must not be treated as completion."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import slurm_utils as su  # noqa: E402


def test_slurm_query_error_defined():
    assert hasattr(su, "SlurmQueryError")
    assert issubclass(su.SlurmQueryError, RuntimeError)


def test_squery_failure_raises_not_empty(monkeypatch):
    """Non-zero returncode + empty stdout -> SlurmQueryError, not empty set."""
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "slurm_load_jobs error: Invalid authentication"

    monkeypatch.setattr(su.subprocess, "run", lambda *a, **k: FakeResult())
    import pytest
    with pytest.raises(su.SlurmQueryError):
        su._running_job_ids(["12345"])


def test_squery_nonzero_with_partial_stdout_returns_alive(monkeypatch):
    """Non-zero returncode but with stdout -> extract alive jobs (partial completion)."""
    class FakeResult:
        returncode = 1
        stdout = "12345\n"
        stderr = "some jobs not found"

    monkeypatch.setattr(su.subprocess, "run", lambda *a, **k: FakeResult())
    alive = su._running_job_ids(["12345", "99999"])
    assert alive == {"12345"}


def test_wait_for_jobs_retries_then_raises_on_query_failure(monkeypatch):
    calls = {"n": 0, "sleeps": 0}

    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "auth error"

    def fake_run(*a, **k):
        calls["n"] += 1
        return FakeResult()

    def fake_sleep(s):
        calls["sleeps"] += 1

    monkeypatch.setattr(su.subprocess, "run", fake_run)
    monkeypatch.setattr(su.time, "sleep", fake_sleep)

    import pytest
    with pytest.raises(su.SlurmQueryError):
        su.wait_for_jobs(["12345"], poll_interval=5, label="TEST")

    # Should have retried 5 times before giving up.
    assert calls["n"] == 6  # 1 initial + 5 retries
    assert calls["sleeps"] == 5


def test_wait_for_jobs_empty_job_ids_returns_immediately():
    su.wait_for_jobs([], poll_interval=1, label="TEST")
