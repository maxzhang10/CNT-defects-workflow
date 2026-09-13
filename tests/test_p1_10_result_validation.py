"""Unit tests for P1-10: done flag must validate the result file.

barrier_and_check (run_multi.py) used to treat any dpnegf_done.flag as
success; a done flag with a missing/empty/corrupt output/negf.out.pth
slipped through to batch retry as "success" and only failed at the end-of-
batch conductance collection. Now done-flag dirs are re-checked with
result_is_readable().
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import negf_provenance as npv  # noqa: E402


def test_result_is_nonempty(tmp_path):
    # No output dir.
    assert not npv.result_is_nonempty(tmp_path)

    out = tmp_path / "output"
    out.mkdir()
    # Empty file.
    (out / "negf.out.pth").write_bytes(b"")
    assert not npv.result_is_nonempty(tmp_path)

    # Non-empty file.
    (out / "negf.out.pth").write_bytes(b"\x80\x02data")
    assert npv.result_is_nonempty(tmp_path)


def test_result_is_readable_missing_or_empty(tmp_path):
    assert not npv.result_is_readable(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"")
    assert not npv.result_is_readable(tmp_path)


def test_result_is_readable_valid_bytes(tmp_path):
    # Non-empty bytes; if torch is installed it will fail to load (not a real
    # checkpoint) -> unreadable. If torch is absent, falls back to non-empty
    # check -> readable. Assert the behavior matches torch availability.
    out = tmp_path / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"not a real checkpoint")
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    assert npv.result_is_readable(tmp_path) == (not has_torch)


def test_barrier_flags_bad_dpnegf_result_treated_as_failed(tmp_path, monkeypatch):
    # A done flag present but result file empty -> must raise (not return ok).
    import run_multi

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "dpnegf_done.flag").write_text("done\n", encoding="utf-8")
    # No output/negf.out.pth -> result_is_readable False.

    # Stub out SLURM-wait and flag collection so the test exercises only the
    # result-validation branch.
    monkeypatch.setattr(run_multi, "wait_for_jobs", lambda *a, **k: None)
    monkeypatch.setattr(
        run_multi, "check_flags",
        lambda workdirs, kind: (list(map(Path, workdirs)), [], []),
    )

    import pytest
    with pytest.raises(RuntimeError, match="未成功"):
        run_multi.barrier_and_check(
            [workdir], "dpnegf", poll_interval=1, dry_run=False,
        )


def test_barrier_passes_when_dpnegf_result_readable(tmp_path, monkeypatch):
    import run_multi

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "dpnegf_done.flag").write_text("done\n", encoding="utf-8")
    out = workdir / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"\x80\x02ok")

    # Force the non-torch fallback so the test is environment-independent.
    monkeypatch.setattr(run_multi, "result_is_readable", lambda d: True)
    monkeypatch.setattr(run_multi, "wait_for_jobs", lambda *a, **k: None)
    monkeypatch.setattr(
        run_multi, "check_flags",
        lambda workdirs, kind: (list(map(Path, workdirs)), [], []),
    )

    # No raise.
    run_multi.barrier_and_check(
        [workdir], "dpnegf", poll_interval=1, dry_run=False,
    )


def test_barrier_lammps_does_not_validate_result_file(tmp_path, monkeypatch):
    # LAMMPS has no negf.out.pth; must not be subjected to result_is_readable.
    import run_multi

    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "lammps_done.flag").write_text("done\n", encoding="utf-8")

    called = {"readable": False}
    def _readable(d):
        called["readable"] = True
        return True
    monkeypatch.setattr(run_multi, "result_is_readable", _readable)
    monkeypatch.setattr(run_multi, "wait_for_jobs", lambda *a, **k: None)
    monkeypatch.setattr(
        run_multi, "check_flags",
        lambda workdirs, kind: (list(map(Path, workdirs)), [], []),
    )

    run_multi.barrier_and_check(
        [workdir], "lammps", poll_interval=1, dry_run=False,
    )
    assert called["readable"] is False
