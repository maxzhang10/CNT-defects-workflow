"""Unit tests for P1-7: failed flag must cause non-zero exit.

sub_lmps.py / sub_dpnegf.py used to `return` (exit 0) on a prior failed
flag, so in local mode run_multi.py's run_cmd (check=True) treated it as
success and continued consuming stale dump/output. main() now returns
EXIT_PRIOR_FAILURE (2).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sub_lmps  # noqa: E402
import sub_dpnegf  # noqa: E402


def _make_workdir(tmp_path: Path, flags):
    """Create a workdir with the given flag files and required inputs."""
    d = tmp_path / "workdir"
    d.mkdir()
    for name in flags:
        (d / name).write_text("prev failure\n", encoding="utf-8")
    return d


def test_lammps_failed_flag_returns_prior_failure(tmp_path, monkeypatch):
    workdir = _make_workdir(tmp_path, ["lammps_failed.flag"])
    monkeypatch.setattr(
        "sys.argv",
        ["sub_lmps.py", str(workdir), "--scheduler", "local"],
    )
    rc = sub_lmps.main()
    assert rc == sub_lmps.EXIT_PRIOR_FAILURE
    assert rc != 0


def test_dpnegf_failed_flag_returns_prior_failure(tmp_path, monkeypatch):
    workdir = _make_workdir(tmp_path, ["dpnegf_failed.flag"])
    monkeypatch.setattr(
        "sys.argv",
        ["sub_dpnegf.py", str(workdir), "--scheduler", "local"],
    )
    rc = sub_dpnegf.main()
    assert rc == sub_dpnegf.EXIT_PRIOR_FAILURE
    assert rc != 0


def test_lammps_done_returns_zero(tmp_path, monkeypatch):
    # done flag (no prior failure) -> skip, exit 0.
    workdir = _make_workdir(tmp_path, ["lammps_done.flag"])
    monkeypatch.setattr(
        "sys.argv",
        ["sub_lmps.py", str(workdir), "--scheduler", "local"],
    )
    assert sub_lmps.main() == 0


def test_dpnegf_done_with_valid_provenance_returns_zero(tmp_path, monkeypatch):
    import sub_dpnegf as dp

    workdir = _make_workdir(tmp_path, ["dpnegf_done.flag"])
    # Set up a valid provenance: non-empty result + matching hash.
    out = workdir / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"\x80\x02")
    (out / "negf_config_hash.txt").write_text("abc123\n", encoding="utf-8")
    (workdir / "expected_negf_config_hash.txt").write_text("abc123\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["sub_dpnegf.py", str(workdir), "--scheduler", "local"],
    )
    assert dp.main() == 0


def test_exit_prior_failure_constant_is_nonzero():
    # Shared contract: the sentinel is a non-zero, distinct exit code.
    assert sub_lmps.EXIT_PRIOR_FAILURE == sub_dpnegf.EXIT_PRIOR_FAILURE
    assert sub_lmps.EXIT_PRIOR_FAILURE == 2
