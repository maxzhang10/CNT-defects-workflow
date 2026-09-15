"""Tests for P2-12: any *.pth model name should be accepted, not just nnenv*.pth."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_multi  # noqa: E402
import sub_dpnegf  # noqa: E402


def _make_dpnegf_workdir(tmp_path, model_name="nnenv.iter201150.pth"):
    d = tmp_path / "dpnegf_step"
    d.mkdir()
    for name in ("input.json", "run.py", "run.sh"):
        (d / name).write_text("placeholder")
    (d / "9_0.xyz").write_text("placeholder")
    (d / model_name).write_text("model")
    return d


def test_is_dpnegf_workdir_accepts_nnenv_prefix(tmp_path):
    d = _make_dpnegf_workdir(tmp_path, "nnenv.iter201150.pth")
    assert run_multi.is_dpnegf_workdir(d) is True


def test_is_dpnegf_workdir_accepts_arbitrary_name(tmp_path):
    d = _make_dpnegf_workdir(tmp_path, "model.pth")
    assert run_multi.is_dpnegf_workdir(d) is True


def test_is_dpnegf_workdir_rejects_no_pth(tmp_path):
    d = _make_dpnegf_workdir(tmp_path, "model.pth")
    (d / "model.pth").unlink()
    assert run_multi.is_dpnegf_workdir(d) is False


def test_sub_dpnegf_check_required_accepts_arbitrary_name(tmp_path):
    d = _make_dpnegf_workdir(tmp_path, "checkpoint.pth")
    # Should not raise.
    sub_dpnegf.check_required_files(d)


def test_sub_dpnegf_check_required_rejects_missing_pth(tmp_path):
    import pytest
    d = _make_dpnegf_workdir(tmp_path, "model.pth")
    (d / "model.pth").unlink()
    with pytest.raises(FileNotFoundError, match=r"\*\.pth"):
        sub_dpnegf.check_required_files(d)
