"""Unit tests for local-scheduler command construction.

Regression tests for P1-5: `--scheduler local` used to invoke the SLURM
run.sh templates, which require SLURM_SUBMIT_DIR / cluster modules / srun
(LAMMPS) and a hardcoded venv (DPNEGF), so local mode failed immediately.

The local branches now bypass run.sh: LAMMPS calls the executable directly
and DPNEGF invokes `python run.py` with the current interpreter.
"""
import os
import sys
from pathlib import Path

# Make repo root and stage importable when run via `pytest` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sub_lmps  # noqa: E402
import sub_dpnegf  # noqa: E402


# ---------------------------------------------------------------- sub_lmps

def test_lammps_local_default_uses_lmp_executable_directly():
    os.environ.pop("CNT_LAMMPS_CMD", None)
    cmd = sub_lmps.local_lammps_command()
    # Must not delegate to the SLURM-templated run.sh.
    assert "bash" not in cmd
    assert "run.sh" not in cmd
    assert cmd[0] == "lmp"
    assert "-in" in cmd and cmd[cmd.index("-in") + 1] == "in.lammps"
    assert "-log" in cmd and cmd[cmd.index("-log") + 1] == "lammps.log"


def test_lammps_local_respects_env_override_with_mpi():
    os.environ["CNT_LAMMPS_CMD"] = "mpirun -np 32 lmp_intel_cpu_intelmpi"
    try:
        cmd = sub_lmps.local_lammps_command()
    finally:
        os.environ.pop("CNT_LAMMPS_CMD")
    assert cmd[:4] == ["mpirun", "-np", "32", "lmp_intel_cpu_intelmpi"]
    assert "-in" in cmd and cmd[cmd.index("-in") + 1] == "in.lammps"
    assert "-log" in cmd and cmd[cmd.index("-log") + 1] == "lammps.log"


def test_lammps_local_rejects_empty_env_override():
    os.environ["CNT_LAMMPS_CMD"] = "   "
    try:
        import pytest
        with pytest.raises(ValueError):
            sub_lmps.local_lammps_command()
    finally:
        os.environ.pop("CNT_LAMMPS_CMD")


# --------------------------------------------------------------- sub_dpnegf

def test_dpnegf_local_python_defaults_to_current_interpreter():
    os.environ.pop("CNT_DPTB_PYTHON", None)
    assert sub_dpnegf.local_dpnegf_python() == sys.executable


def test_dpnegf_local_python_respects_env_override():
    os.environ["CNT_DPTB_PYTHON"] = "/opt/venv/bin/python"
    try:
        assert sub_dpnegf.local_dpnegf_python() == "/opt/venv/bin/python"
    finally:
        os.environ.pop("CNT_DPTB_PYTHON")


def test_dpnegf_provenance_hash_copied_on_success(tmp_path):
    # expected hash present -> should be copied to output/negf_config_hash.txt
    expected = sub_dpnegf.expected_hash_path(tmp_path)
    expected.write_text("deadbeef\n", encoding="utf-8")

    sub_dpnegf.copy_provenance_hash_on_success(tmp_path)

    target = sub_dpnegf.provenance_hash_path(tmp_path)
    assert target.is_file()
    assert target.read_text(encoding="utf-8").strip() == "deadbeef"


def test_dpnegf_provenance_hash_noop_when_expected_absent(tmp_path):
    # No expected hash file -> no crash, no target created.
    sub_dpnegf.copy_provenance_hash_on_success(tmp_path)
    assert not sub_dpnegf.provenance_hash_path(tmp_path).exists()
