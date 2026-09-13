"""Unit tests for P1-6 fixes.

Two concerns:
1. resubmit_negf.py must clear stale done/failed/submitted flags and output
   before re-submitting, so it does not bypass the flag protocol.
2. The DPNEGF run.sh template must clean stale flags at start, write started
   immediately, and use a trap to write failed (or done) on exit.
"""
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import resubmit_negf  # noqa: E402


def test_negf_succeeded_checks_nonempty_result(tmp_path):
    # No output dir -> not succeeded.
    assert not resubmit_negf.negf_succeeded(tmp_path)

    # Empty result file -> not succeeded.
    out = tmp_path / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"")
    assert not resubmit_negf.negf_succeeded(tmp_path)

    # Non-empty result file -> succeeded.
    (out / "negf.out.pth").write_bytes(b"\x80\x02")
    assert resubmit_negf.negf_succeeded(tmp_path)


def test_reset_workdir_state_clears_all_flags_and_output(tmp_path):
    # Seed a full set of stale flags, a submit record, and output.
    for name in (
        "dpnegf_done.flag",
        "dpnegf_failed.flag",
        "dpnegf_started.flag",
        "dpnegf_submitted.flag",
        "dpnegf_submit_failed.flag",
        "job_id.txt",
        "expected_negf_config_hash.txt",
    ):
        (tmp_path / name).write_text("stale\n", encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    (out / "negf.out.pth").write_bytes(b"\x80\x02")
    (out / "negf_config_hash.txt").write_text("deadbeef\n", encoding="utf-8")

    resubmit_negf.reset_workdir_state(tmp_path)

    # All state flags and submit records removed.
    for name in (
        "dpnegf_done.flag",
        "dpnegf_failed.flag",
        "dpnegf_started.flag",
        "dpnegf_submitted.flag",
        "dpnegf_submit_failed.flag",
        "job_id.txt",
    ):
        assert not (tmp_path / name).exists(), f"{name} should be removed"

    # output/ removed entirely.
    assert not out.exists()

    # expected_negf_config_hash.txt MUST be preserved (copy_input wrote it;
    # sub_dpnegf needs it to write provenance after a successful re-run).
    assert (tmp_path / "expected_negf_config_hash.txt").exists()


def test_find_dpnegf_workdirs_dedupes(tmp_path):
    # Build: root/dpnegf/<t>/run.sh and root/dpnegf/run.sh
    dpnegf = tmp_path / "dpnegf"
    dpnegf.mkdir()
    (dpnegf / "run.sh").write_text("#!/bin/bash\n")
    sub = dpnegf / "40000"
    sub.mkdir()
    (sub / "run.sh").write_text("#!/bin/bash\n")

    workdirs = list(resubmit_negf.find_dpnegf_workdirs(tmp_path))

    # Each run.sh parent is a workdir; no duplicates.
    assert len(workdirs) == 2
    assert len({w.resolve() for w in workdirs}) == 2
    parents = {w.name for w in workdirs}
    assert parents == {"dpnegf", "40000"}


def test_dpnegf_run_sh_implements_state_protocol():
    run_sh = _REPO_ROOT / "input_files" / "dpnegf" / "run.sh"
    text = run_sh.read_text(encoding="utf-8")

    # Robust error handling matching the LAMMPS template.
    assert "set -Eeuo pipefail" in text

    # Clean stale flags at start.
    assert "rm -f dpnegf_done.flag" in text
    assert "rm -f dpnegf_failed.flag" in text
    assert "rm -f dpnegf_started.flag" in text

    # Write started immediately.
    assert "dpnegf_started.flag" in text
    assert text.index("rm -f dpnegf_started.flag") < text.index("dpnegf_started.flag")

    # trap writes failed on non-zero exit, done on success.
    assert "trap on_exit EXIT" in text
    assert "dpnegf_failed.flag" in text
    assert "dpnegf_done.flag" in text
    assert "rc != 0" in text or "rc -ne 0" in text

    # On success, remove stale failed and write provenance hash.
    assert "cp expected_negf_config_hash.txt output/negf_config_hash.txt" in text
    # The done/failed writing lives inside on_exit (not the old inline form).
    assert "touch dpnegf_done.flag" in text
