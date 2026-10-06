"""Unit tests for P1-6: DPNEGF run.sh state-flag protocol.

The DPNEGF run.sh template must clean stale flags at start, write started
immediately, and use a trap to write failed (or done) on exit.

Historical note: this module also covered resubmit_negf.py, which was
removed when flow-task resubmission moved into run_multi.py (ff5fe2e);
re-running run_multi.py on the same root now handles retry/state cleanup.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
