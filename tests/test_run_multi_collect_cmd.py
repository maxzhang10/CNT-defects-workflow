"""Unit tests for run_multi.py command construction.

Regression test for the P1-1 bug where run_multi.py invoked
collect_md_conductance.py with unsupported --config / --structure-dir
options, causing an argparse "ambiguous option" failure after all the
expensive DPNEGF computation had already finished.

Energy references (Ef/Ec/Ev) and the transport temperature are now read
directly from each replica's negf.out.pth, so the collect command no longer
carries --e-fermi or --transport-temperature-k.
"""
import sys
from pathlib import Path

# Make the repo root importable when run via `pytest` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_multi  # noqa: E402


def _flatten_pairs(cmd, flag):
    """Return the values supplied for a repeated `flag` in `cmd`."""
    vals = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag and i + 1 < len(cmd):
            vals.append(cmd[i + 1])
            i += 2
        else:
            i += 1
    return vals


def test_cmd_uses_supported_options_only():
    py = "/usr/bin/python3"
    script = Path("/repo/stage/collect_md_conductance.py")
    root = Path("/data/300K/5_5")
    dirs = [Path("/data/300K/5_5/DV_DV"), Path("/data/300K/5_5/SW")]

    cmd = run_multi.build_collect_conductance_cmd(
        python=py,
        script=script,
        root=root,
        configuration_dirs=dirs,
    )

    # The collector accepts exactly these flags; any other long option is a bug.
    assert cmd[0] == py
    assert cmd[1] == str(script)
    assert "--root" in cmd
    assert cmd[cmd.index("--root") + 1] == str(root)

    # Energy references now come from negf.out.pth; no external energy flag.
    assert "--e-fermi" not in cmd
    assert "--transport-temperature-k" not in cmd

    # The flags that used to trigger the argparse failure must be absent.
    assert "--config" not in cmd
    assert "--structure-dir" not in cmd

    # Each configuration directory is passed via the correct --configuration-dir.
    assert _flatten_pairs(cmd, "--configuration-dir") == [str(d) for d in dirs]


def test_cmd_has_no_energy_flags_with_single_dir():
    # Even with a single configuration dir, no external energy flags appear.
    cmd = run_multi.build_collect_conductance_cmd(
        python="/usr/bin/python3",
        script=Path("/repo/stage/collect_md_conductance.py"),
        root=Path("/data"),
        configuration_dirs=[Path("/data/DV_DV")],
    )
    assert "--e-fermi" not in cmd
    assert "--transport-temperature-k" not in cmd
    assert _flatten_pairs(cmd, "--configuration-dir") == [str(Path("/data/DV_DV"))]


def test_cmd_flags_all_appear_before_repeated_configuration_dirs():
    # Collector's argparse requires --root; the construction must always emit it
    # even with zero configuration directories (the collector then scans root).
    cmd = run_multi.build_collect_conductance_cmd(
        python="/usr/bin/python3",
        script=Path("/repo/stage/collect_md_conductance.py"),
        root=Path("/data"),
        configuration_dirs=[],
    )
    assert "--root" in cmd
    assert "--e-fermi" not in cmd
    assert _flatten_pairs(cmd, "--configuration-dir") == []
