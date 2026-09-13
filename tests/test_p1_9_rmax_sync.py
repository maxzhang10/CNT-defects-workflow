"""Unit test for P1-9: workflow r_max synced to AtomicData_options.r_max.

copy_input_dpnegf.py used r_max only to compute principal-layer partitioning
but left the top-level AtomicData_options.r_max at the template's fixed 6.5,
so PL coupling range and Hamiltonian cutoff could diverge when r_max != 6.5.
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import copy_input_dpnegf as ci  # noqa: E402


def _make_struct_fdf(path: Path, n_atoms: int):
    """Write a minimal STRUCT.fdf with the given number of atoms."""
    lines = ["NumberOfAtoms %d" % n_atoms, "%block AtomicCoordinatesAndAtomicSpecies"]
    for i in range(n_atoms):
        lines.append(f"0.0 0.0 0.0 1")
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_input_json(path: Path, r_max=6.5):
    """Write a minimal input.json mirroring the template's relevant structure."""
    data = {
        "task_options": {
            "espacing": 0.1,
            "emin": -0.5,
            "emax": 0.5,
            "self_energy_options": {"cache": {}},
            "stru_options": {
                "lead_L": {"id": "0-0"},
                "device": {"id": "0-0"},
                "lead_R": {"id": "0-0"},
            },
        },
        "AtomicData_options": {"r_max": r_max},
    }
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def _make_run_py(path: Path):
    """Write a minimal run.py with model_path / structure lines to rewrite."""
    path.write_text(
        'model_path = "./old.pth"\n'
        'structure = "./old.xyz"\n',
        encoding="utf-8",
    )


def test_r_max_synced_to_atomic_data_options(tmp_path):
    # Use a (5,5) tube: geo_info(5,5, r_max, 5) yields a deterministic PL size.
    # Pick n_total large enough that 2*n_elec < n_total for any reasonable r_max.
    import cnt_geometry

    r_max = 7.5  # != template default 6.5
    T, N_uc, l_PL, length = cnt_geometry.geo_info(5, 5, r_max, 5)
    n_atoms_per_pl = N_uc * l_PL
    n_elec = 2 * n_atoms_per_pl
    n_total = N_uc * length
    assert n_total > 2 * n_elec, "test fixture: n_total must exceed 2*n_elec"

    _make_struct_fdf(tmp_path / "STRUCT.fdf", n_total)
    _make_input_json(tmp_path / "input.json", r_max=6.5)  # stale template value
    _make_run_py(tmp_path / "run.py")

    ci.update_input_json_for_leaf(
        leaf_dir=tmp_path,
        model_filename="nnenv.pth",
        chirality=(5, 5),
        espacing=0.1,
        negf_energy_window=(-0.5, 0.5),
        self_energy_cache={"use_saved": True},
        r_max=r_max,
        n_lead_pl=2,
        l_def=5,
    )

    data = json.loads((tmp_path / "input.json").read_text(encoding="utf-8"))
    assert data["AtomicData_options"]["r_max"] == r_max


def test_r_max_atomic_data_options_created_if_absent(tmp_path):
    # Template missing AtomicData_options entirely -> should be created.
    import cnt_geometry

    r_max = 6.5
    T, N_uc, l_PL, length = cnt_geometry.geo_info(5, 5, r_max, 5)
    n_total = N_uc * length

    _make_struct_fdf(tmp_path / "STRUCT.fdf", n_total)
    data = {
        "task_options": {
            "espacing": 0.1,
            "emin": -0.5,
            "emax": 0.5,
            "self_energy_options": {"cache": {}},
            "stru_options": {
                "lead_L": {"id": "0-0"},
                "device": {"id": "0-0"},
                "lead_R": {"id": "0-0"},
            },
        },
        # no AtomicData_options
    }
    (tmp_path / "input.json").write_text(json.dumps(data, indent=4), encoding="utf-8")
    _make_run_py(tmp_path / "run.py")

    ci.update_input_json_for_leaf(
        leaf_dir=tmp_path,
        model_filename="nnenv.pth",
        chirality=(5, 5),
        espacing=0.1,
        negf_energy_window=(-0.5, 0.5),
        self_energy_cache=None,
        r_max=r_max,
        n_lead_pl=2,
        l_def=5,
    )

    out = json.loads((tmp_path / "input.json").read_text(encoding="utf-8"))
    assert out["AtomicData_options"]["r_max"] == r_max
