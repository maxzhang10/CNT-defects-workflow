"""Tests for P2-1: non-positive conductance handling.

Substantially negative conductance must fail the replica; tiny negative
noise is clipped to zero and marked censored (excluded from log stats but
reported). valid_replicas must not count censored replicas as fully valid.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_md_conductance as cmc  # noqa: E402


def _make_result(replica, conductance, censored=False, mode="fermi"):
    return cmc.ReplicaResult(
        configuration_dir="/cfg",
        configuration_name="DV_DV",
        temperature_K=300,
        chirality_m=5,
        chirality_n=5,
        structures="5775",
        l_def=5,
        n_defects=2,
        density_A_inv=0.1,
        replica=replica,
        expected_replicas=2,
        structure_seed=1,
        lammps_seed=1,
        md_steps=0,
        dpnegf_sample="0",
        conductance_G0=conductance,
        ln_conductance_G0=(None if conductance <= 0 else __import__("math").log(conductance)),
        e_fermi_eV=0.0,
        transport_temperature_K=300.0,
        conductance_method="test",
        grid_min_eV=-0.5,
        grid_max_eV=0.5,
        n_energy_points=10,
        replica_dir=f"/r/{replica}",
        result_file=f"/r/{replica}/out.pth",
        conductance_mode=mode,
        censored=censored,
    )


def test_fields_exist():
    assert "censored" in cmc.ReplicaResult.__dataclass_fields__
    assert "n_censored_replicas" in cmc.ConfigurationSummary.__dataclass_fields__


def test_censored_excluded_from_log_stats():
    # One positive (0.5), one censored-zero replica.
    results = [
        _make_result(1, 0.5, censored=False),
        _make_result(2, 0.0, censored=True),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    # valid_replicas counts only non-censored.
    assert summary.valid_replicas == 1
    assert summary.n_censored_replicas == 1
    # geometric mean over the single positive value = 0.5.
    assert summary.geometric_mean_G0 is not None
    assert abs(summary.geometric_mean_G0 - 0.5) < 1e-9


def test_all_positive_not_censored():
    results = [
        _make_result(1, 0.4),
        _make_result(2, 0.6),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    assert summary.valid_replicas == 2
    assert summary.n_censored_replicas == 0
    # geometric mean of 0.4, 0.6 = sqrt(0.24).
    import math
    assert abs(summary.geometric_mean_G0 - math.sqrt(0.24)) < 1e-9


def test_all_zero_censored_gives_none_log_stats():
    results = [
        _make_result(1, 0.0, censored=True),
        _make_result(2, 0.0, censored=True),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    assert summary.valid_replicas == 0
    assert summary.n_censored_replicas == 2
    assert summary.geometric_mean_G0 is None
    assert summary.mean_ln_G0 is None
