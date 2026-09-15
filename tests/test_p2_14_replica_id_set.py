"""Tests for P2-14: expected replica count must compare ID sets, not just count.

A configuration directory that contains replica_001 and a mis-numbered
replica_003 (when expected=2) must report replica_002 as missing and
replica_003 as unexpected, rather than silently treating valid_replicas=2
as complete.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_md_conductance as cmc  # noqa: E402


def _make_result(replica, conductance=0.5, expected_replicas=2):
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
        expected_replicas=expected_replicas,
        structure_seed=1,
        lammps_seed=1,
        md_steps=0,
        dpnegf_sample="0",
        conductance_G0=conductance,
        ln_conductance_G0=0.0,
        e_fermi_eV=0.0,
        transport_temperature_K=300.0,
        conductance_method="test",
        grid_min_eV=-0.5,
        grid_max_eV=0.5,
        n_energy_points=10,
        replica_dir=f"/r/{replica}",
        result_file=f"/r/{replica}/out.pth",
        conductance_mode="fermi",
        censored=False,
    )


def test_summary_has_id_fields():
    assert "missing_replica_ids" in cmc.ConfigurationSummary.__dataclass_fields__
    assert "unexpected_replica_ids" in cmc.ConfigurationSummary.__dataclass_fields__


def test_missing_and_unexpected_detected():
    """replica_001 + replica_003 with expected=2 → missing {2}, unexpected {3}."""
    results = [
        _make_result(1),
        _make_result(3),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    # Count looks fine (2 valid) but IDs are wrong.
    assert summary.valid_replicas == 2
    assert summary.missing_replica_ids == (2,)
    assert summary.unexpected_replica_ids == (3,)


def test_complete_set_no_mismatch():
    """replica_001 + replica_002 with expected=2 → no missing/unexpected."""
    results = [
        _make_result(1),
        _make_result(2),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    assert summary.missing_replica_ids == ()
    assert summary.unexpected_replica_ids == ()
    assert summary.valid_replicas == 2


def test_duplicate_id_flagged():
    """Two replica_001 with expected=2 → missing {2}, no unexpected (dup)."""
    results = [
        _make_result(1),
        _make_result(1),
    ]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    assert summary.missing_replica_ids == (2,)
    # {1} is a subset of {1,2}, so nothing unexpected — but count still 2.
    assert summary.unexpected_replica_ids == ()


def test_out_of_range_only():
    """Only replica_003 when expected=2 → missing {1,2}, unexpected {3}."""
    results = [_make_result(3)]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=2,
    )
    assert summary.missing_replica_ids == (1, 2)
    assert summary.unexpected_replica_ids == (3,)


def test_no_expected_ids_when_none():
    """When expected is None, no ID comparison is performed."""
    results = [_make_result(5, expected_replicas=None)]
    summary = cmc.summarize_configuration(
        configuration_dir=Path("/cfg"),
        results=results,
        n_problems=0,
        expected_replicas_override=None,
    )
    assert summary.expected_replicas is None
    assert summary.missing_replica_ids == ()
    assert summary.unexpected_replica_ids == ()
