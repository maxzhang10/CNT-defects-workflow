"""Tests for P2-10 (violin grouping includes temp/chirality) and P2-15 (md_steps=0 layout)."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import violin  # noqa: E402


# --- P2-10 ---

def test_group_includes_temperature():
    p = Path("/data/500K/5_5/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    group = violin.get_group_name(p)
    assert "500K" in group
    assert "5_5" in group
    assert "DV_DV" in group


def test_different_temperatures_get_different_groups():
    p50 = Path("/data/50K/5_5/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    p300 = Path("/data/300K/5_5/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    assert violin.get_group_name(p50) != violin.get_group_name(p300)


def test_different_chirality_get_different_groups():
    p55 = Path("/data/300K/5_5/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    p90 = Path("/data/300K/9_0/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    assert violin.get_group_name(p55) != violin.get_group_name(p90)


def test_old_directory_structure_group():
    p = Path("/data/300K/5_5/DV_DV/dpnegf/40000/output/negf.out.pth")
    group = violin.get_group_name(p)
    assert "300K" in group
    assert "5_5" in group
    assert "DV_DV" in group


# --- P2-15 ---

def test_accepts_md_step_layout():
    p = Path("/data/300K/5_5/DV_DV/replica_001/dpnegf/200000/output/negf.out.pth")
    assert violin.is_dpnegf_result(p) is True


def test_accepts_md_steps_zero_layout():
    p = Path("/data/300K/5_5/DV_DV/replica_001/dpnegf/output/negf.out.pth")
    assert violin.is_dpnegf_result(p) is True


def test_rejects_non_dpnegf_layout():
    p = Path("/data/300K/5_5/DV_DV/replica_001/something/output/negf.out.pth")
    assert violin.is_dpnegf_result(p) is False
