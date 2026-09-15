"""Tests for P2-9: POSCAR must use repositioned atoms (same as data.lmp).

Without repositioning, H atoms remain at the array end; md_steps=0 path
generates STRUCT.fdf directly from POSCAR, so the right electrode range
would include H instead of pure C, breaking the homogeneous-carbon-lead
assumption.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exporters  # noqa: E402
import numpy as np  # noqa: E402


def test_reposition_hydrogens_moves_h_before_right_lead():
    """reposition_hydrogens should place H between middle C and right C."""
    from ase import Atoms
    # 10 C atoms + 2 H atoms appended at end.
    # N=2: left=[C0,C1], right=[C8,C9], middle=[C2..C7], H=[H10,H11]
    c_positions = [[i * 1.0, 0.0, 0.0] for i in range(10)]
    h_positions = [[5.0, 1.0, 0.0], [5.0, -1.0, 0.0]]
    atoms = Atoms("C10H2", positions=c_positions + h_positions)

    repositioned = exporters.reposition_hydrogens(atoms, N=2)

    symbols = repositioned.get_chemical_symbols()
    # Expected: C C (left) | C*6 (middle) | H H | C C (right)
    assert symbols[:2] == ["C", "C"]
    assert symbols[2:8] == ["C"] * 6
    assert symbols[8:10] == ["H", "H"]
    assert symbols[10:12] == ["C", "C"]


def test_reposition_no_h_atoms_unchanged():
    from ase import Atoms
    atoms = Atoms("C10", positions=[[i, 0, 0] for i in range(10)])
    repositioned = exporters.reposition_hydrogens(atoms, N=2)
    assert repositioned.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_reposition_insufficient_c_raises():
    import pytest
    from ase import Atoms
    atoms = Atoms("C3H2", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 1, 0], [5, -1, 0]])
    with pytest.raises(ValueError, match="C 原子数量不足"):
        exporters.reposition_hydrogens(atoms, N=2)
