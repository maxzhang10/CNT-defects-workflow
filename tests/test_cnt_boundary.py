"""CNT 单胞轴向边界的浮点误差不得改变电极原子顺序。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage"))
from cnt_geometry import build_unit_cnt, clean_cnt_by_shift_wrap_anchor


def test_clean_cnt_snaps_only_roundoff_at_axial_boundary():
    length = 24.9613220803707
    tube = Atoms(
        "C4",
        positions=[
            (1, 1, -1.0373646379911987e-13),
            (2, 1, 0.0),
            (3, 1, 0.12117146640932382),
            (4, 1, 1.0),
        ],
        cell=[10.0, 10.0, length],
        pbc=True,
    )

    cleaned = clean_cnt_by_shift_wrap_anchor(tube)
    z = cleaned.positions[:, 2]

    assert np.count_nonzero(z == 0.0) == 2
    assert np.all(z >= 0.0)
    assert np.all(z < length - 1e-10)
    np.testing.assert_allclose(z[2:], [0.12117146640932382, 1.0], atol=1e-10)


def test_generated_13_7_unit_has_nonnegative_ordered_z():
    tube = build_unit_cnt(13, 7, vacuum=50.0)
    cleaned = clean_cnt_by_shift_wrap_anchor(tube)
    z = cleaned.positions[:, 2]

    assert len(cleaned) == 412
    assert z[0] == 0.0
    assert np.all(z >= 0.0)
    assert np.all(np.diff(z) >= 0.0)
