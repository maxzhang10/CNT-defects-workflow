import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
from ase.build import nanotube

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage"))
from cnt_geometry import set_reference_cyl
from defects import CNT_5775, find_nearest


@pytest.mark.parametrize("prior_vacancy", [False, True])
def test_axial_bond_rotates_to_circumference(prior_vacancy):
    tube = nanotube(8, 0, length=4)
    tube.translate([11.0, -4.0, 2.0])
    axis_xy = tube.positions[:, :2].mean(axis=0)
    if prior_vacancy:
        set_reference_cyl(tube)
        tube = tube.copy()
        del tube[0]

    i = np.argmin(abs(tube.positions[:, 2] - tube.positions[:, 2].mean()))
    for rank, (_, j) in enumerate(find_nearest(tube, i + 1)):
        bond = tube.positions[j] - tube.positions[i]
        if np.linalg.norm(bond[:2]) < 1e-10:
            break
    else:
        pytest.fail("Expected an axial C–C bond in a zigzag CNT")

    midpoint = (tube.positions[i] + tube.positions[j]) / 2
    radial = np.r_[midpoint[:2] - axis_xy, 0.0]
    tangent_bond = np.cross(radial / np.linalg.norm(radial), bond)
    result = CNT_5775(tube, i + 1, neighbor_rank=rank)
    expected = tube.positions.copy()
    expected[i] = midpoint - tangent_bond / 2
    expected[j] = midpoint + tangent_bond / 2
    np.testing.assert_allclose(result.positions, expected, atol=1e-10, rtol=0)
