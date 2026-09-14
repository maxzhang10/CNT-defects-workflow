"""Tests for scattering-region-only structure generation (length == l_def).

Regression test for the port of /personal/CNT_defects_package_scatter's
`total_length` mechanism: when total_length == l_def the defect region
spans the whole tube (z from 0), with no electrode PL buffers; otherwise
it stays in the middle 2PL-2PL-defects-2PL-2PL layout.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cnt_geometry  # noqa: E402
import ele_multi_defects_ele as gen  # noqa: E402


def _build_tube(m, n, length):
    tube_unit = cnt_geometry.build_unit_cnt(m, n, vacuum=10.0)
    tube_clean = cnt_geometry.clean_cnt_by_shift_wrap_anchor(tube_unit)
    tube = tube_clean * (1, 1, length)
    cnt_geometry.set_reference_cyl(tube)
    return tube


def test_scattering_only_defects_span_whole_tube():
    m, n = 5, 5
    r_max = 6.5
    l_def = 8
    T, N_uc, l_PL, _ = cnt_geometry.geo_info(m, n, r_max, l_def)

    tube = _build_tube(m, n, l_def)
    edge_margin = 1.42

    coords, indices = gen.generate_defect_coords(
        tube=tube,
        N=2,
        T=T,
        l_PL=l_PL,
        l_def=l_def,
        seed=20260705,
        total_length=l_def,
    )

    z_low = edge_margin
    z_high = l_def * T - edge_margin
    for _theta, z in coords:
        assert z_low <= z <= z_high, f"defect z={z} outside [{z_low}, {z_high}]"
    assert len(coords) == 2
    assert len(indices) == 2


def test_electrode_mode_defects_in_middle_region():
    m, n = 5, 5
    r_max = 6.5
    l_def = 8
    T, N_uc, l_PL, _ = cnt_geometry.geo_info(m, n, r_max, l_def)
    length = l_PL * 8 + l_def  # full 2PL-2PL-defects-2PL-2PL layout

    tube = _build_tube(m, n, length)
    edge_margin = 1.42

    coords, indices = gen.generate_defect_coords(
        tube=tube,
        N=2,
        T=T,
        l_PL=l_PL,
        l_def=l_def,
        seed=20260705,
        total_length=length,
    )

    z_low = 4 * l_PL * T + edge_margin
    z_high = (4 * l_PL + l_def) * T - edge_margin
    for _theta, z in coords:
        assert z_low <= z <= z_high, f"defect z={z} outside [{z_low}, {z_high}]"
    assert len(coords) == 2


def test_scattering_only_with_no_total_length_falls_back_to_electrode_mode():
    # total_length=None must behave like electrode mode (backwards compatible).
    m, n = 5, 5
    r_max = 6.5
    l_def = 8
    T, N_uc, l_PL, _ = cnt_geometry.geo_info(m, n, r_max, l_def)
    length = l_PL * 8 + l_def

    tube = _build_tube(m, n, length)
    edge_margin = 1.42

    coords, _ = gen.generate_defect_coords(
        tube=tube,
        N=2,
        T=T,
        l_PL=l_PL,
        l_def=l_def,
        seed=20260705,
        total_length=None,
    )

    z_low = 4 * l_PL * T + edge_margin
    z_high = (4 * l_PL + l_def) * T - edge_margin
    for _theta, z in coords:
        assert z_low <= z <= z_high
