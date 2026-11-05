"""Tests for P2-4 (triclinic box/origin) and P2-5 (Bohr unit conversion)."""
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import dump2fdf_batch as d2f  # noqa: E402


# --- P2-4: triclinic box bounds ---

def test_orthorhombic_box_bounds_unchanged():
    box_lines = "0.0 10.0\n0.0 10.0\n0.0 10.0\n"
    H, tilts, origin = d2f.read_box_bounds(io.StringIO(box_lines))
    assert tilts == (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    assert np.allclose(origin, [0.0, 0.0, 0.0])
    assert np.allclose(np.diag(H), [10.0, 10.0, 10.0])


def test_triclinic_bounds_recover_true_box():
    # LAMMPS dump: xlo_bound xhi_bound xy / ylo_bound yhi_bound xz / zlo_bound zhi_bound yz
    # Use a known example: true box xlo=0,xhi=10,xy=2 -> bounds shift.
    # For xy=2: xlo_bound = xlo - min(0,xy) = 0 - 0 = 0; xhi_bound = xhi - max(0,xy) = 10-2=8
    # Actually LAMMPS: xlo_bound = xlo + min(0,xy,xz,xy+xz); so reverse: xlo = xlo_bound - min(0,...)
    # Let's just verify round-trip with small tilt.
    box_lines = "0.0 10.0 2.0\n0.0 10.0 0.0\n0.0 10.0 0.0\n"
    H, tilts, origin = d2f.read_box_bounds(io.StringIO(box_lines))
    xy, xz, yz, Lx, Ly, Lz = tilts
    # With only xy=2 nonzero: xlo = 0 + min(0,2,0,2) = 0; xhi = 10 + max(0,2,0,2) = 12
    assert np.isclose(Lx, 12.0)  # true Lx = xhi - xlo = 12 - 0
    assert np.isclose(Ly, 10.0)
    assert np.isclose(Lz, 10.0)
    assert np.isclose(origin[0], 0.0)
    # b vector has xy tilt.
    assert np.isclose(H[0, 1], 2.0)


def test_wrap_with_nonzero_origin():
    # Box from 5 to 15 in each direction; origin=(5,5,5).
    H = np.diag([10.0, 10.0, 10.0])
    origin = np.array([5.0, 5.0, 5.0])
    # A point at 16.0 in x should wrap to 6.0 (i.e., 5 + 1).
    pos = np.array([[16.0, 5.0, 5.0]])
    wrapped = d2f.wrap_positions_cartesian(pos, H, origin=origin)
    assert np.isclose(wrapped[0, 0], 6.0)


def test_wrap_without_origin_legacy():
    # origin=None -> old behavior (no shift), for backward compat.
    H = np.diag([10.0, 10.0, 10.0])
    pos = np.array([[16.0, 0.0, 0.0]])
    wrapped = d2f.wrap_positions_cartesian(pos, H, origin=None)
    assert np.isclose(wrapped[0, 0], 6.0)  # 16 mod 10 = 6


# --- P2-5: Bohr unit conversion ---

def test_bohr_output_scales_coordinates(tmp_path):
    H = np.diag([10.0, 10.0, 10.0])  # Å
    atoms = [(1, 1, 5.0, 5.0, 5.0)]  # Å
    out = tmp_path / "test.fdf"

    d2f.write_siesta_fdf(str(out), H, atoms, coord_unit="Bohr", wrap=False)
    text = out.read_text(encoding="utf-8")

    # 5.0 Å / 0.529177... = ~9.4486306 Bohr
    bohr = 0.529177210903
    expected = 5.0 / bohr
    # The output uses %.8f; check the value appears.
    assert f"{expected:.8f}" in text or f"{expected:.7f}" in text or "9.4486306" in text


def test_ang_output_no_scaling(tmp_path):
    H = np.diag([10.0, 10.0, 10.0])
    atoms = [(1, 1, 5.0, 5.0, 5.0)]
    out = tmp_path / "test.fdf"

    d2f.write_siesta_fdf(str(out), H, atoms, coord_unit="Ang", wrap=False)
    text = out.read_text(encoding="utf-8")

    assert "5.00000000" in text


def test_restore_wrapped_first_pl_without_rounding_or_touching_right_lead(tmp_path):
    lz = 81.16390017
    dump = tmp_path / "optimized.dump"
    dump.write_text(
        "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n4\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n"
        f"0 {lz}\nITEM: ATOMS id type x y z\n"
        f"4 1 1 1 {lz - 0.000001}\n"
        "2 1 1 1 7.37853645\n"
        "3 1 1 1 14.7570729\n"
        f"1 1 1 1 {lz - 0.000002}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    written = d2f.extract_from_one_dump(
        str(dump), str(out), every=1, coord_unit="Ang", wrap=False,
        start=None, end=None, skip_zero=False, samples=1, left_pl_atoms=2,
    )
    assert written == 1
    lines = (out / "1" / "STRUCT.fdf").read_text(encoding="utf-8").splitlines()
    coordinates = [line.split() for line in lines if "# id " in line]
    np.testing.assert_allclose(
        [float(line[2]) for line in coordinates],
        [-0.000002, 7.37853645, 14.7570729, lz - 0.000001],
        rtol=0, atol=1e-8,
    )
