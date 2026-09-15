"""Tests for P2-11: coverage normalization must be continuous.

The old code divided kernel by coverage only when coverage < 0.995, causing
a discontinuity at the threshold. Now normalization is always applied.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import collect_md_conductance as cmc  # noqa: E402


def _call_interpolate(energy, transmission, e_fermi, kbt):
    """Call interpolate_conductance with a fine eval grid."""
    return cmc.interpolate_conductance(
        energy, transmission, e_fermi, kbt
    )


def test_constant_transmission_gives_close_to_constant():
    """T(E)=1 should give G/G0 ≈ 1 for wide energy range at any temperature."""
    # Wide energy range -> coverage ~1.
    energy = np.linspace(-1.0, 1.0, 10001)
    transmission = np.ones_like(energy)
    kbt = 0.02585  # 300 K
    g = _call_interpolate(energy, transmission, 0.0, kbt)
    assert abs(g - 1.0) < 1e-3


def test_no_discontinuity_at_threshold():
    """Same T(E), slightly different energy ranges crossing 0.995 coverage
    should give close results (no jump)."""
    kbt = 0.02585
    # Narrower range -> coverage just below/above 0.995.
    e_wide = np.linspace(-0.5, 0.5, 50001)
    e_narrow = np.linspace(-0.45, 0.45, 45001)
    t = np.ones_like(e_wide)
    t_narrow = np.ones_like(e_narrow)

    g_wide = _call_interpolate(e_wide, t, 0.0, kbt)
    g_narrow = _call_interpolate(e_narrow, t_narrow, 0.0, kbt)

    # With unified normalization, both should be close to 1 (not a jump).
    assert abs(g_wide - 1.0) < 0.01
    assert abs(g_narrow - 1.0) < 0.01
    # No large discontinuity between the two.
    assert abs(g_wide - g_narrow) < 0.02
