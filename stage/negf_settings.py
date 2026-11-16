"""Validation shared by batch configuration and standalone DPNEGF preparation."""
import math


def normalize_e_fermi(value):
    """None requests recalculation; any finite real number is a fixed Ef (eV)."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"e_fermi 必须为有限数值（eV）或 None/null，当前值: {value!r}")
    return float(value)
