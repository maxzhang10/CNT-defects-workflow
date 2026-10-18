"""Validation and canonical form for defect anchor azimuth ranges."""
import math


def normalize_angle_range(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError("defect_angle_range_deg must contain two finite numbers")
    start, end = map(float, value)
    width = end - start
    if not 0 < width <= 360:
        raise ValueError("defect_angle_range_deg requires 0 < end - start <= 360")
    if width == 360:
        return [0.0, 360.0]
    start %= 360
    return [start, start + width]
