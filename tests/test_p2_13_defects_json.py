"""Tests for P2-13: defect composition is persisted to defects.json."""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ele_multi_defects_ele as ele  # noqa: E402
import cnt_geometry  # noqa: E402


def _make_tube():
    """Build a real CNT and set its cylindrical reference axis."""
    tube = cnt_geometry.build_unit_cnt(6, 6)
    cnt_geometry.set_reference_cyl(tube)
    return tube


def test_defect_log_is_json_serializable():
    """defect_log entries must be JSON-serializable for defects.json."""
    tube = _make_tube()
    coords = [(0.0, 5.0), (1.5, 10.0)]
    type_list = ["DV", "MVH"]

    _, defect_log = ele.multi_defects_ele(tube, coords, type_list, seed=42)

    # Should serialize without error.
    serialized = json.dumps(defect_log)
    restored = json.loads(serialized)
    assert len(restored) == 2

    for item in defect_log:
        assert "no" in item
        assert "type" in item
        assert "theta" in item
        assert "z" in item
        assert "index_when_created" in item
        assert item["type"] in type_list


def test_defect_log_matches_actual_choices():
    """Same seed should produce same defect type choices."""
    coords = [(0.0, 5.0), (1.5, 10.0)]
    type_list = ["DV", "MVH", "5775"]

    tube1 = _make_tube()
    _, log1 = ele.multi_defects_ele(tube1, coords, type_list, seed=99)

    tube2 = _make_tube()
    _, log2 = ele.multi_defects_ele(tube2, coords, type_list, seed=99)

    types1 = [d["type"] for d in log1]
    types2 = [d["type"] for d in log2]
    assert types1 == types2
