"""Tests for P2-3: collector must cross-check replica config consistency."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_md_conductance as cmc  # noqa: E402


def test_config_mismatches_detects_difference():
    a = {"temperature": 300, "chirality": [5, 5], "l_def": 5}
    b = {"temperature": 500, "chirality": [5, 5], "l_def": 5}
    m = cmc._config_mismatches(a, b, ["temperature", "chirality", "l_def"])
    assert len(m) == 1
    assert "temperature" in m[0]


def test_config_mismatches_identical_returns_empty():
    a = {"temperature": 300, "chirality": [5, 5]}
    m = cmc._config_mismatches(a, a, ["temperature", "chirality"])
    assert m == []


def test_config_mismatches_missing_key_both():
    # Both missing -> considered equal (None == None).
    a = {"temperature": 300}
    b = {"temperature": 300}
    m = cmc._config_mismatches(a, b, ["Ec_eV"])
    assert m == []


def test_config_mismatches_one_missing():
    a = {"Ec_eV": 0.3}
    b = {}
    m = cmc._config_mismatches(a, b, ["Ec_eV"])
    assert len(m) == 1
    assert "Ec_eV" in m[0]
