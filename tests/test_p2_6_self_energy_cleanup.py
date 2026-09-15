"""Tests for P2-6: self-energy cleanup must use configured path, not just output/self_energy."""
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_multi  # noqa: E402


def test_cleans_leaf_level_self_energy_dir(tmp_path):
    # Simulate: leaf/self_energy/ (configured path ./self_energy/) + leaf/output/HS_device.h5
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    se_dir = leaf / "self_energy"
    se_dir.mkdir()
    (se_dir / "cache.h5").write_bytes(b"data")
    out = leaf / "output"
    out.mkdir()
    (out / "HS_device.h5").write_bytes(b"data")

    run_multi.clean_dpnegf_output_cache(
        tmp_path, save_self_energy=False, self_energy_save_path="./self_energy/"
    )

    assert not se_dir.exists()       # configured path cleaned
    assert not (out / "HS_device.h5").exists()


def test_preserves_when_save_self_energy_true(tmp_path):
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    se_dir = leaf / "self_energy"
    se_dir.mkdir()
    (se_dir / "cache.h5").write_bytes(b"data")
    out = leaf / "output"
    out.mkdir()
    (out / "HS_device.h5").write_bytes(b"data")

    run_multi.clean_dpnegf_output_cache(
        tmp_path, save_self_energy=True, self_energy_save_path="./self_energy/"
    )

    assert se_dir.exists()              # preserved
    # HS files are cleaned even when self_energy is preserved.
    assert not (out / "HS_device.h5").exists()


def test_default_cleans_output_self_energy(tmp_path):
    # No configured path -> still clean output/self_energy (legacy).
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    out = leaf / "output"
    out.mkdir()
    se_dir = out / "self_energy"
    se_dir.mkdir()
    (se_dir / "cache.h5").write_bytes(b"data")

    run_multi.clean_dpnegf_output_cache(tmp_path, save_self_energy=False)

    assert not se_dir.exists()


def test_no_se_path_no_crash(tmp_path):
    run_multi.clean_dpnegf_output_cache(
        tmp_path, save_self_energy=False, self_energy_save_path=None
    )
