"""An external UC must control both batch rounding and the expanded geometry."""

import json
import sys
from pathlib import Path

import pytest


STAGE = Path(__file__).resolve().parents[1] / "stage"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

import batch_config as bc  # noqa: E402
from uc_source import read_poscar_metadata  # noqa: E402


def _write_poscar(path, period=2.5, count=20):
    path.write_text(
        "relaxed CNT UC\n1.0\n20 0 0\n0 20 0\n"
        f"0 0 {period}\nC\n{count}\nCartesian\n"
        + "".join(f"10 10 {i * period / count}\n" for i in range(count)),
        encoding="utf-8",
    )


def test_external_uc_controls_length_rounding_and_identity(tmp_path):
    uc_path = tmp_path / "5_5" / "POSCAR"
    uc_path.parent.mkdir()
    _write_poscar(uc_path)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({
        "template": {"uc_file": "{chirality}/POSCAR"},
        "temperatures": [100],
        "configs": [{
            "chirality": [5, 5],
            "length_angstrom": 10.0,
            "defect_density": 0.05,
            "structures": ["5775"],
            "replicas": 20,
        }],
    }), encoding="utf-8")

    cfg = bc.load(batch_path)["configs"][0]
    assert cfg["l_def"] == 4
    assert cfg["uc_length_angstrom"] == 2.5
    assert cfg["uc_file"] == str(uc_path.resolve())
    assert cfg["uc_sha256"] == read_poscar_metadata(uc_path)["sha256"]
    assert cfg["N_defects"] == 1  # default same_count: half-up from 0.5


def test_external_uc_geometry_uses_file_period(tmp_path):
    pytest.importorskip("ase")
    import cnt_geometry

    uc_path = tmp_path / "POSCAR"
    _write_poscar(uc_path)
    meta = read_poscar_metadata(uc_path)
    unit = cnt_geometry.build_unit_cnt(5, 5, uc_file=uc_path,
                                       uc_sha256=meta["sha256"])
    period, atoms, _, _ = cnt_geometry.geo_info(
        5, 5, 6.5, 4, uc_file=uc_path, uc_sha256=meta["sha256"]
    )
    assert len(unit) == atoms == 20
    assert period == pytest.approx(2.5)
    assert unit.cell[2, 2] == pytest.approx(2.5)

    _write_poscar(uc_path, period=2.6)
    with pytest.raises(ValueError, match="发生变化"):
        cnt_geometry.geo_info(
            5, 5, 6.5, 4, uc_file=uc_path, uc_sha256=meta["sha256"]
        )


def test_external_uc_rejects_wrong_atom_count(tmp_path):
    uc_path = tmp_path / "POSCAR"
    _write_poscar(uc_path, count=19)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({
        "temperatures": [100],
        "configs": [{
            "chirality": [5, 5], "uc_file": "POSCAR", "l_def": 4,
            "N_defects": 1, "structures": ["5775"], "replicas": 1,
        }],
    }), encoding="utf-8")
    with pytest.raises(bc.BatchConfigError, match="20 个 C 原子"):
        bc.load(batch_path)


def test_external_uc_rejects_axial_vacuum(tmp_path):
    uc_path = tmp_path / "POSCAR"
    _write_poscar(uc_path)
    original = uc_path.read_text(encoding="utf-8")
    uc_path.write_text(original.replace("0 0 2.5\n", "0 0 12.5\n"),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="轴向存在"):
        read_poscar_metadata(uc_path)


def test_shared_cache_requires_uc_and_electronic_temperature(tmp_path):
    cache = {
        "use_saved": True,
        "save_path": "{data_root}/self_energy/{chirality}/",
    }
    with pytest.raises(bc.BatchConfigError, match="uc_sha256"):
        bc.resolve_self_energy_cache(
            cache, m=5, n=5, data_root=tmp_path,
            uc_sha256="a" * 64, ele_T=100,
        )
    cache["save_path"] += "{uc_sha256}/eleT_{ele_T}/"
    resolved = bc.resolve_self_energy_cache(
        cache, m=5, n=5, data_root=tmp_path,
        uc_sha256="a" * 64, ele_T=100,
    )
    assert "/" + "a" * 64 + "/eleT_100/" in resolved["save_path"].replace("\\", "/")
