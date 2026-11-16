"""Exercise energy settings from batch input through actual solver templates."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stage")]
import batch_config
import batch_generate
import copy_input_dpnegf
from negf_provenance import compute_negf_config_hash


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("chirality,edges", [((8, 2), False), ((7, 0), True)])
@pytest.mark.parametrize("ef", [None, 0, -1.2, 1.6892654299736023])
def test_solver_input(tmp_path, monkeypatch, device, chirality, edges, ef):
    template = ROOT / "input_files" / f"{device}_cluster" / "dpnegf" / "input.json"
    original = json.loads(template.read_text(encoding="utf-8"))
    (tmp_path / "input.json").write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "STRUCT.fdf").write_text("fixture", encoding="utf-8")
    (tmp_path / "run.py").write_text('model_path = "old"\nstructure = "old"\n', encoding="utf-8")
    monkeypatch.setattr(copy_input_dpnegf, "count_atoms_from_struct_fdf", lambda _: 100)
    monkeypatch.setattr(copy_input_dpnegf.cnt_geometry, "geo_info", lambda *args: (4.0, 2, 1, 50))
    kwargs = {} if ef is None else {"e_fermi": ef}
    copy_input_dpnegf.update_input_json_for_leaf(
        tmp_path, "model.pth", chirality, {"use_saved": False, "save_path": "./cache"},
        ["Ev", "Ec"] if edges else ["Ef"], **kwargs,
    )
    result = json.loads((tmp_path / "input.json").read_text(encoding="utf-8"))["task_options"]
    assert result["stru_options"]["compute_band_edges"] is edges
    if ef is None:
        assert "e_fermi" not in result
    else:
        assert result["e_fermi"] == ef
    for key in ("rgf_options", "energy_grid", "ele_T"):
        assert result[key] == original["task_options"][key]


def raw_batch(mode):
    raw = {"temperatures": [100], "template": {},
           "configs": [{"m": 8, "n": 2, "l_def": 3, "N_defects": 1,
                        "structures": ["MVH"], "replicas": 1}]}
    if mode == "assemble":
        source = str(ROOT / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR")
        raw["structure_mode"] = "assemble"
        raw["configs"] = [{"chirality": [8, 2], "perfect_module": source,
                           "defect_modules": {"X": [source, 1]}, "defect_counts": {"X": 1},
                           "scatter_uc_count": 3, "buffer_pl": 2,
                           "rotation_angles_deg": [0], "replicas": 1}]
    return raw


@pytest.mark.parametrize("mode", ["normal", "assemble"])
@pytest.mark.parametrize("template,entry,expected", [
    ({}, {}, None), ({"e_fermi": 1.5}, {}, 1.5),
    ({"e_fermi": 1.5}, {"e_fermi": 0}, 0),
    ({"e_fermi": 1.5}, {"e_fermi": -2}, -2),
    ({"e_fermi": 1.5}, {"e_fermi": None}, None),
])
def test_batch_to_workflow(tmp_path, mode, template, entry, expected):
    raw = raw_batch(mode)
    raw["template"].update(template)
    raw["configs"][0].update(entry)
    batch = batch_config._validate_batch(raw, tmp_path / "batch.json")
    task = batch_generate.build_tasks(batch, tmp_path)[0]
    config = batch_generate.make_task_config(task, batch)
    assert config["e_fermi"] == expected


@pytest.mark.parametrize("mode", ["normal", "assemble"])
@pytest.mark.parametrize("level", ["template", "entry"])
@pytest.mark.parametrize("value", [True, "1.5", float("nan"), float("inf"), float("-inf")])
def test_invalid_configuration(tmp_path, mode, level, value):
    raw = raw_batch(mode)
    target = raw["template"] if level == "template" else raw["configs"][0]
    target["e_fermi"] = value
    with pytest.raises(ValueError, match="e_fermi"):
        batch_config._validate_batch(raw, tmp_path / "batch.json")


def test_provenance_rejects_old_policy_and_changed_fermi():
    base = {"chirality": [8, 2]}
    old_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    automatic = compute_negf_config_hash(base)
    assert automatic != old_hash
    assert automatic == compute_negf_config_hash({**base, "e_fermi": None})
    assert automatic != compute_negf_config_hash({**base, "e_fermi": 0})
    assert compute_negf_config_hash({**base, "e_fermi": 1}) == compute_negf_config_hash({**base, "e_fermi": 1.0})
    assert compute_negf_config_hash({**base, "e_fermi": 1}) != compute_negf_config_hash({**base, "e_fermi": 2})
