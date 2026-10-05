"""Per-chirality self_energy_cache 接口测试。

不同手性的电极自能互不相同，批次中每个手性应能指向自己的
self_energy 缓存文件（同手性的所有长度 / replica 共享一份）：

- batch_config: template / per-config self_energy_cache 校验，
  save_path 占位符解析（{m} {n} {chirality} {data_root}）。
- batch_generate: 任务展开后 workflow config 带按手性解析后的路径，
  该字段进入 provenance hash（NEGF_CONFIG_KEYS 已含 self_energy_cache）。
- run_multi: 绝对路径共享缓存不被自动清理，相对路径行为不变。
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_DIR = _REPO_ROOT / "stage"
for _p in (_REPO_ROOT, _STAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import batch_config as bc  # noqa: E402
import batch_generate  # noqa: E402
import negf_provenance  # noqa: E402
import run_multi  # noqa: E402


def _minimal_batch(tmp_path):
    return {
        "template": {
            "r_max": 6.5,
            "md_steps": 0,
            "save_self_energy": True,
            "self_energy_cache": {
                "use_saved": True,
                "save_path": "{data_root}/self_energy/{chirality}/",
            },
            "md_sampling": {"n_samples": 1},
        },
        "temperatures": [500],
        "lammps_mode": "opt",
        "base_structure_seed": 20260705,
        "base_lammps_seed": 23456789,
        "data_root": tmp_path,
        "configs": [
            {
                "chirality": [13, 1],
                "m": 13,
                "n": 1,
                "l_def": 3,
                "N_defects": 1,
                "structures": ["MVH"],
                "replicas": 2,
                "conductance_mode": "band_edge_bias",
                "self_energy_cache": None,
            },
            {
                "chirality": [12, 3],
                "m": 12,
                "n": 3,
                "l_def": 3,
                "N_defects": 1,
                "structures": ["MVH"],
                "replicas": 1,
                "conductance_mode": "band_edge_bias",
                "self_energy_cache": None,
            },
        ],
    }


def test_resolve_placeholders():
    cache = {"use_saved": True, "save_path": "{data_root}/se/{chirality}/"}
    resolved = bc.resolve_self_energy_cache(
        cache, m=13, n=1, data_root=Path("/data/root")
    )
    assert resolved == {
        "use_saved": True,
        "save_path": "/data/root/se/13_1/",
    }


def test_resolve_keeps_relative_path_relative():
    cache = {"use_saved": False, "save_path": "./self_energy/"}
    resolved = bc.resolve_self_energy_cache(
        cache, m=9, n=9, data_root=Path("/data/root")
    )
    assert resolved["save_path"] == "./self_energy/"


def test_resolve_unknown_placeholder_fails():
    cache = {"use_saved": True, "save_path": "./se_{foo}/"}
    with pytest.raises(bc.BatchConfigError):
        bc.resolve_self_energy_cache(cache, m=9, n=9, data_root=Path("/x"))


def test_validate_per_config_override():
    entry = {
        "chirality": [13, 1],
        "l_def": 3,
        "N_defects": 1,
        "structures": ["MVH"],
        "replicas": 5,
        "self_energy_cache": {"use_saved": True, "save_path": "/shared/se/"},
    }
    out = bc.validate_config_entry(entry, 0, Path("batch.json"))
    assert out["self_energy_cache"] == {"use_saved": True, "save_path": "/shared/se/"}

    no_override = dict(entry)
    del no_override["self_energy_cache"]
    out = bc.validate_config_entry(no_override, 0, Path("batch.json"))
    assert out["self_energy_cache"] is None

    bad = dict(entry)
    bad["self_energy_cache"] = {"use_saved": True, "save_path": ""}
    with pytest.raises(bc.BatchConfigError):
        bc.validate_config_entry(bad, 0, Path("batch.json"))


def test_tasks_get_per_chirality_cache_path(tmp_path):
    batch = _minimal_batch(tmp_path)
    tasks = batch_generate.build_tasks(batch=batch, base_root=batch["data_root"])

    by_chirality = {}
    for task in tasks:
        key = (task["m"], task["n"])
        by_chirality.setdefault(key, []).append(task["self_energy_cache"])

    expected_13_1 = {
        "use_saved": True,
        "save_path": f"{tmp_path}/self_energy/13_1/",
    }
    expected_12_3 = {
        "use_saved": True,
        "save_path": f"{tmp_path}/self_energy/12_3/",
    }

    # 同手性的所有 replica 共享同一路径；不同手性路径不同。
    assert all(c == expected_13_1 for c in by_chirality[(13, 1)])
    assert all(c == expected_12_3 for c in by_chirality[(12, 3)])
    assert expected_13_1["save_path"] != expected_12_3["save_path"]

    # 展开进 workflow_config（进入 provenance hash 的字段集合）。
    config = batch_generate.make_task_config(
        tasks[0], _minimal_batch(tmp_path)
    )
    assert config["self_energy_cache"] == expected_13_1
    assert "self_energy_cache" in negf_provenance.NEGF_CONFIG_KEYS


def test_per_config_override_wins_over_template(tmp_path):
    batch = _minimal_batch(tmp_path)
    override = {"use_saved": False, "save_path": "./self_energy/"}
    batch["configs"][0]["self_energy_cache"] = override

    tasks = batch_generate.build_tasks(batch=batch, base_root=batch["data_root"])
    caches = {
        (t["m"], t["n"]): t["self_energy_cache"] for t in tasks
    }
    # per-config 覆盖的相对路径保持相对语义（leaf 内缓存）。
    assert caches[(13, 1)] == override
    # 未覆盖的 config 仍用 template + 占位符解析。
    assert caches[(12, 3)]["save_path"] == f"{tmp_path}/self_energy/12_3/"


def test_absolute_shared_cache_not_auto_cleaned(tmp_path):
    leaf = tmp_path / "replica_001" / "dpnegf" / "1"
    (leaf / "output" / "self_energy").mkdir(parents=True)
    shared = tmp_path / "self_energy" / "13_1"
    shared.mkdir(parents=True)

    run_multi.clean_dpnegf_output_cache(
        tmp_path,
        save_self_energy=False,
        self_energy_save_path=str(shared),
    )

    # leaf 内的 output/self_energy 照常清理；
    # 绝对路径共享缓存必须保留（由用户管理）。
    assert not (leaf / "output" / "self_energy").exists()
    assert shared.is_dir()


def test_relative_cache_still_cleaned(tmp_path):
    leaf = tmp_path / "replica_001" / "dpnegf" / "1"
    (leaf / "output").mkdir(parents=True)
    (leaf / "self_energy").mkdir(parents=True)

    run_multi.clean_dpnegf_output_cache(
        tmp_path,
        save_self_energy=False,
        self_energy_save_path="./self_energy/",
    )

    assert not (leaf / "self_energy").exists()
