"""Plan B（length_angstrom / defect_density -> l_def / N_defects）接口测试。

- batch_config: 长度/缺陷两维度各自二选一校验，目标值确定性 round，
  解析结果与手写 l_def/N_defects 完全等价（目标键不进下游 config）。
- batch_generate: build_tasks / make_task_config 产出与手写配置一致，
  provenance hash 与手写配置一致（等价配置复用同一目录与 seed）。
- 解析表 format_resolution_table：仅 Plan B 批次有输出，内容含目标值、
  解析结果与取整误差。
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


# (13,1) 的 UC 长度与 cnt_family_config 注释一致：19.209404 Å
UC_13_1 = 19.209404


def _write_batch(tmp_path, configs):
    import json

    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "temperatures": [100],
                "lammps_mode": "md",
                "configs": configs,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_uc_length_matches_known_value():
    assert bc.uc_length(13, 1) == pytest.approx(UC_13_1, abs=1e-6)
    assert bc.uc_length(9, 9) == pytest.approx(2.459512, abs=1e-6)


def test_plan_b_length_only(tmp_path):
    """length_angstrom 单独使用：长度 round 到最近 UC。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "length_angstrom": 230.0,
                "N_defects": 4,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    batch = bc.load(path)
    cfg = batch["configs"][0]
    # 230 / 19.209404 = 11.973 -> 12 UC，实际 230.51 Å
    assert cfg["l_def"] == 12
    assert cfg["N_defects"] == 4
    assert cfg["_resolution"] == {"length_angstrom": 230.0}


def test_plan_b_density_only(tmp_path):
    """defect_density 单独使用：密度按实际长度反推缺陷数。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "l_def": 12,
                "defect_density": 0.01736,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    batch = bc.load(path)
    cfg = batch["configs"][0]
    assert cfg["l_def"] == 12
    # 0.01736 * 12 * 19.209404 = 4.0008 -> 4 个缺陷
    assert cfg["N_defects"] == 4
    assert cfg["_resolution"] == {"defect_density": 0.01736}


def test_plan_b_both_dimensions(tmp_path):
    """两个维度同时 Plan B，可混合使用。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "length_angstrom": 1152.6,
                "defect_density": 0.01736,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    batch = bc.load(path)
    cfg = batch["configs"][0]
    # 1152.6 / 19.209404 = 60.0 -> 60 UC；0.01736*60*19.209404 = 20.005 -> 20
    assert cfg["l_def"] == 60
    assert cfg["N_defects"] == 20


def test_plan_b_deterministic_round(tmp_path):
    """round 用 int(x+0.5)：x.5 向上取整，无 banker's rounding。"""
    assert bc._round_positive(11.5) == 12
    assert bc._round_positive(12.5) == 13
    assert bc._round_positive(0.4) == 0
    # 小长度也至少 1 UC
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "length_angstrom": 1.0,
                "defect_density": 0.0001,
                "structures": ["MVH"],
                "replicas": 1,
            }
        ],
    )
    cfg = bc.load(path)["configs"][0]
    assert cfg["l_def"] == 1
    assert cfg["N_defects"] == 1


def test_plan_b_same_dimension_conflict(tmp_path):
    """同一维度两个键都写 -> 报错。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "l_def": 12,
                "length_angstrom": 230.0,
                "N_defects": 4,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    with pytest.raises(bc.BatchConfigError, match="长度维度"):
        bc.load(path)


def test_plan_b_dimension_missing(tmp_path):
    """同一维度都不写 -> 报错。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "N_defects": 4,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    with pytest.raises(bc.BatchConfigError, match="长度维度"):
        bc.load(path)

    path2 = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "l_def": 12,
                "structures": ["MVH"],
                "replicas": 60,
            }
        ],
    )
    with pytest.raises(bc.BatchConfigError, match="缺陷维度"):
        bc.load(path2)


def test_plan_b_invalid_target_values(tmp_path):
    """目标值必须为有限正数。"""
    for key in ("length_angstrom", "defect_density"):
        for bad in (0, -1.0, "abc", True):
            entry = {
                "chirality": [13, 1],
                "structures": ["MVH"],
                "replicas": 60,
                "l_def": 12,
                "N_defects": 4,
            }
            # 用目标键替换同维度的手写键
            if key == "length_angstrom":
                entry.pop("l_def")
            else:
                entry.pop("N_defects")
            entry[key] = bad
            path = _write_batch(tmp_path, [entry])
            with pytest.raises(bc.BatchConfigError):
                bc.load(path)


def test_plan_b_equivalent_to_handwritten(tmp_path):
    """Plan B 解析结果与手写 l_def/N_defects 的下游行为完全一致：

    任务展开（目录/seed/密度）相同，workflow config 相同，
    provenance hash 相同，且目标键不进入 workflow config。
    """
    plan_b = bc.load(
        _write_batch(
            tmp_path,
            [
                {
                    "chirality": [13, 1],
                    "length_angstrom": 230.0,
                    "defect_density": 0.017347,
                    "structures": ["MVH"],
                    "replicas": 2,
                }
            ],
        )
    )
    handwritten = bc.load(
        _write_batch(
            tmp_path,
            [
                {
                    "chirality": [13, 1],
                    "l_def": 12,
                    "N_defects": 4,
                    "structures": ["MVH"],
                    "replicas": 2,
                }
            ],
        )
    )

    tasks_a = batch_generate.build_tasks(plan_b, plan_b["data_root"])
    tasks_b = batch_generate.build_tasks(handwritten, handwritten["data_root"])
    assert len(tasks_a) == len(tasks_b) == 2
    for ta, tb in zip(tasks_a, tasks_b):
        assert ta["task_id"] == tb["task_id"]
        assert ta["density"] == tb["density"]
        assert ta["seed"] == tb["seed"]
        cfg_a = batch_generate.make_task_config(ta, plan_b)
        cfg_b = batch_generate.make_task_config(tb, handwritten)
        assert "length_angstrom" not in cfg_a
        assert "defect_density" not in cfg_a
        assert "_resolution" not in cfg_a
        assert cfg_a == cfg_b
        assert negf_provenance.compute_negf_config_hash(
            cfg_a
        ) == negf_provenance.compute_negf_config_hash(cfg_b)


def test_resolution_table_output(tmp_path):
    """解析表仅对 Plan B 配置输出，含目标值 / 解析结果 / 误差。"""
    path = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "length_angstrom": 230.0,
                "defect_density": 0.017347,
                "structures": ["MVH"],
                "replicas": 2,
            },
            {  # 手写维度在表中标注为"手写"
                "chirality": [9, 9],
                "l_def": 12,
                "defect_density": 0.14,
                "structures": ["MVH"],
                "replicas": 2,
            },
        ],
    )
    batch = bc.load(path)
    table = bc.format_resolution_table(batch["configs"])
    assert table is not None
    assert "Plan B 解析表" in table
    assert "(13,1)" in table and "l_def=12" in table
    assert "230.0" in table and "0.017347" in table
    assert "(9,9)" in table and "手写" in table

    # 纯手写批次没有解析表
    path2 = _write_batch(
        tmp_path,
        [
            {
                "chirality": [13, 1],
                "l_def": 12,
                "N_defects": 4,
                "structures": ["MVH"],
                "replicas": 2,
            }
        ],
    )
    assert bc.format_resolution_table(bc.load(path2)["configs"]) is None
