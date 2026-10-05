#!/usr/bin/env python3
"""
批次配置（batch.json）的加载、校验与任务展开。

batch.json 是批任务的唯一配置入口，取代旧版 batch_generate.py 内的
TEMPLATE / CHIRAL_CONFIGS / TEMPERATURES / BASE_*_SEED 全局常量。

文件格式
--------
{
  "template": {           # 所有物理配置的公共默认值（可选）
    "r_max": 6.5,
    "md_steps": 50000,
    "save_self_energy": false,
    "self_energy_cache": {"use_saved": true, "save_path": "./self_energy/"},
    "md_sampling": {"n_samples": 1}
  },
  "temperatures": [500],
  "lammps_mode": "opt",          # "md" NVT 退火 / "opt" 几何优化
  "base_structure_seed": 20260705,
  "base_lammps_seed": 23456789,
  "data_root": ".",              # 相对路径按 batch.json 所在目录解析
  "configs": [
    {
      "chirality": [5, 5],
      "l_def": 8,
      "N_defects": 2,
      "structures": ["5775"],
      "conductance_mode": "band_edge_bias",   # fermi / band_edge_bias
      "replicas": 20            # 必填：每个物理配置显式声明自己的
                                # replica 数，不同长度可设不同值
    }
  ]
}

确定性约定
----------
任务枚举顺序固定为 configs 顺序 × temperatures 顺序 × replica 序号。
seed = base_seed + task_index，同一物理目录（由 chirality/l_def/
N_defects/structures/密度/温度决定）的 replica 序号固定，与 replicas
总数无关：调大某个 config 的 replicas 只会追加新 replica，不影响已有
replica 的 seed 与目录。
"""

from __future__ import annotations

import json
from pathlib import Path


# template 层的公共物理默认值。旧版硬编码在 batch_generate.py 的
# TEMPLATE 全局常量中；JSON 的 template 字段可逐项覆盖。
DEFAULT_TEMPLATE = {
    "r_max": 6.5,
    "md_steps": 50000,
    "save_self_energy": False,
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
    # 每条独立 LAMMPS 轨迹只取最后一帧做 DPNEGF。
    "md_sampling": {"n_samples": 1},
}

CONDUCTANCE_MODES = ("fermi", "band_edge_bias")
LAMMPS_MODES = ("md", "opt")

_REQUIRED_CONFIG_KEYS = (
    "chirality",
    "l_def",
    "N_defects",
    "structures",
    "replicas",
)


class BatchConfigError(ValueError):
    """batch.json 内容不合法。"""


def _fail(path, msg):
    raise BatchConfigError(f"{path}: {msg}")


def _require_int(value, name, path, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"{name} 必须为不小于 {minimum} 的整数，当前值: {value!r}")
    return value


def _require_bool(value, name, path):
    if not isinstance(value, bool):
        _fail(path, f"{name} 必须为 true/false，当前值: {value!r}")
    return value


def validate_config_entry(entry, index, path):
    """校验单个物理配置条目，返回补齐 chirality 后的新 dict。"""
    if not isinstance(entry, dict):
        _fail(path, f"configs[{index}] 必须为对象，当前值: {entry!r}")

    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in entry]
    if missing:
        _fail(path, f"configs[{index}] 缺少必需字段: {', '.join(missing)}")

    chirality = entry["chirality"]
    if (
        not isinstance(chirality, list)
        or len(chirality) != 2
        or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                   for x in chirality)
    ):
        _fail(
            path,
            f"configs[{index}].chirality 必须为两个正整数，如 [5, 5]，"
            f"当前值: {chirality!r}",
        )
    m, n = chirality

    l_def = _require_int(entry["l_def"], f"configs[{index}].l_def", path)
    n_defects = _require_int(
        entry["N_defects"], f"configs[{index}].N_defects", path
    )

    structures = entry["structures"]
    if (
        not isinstance(structures, list)
        or not structures
        or not all(isinstance(s, str) and s for s in structures)
    ):
        _fail(
            path,
            f"configs[{index}].structures 必须为非空字符串列表，"
            f"如 [\"5775\"]，当前值: {structures!r}",
        )

    conductance_mode = entry.get("conductance_mode", "fermi")
    if conductance_mode not in CONDUCTANCE_MODES:
        _fail(
            path,
            f"configs[{index}].conductance_mode 必须为 "
            f"{' / '.join(CONDUCTANCE_MODES)}，当前值: {conductance_mode!r}",
        )

    out = dict(entry)
    out["m"] = m
    out["n"] = n
    out["l_def"] = l_def
    out["N_defects"] = n_defects
    out["structures"] = list(structures)
    out["conductance_mode"] = conductance_mode
    # replicas 是每个物理配置的必填显式字段（不接受全局默认），
    # 杜绝"忘了写就静默用全局值"的歧义。
    out["replicas"] = _require_int(
        entry["replicas"], f"configs[{index}].replicas", path
    )
    return out


def load_batch_config(path):
    """
    加载并校验 batch.json，返回补齐默认值后的配置 dict：

        {
          "template": {...},          # 公共物理参数（默认值 + template 覆盖）
          "temperatures": [int, ...],
          "lammps_mode": "md"|"opt",
          "replicas": int,            # 全局默认 replica 数
          "base_structure_seed": int,
          "base_lammps_seed": int,
          "data_root": Path,          # 已解析为绝对路径
          "configs": [ {...}, ... ],  # 每项已校验并补齐 m/n 等字段
        }

    所有格式问题在加载时一次性报出（fail-fast），不等到任务展开中途。
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"batch 配置文件不存在: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(path, f"JSON 解析失败: {exc}")

    if not isinstance(raw, dict):
        _fail(path, "顶层必须为 JSON 对象")

    template_raw = raw.get("template", {})
    if not isinstance(template_raw, dict):
        _fail(path, f"template 必须为对象，当前值: {template_raw!r}")
    template = {**DEFAULT_TEMPLATE, **template_raw}
    _require_bool(template["save_self_energy"], "template.save_self_energy", path)

    temperatures = raw.get("temperatures")
    if (
        not isinstance(temperatures, list)
        or not temperatures
        or not all(isinstance(t, int) and not isinstance(t, bool) and t > 0
                   for t in temperatures)
    ):
        _fail(
            path,
            f"temperatures 必须为非空正整数列表，如 [300, 500]，"
            f"当前值: {temperatures!r}",
        )

    lammps_mode = raw.get("lammps_mode", "md")
    if lammps_mode not in LAMMPS_MODES:
        _fail(
            path,
            f"lammps_mode 必须为 {' / '.join(LAMMPS_MODES)}，"
            f"当前值: {lammps_mode!r}",
        )

    base_structure_seed = _require_int(
        raw.get("base_structure_seed", 20260705), "base_structure_seed", path
    )
    base_lammps_seed = _require_int(
        raw.get("base_lammps_seed", 23456789), "base_lammps_seed", path
    )

    data_root_raw = raw.get("data_root", ".")
    if not isinstance(data_root_raw, str) or not data_root_raw:
        _fail(path, f"data_root 必须为非空字符串，当前值: {data_root_raw!r}")
    # 相对路径按 batch.json 所在目录解析，保证从任意 cwd 启动行为一致。
    data_root = Path(data_root_raw).expanduser()
    if not data_root.is_absolute():
        data_root = (path.parent / data_root).resolve()

    configs_raw = raw.get("configs")
    if not isinstance(configs_raw, list) or not configs_raw:
        _fail(path, "configs 必须为非空列表")

    configs = [
        validate_config_entry(entry, i, path)
        for i, entry in enumerate(configs_raw)
    ]

    return {
        "template": template,
        "temperatures": list(temperatures),
        "lammps_mode": lammps_mode,
        "base_structure_seed": base_structure_seed,
        "base_lammps_seed": base_lammps_seed,
        "data_root": data_root,
        "configs": configs,
    }
