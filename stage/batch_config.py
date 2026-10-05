#!/usr/bin/env python3
"""
批次配置（batch.json 或 batch.py）的加载、校验与任务展开。

批任务是配置驱动的：配置文件取代旧版 batch_generate.py 内的
TEMPLATE / CHIRAL_CONFIGS / TEMPERATURES / BASE_*_SEED 全局常量。

支持两种格式（按文件后缀分派，校验规则完全一致）：

batch.json —— 简单配置、工具生成时使用：

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
  "data_root": ".",              # 相对路径按配置文件所在目录解析
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

batch.py —— 配置项多、需要循环/变量/条件时使用（同名变量赋值即可）：

    configs = [
        {
            "chirality": [5, 5],
            "l_def": l,
            "N_defects": 2,
            "structures": ["5775"],
            "conductance_mode": "band_edge_bias",
            "replicas": 20 if l <= 8 else 40,
        }
        for l in (5, 8, 12)
    ]
    temperatures = [500]
    lammps_mode = "opt"
    base_structure_seed = 20260705
    base_lammps_seed = 23456789
    data_root = "."
    template = {"r_max": 6.5}

    # 只写数据生成逻辑（循环/变量/算术），命名空间为白名单，
    # 不注入 import / open 等能力。

确定性约定
----------
任务枚举顺序固定为 configs 顺序 × temperatures 顺序 × replica 序号。
seed = base_seed + config 内 replica 序号，同一物理目录（由
chirality/l_def/N_defects/structures/密度/温度决定）的 replica 序号
固定，与 replicas 总数无关：调大某个 config 的 replicas 只会追加新
replica，不影响已有 replica 的 seed 与目录。
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


# batch.py 加载时注入的白名单命名空间：只放数据生成用到的工具，
# 不给 import / open / exec 等能力。配置文件是数据不是脚本。
_PY_BUILTINS_ALLOW = (
    "dict", "list", "tuple", "set", "int", "float", "str", "bool",
    "len", "range", "enumerate", "zip", "sorted", "min", "max",
    "sum", "abs", "round",
)

# batch.py 中会被收集的顶层变量名（其余变量如循环临时量自动忽略）。
_BATCH_PY_KEYS = (
    "template",
    "temperatures",
    "lammps_mode",
    "base_structure_seed",
    "base_lammps_seed",
    "data_root",
    "configs",
)


def _validate_batch(raw, path):
    """
    校验原始配置 dict 并补齐默认值，返回标准 batch dict：

        {
          "template": {...},          # 公共物理参数（默认值 + template 覆盖）
          "temperatures": [int, ...],
          "lammps_mode": "md"|"opt",
          "base_structure_seed": int,
          "base_lammps_seed": int,
          "data_root": Path,          # 已解析为绝对路径
          "configs": [ {...}, ... ],  # 每项已校验并补齐 m/n 等字段
        }

    JSON 与 PY 两种格式共用这一套校验，报错行为完全一致。
    所有格式问题在加载时一次性报出（fail-fast），不等到任务展开中途。
    """
    if not isinstance(raw, dict):
        _fail(path, "顶层必须为对象")

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
    # 相对路径按配置文件所在目录解析，保证从任意 cwd 启动行为一致。
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


def load_batch_config(path):
    """从 batch.json 加载批次配置（格式见模块 docstring）。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"batch 配置文件不存在: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(path, f"JSON 解析失败: {exc}")

    return _validate_batch(raw, path)


def load_batch_py(path):
    """
    从 batch.py 加载批次配置：在只有数据工具的白名单命名空间中
    exec 文件内容，收集同名顶层变量后走与 JSON 完全相同的校验。

    只应写数据生成逻辑（循环 / 变量 / 算术），文件不会获得
    import / open 等能力。
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"batch 配置文件不存在: {path}")

    import builtins
    safe_globals = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in _PY_BUILTINS_ALLOW
        },
    }

    try:
        code = compile(
            path.read_text(encoding="utf-8"),
            str(path),
            "exec",
        )
        exec(code, safe_globals)
    except Exception as exc:
        _fail(path, f"配置脚本执行失败: {type(exc).__name__}: {exc}")

    raw = {
        key: safe_globals[key]
        for key in _BATCH_PY_KEYS
        if key in safe_globals
    }
    return _validate_batch(raw, path)


def load(path):
    """按文件后缀分派加载 batch 配置（.py / 其他一律按 JSON）。"""
    path = Path(path)
    if path.suffix == ".py":
        return load_batch_py(path)
    return load_batch_config(path)
