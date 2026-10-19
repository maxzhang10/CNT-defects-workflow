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
    "defect_count_mode": "same_count",
    "uc_file": null,        # 可选：外部单 UC POSCAR，路径相对批配置
    "save_self_energy": false,
    "self_energy_cache": {"use_saved": true, "save_path": "./self_energy/"},
    "md_sampling": {"n_samples": 1}
  },
  "temperatures": [500],
  "lammps_mode": "opt",          # "md" NVT 退火 / "opt" 几何优化
  "base_structure_seed": 20260705,
  "base_lammps_seed": 23456789,
  "data_root": ".",              # 相对路径按配置文件所在目录解析
  "max_parallel": 16,            # 同时运行的独立 workflow 数上限（可选，
                                # 缺省 16；CLI --max-parallel 可临时覆盖）
  "configs": [
    {
      "chirality": [5, 5],
      "buffer_pl": 2,          # 每侧缓冲层厚度，单位 PL；省略则按手性默认
      "l_def": 8,
      "N_defects": 2,
      "structures": ["5775"],
      "replicas": 20,           # 必填：每个物理配置显式声明自己的
                                # replica 数，不同长度可设不同值
      "self_energy_cache": {    # 可选：覆盖 template 同名配置，
        "use_saved": true,      # 实现不同手性使用不同 self_energy 文件
        "save_path": "{data_root}/self_energy/{chirality}/"
      }
    }
  ]
}

Plan B：长度 + 缺陷密度（替代手写 l_def / N_defects）
------------------------------------------------------
每个 config 的两个维度可以分别二选一（同维度两个都写或都不写即报错）：

    长度维度：l_def（UC 数）        或 length_angstrom（目标散射区长度/Å）
    缺陷维度：N_defects（缺陷数）   或 defect_density（目标缺陷密度/Å⁻¹）

校验时按手性的 UC 长度 T 把目标长度确定性 round 到最近整数：

    l_def     = max(1, round(length_angstrom / T))     实际长度  L = l_def·T

密度到缺陷数支持两种模式：

    defect_count_mode = "same_count"（默认）
        N_defects = max(1, round(defect_density · L))
        每个 replica 使用相同缺陷数，兼容旧行为。

    defect_count_mode = "balanced_count"
        在相邻两个整数缺陷数之间做前缀稳定分配，使任意前 R 个
        replica 的缺陷总数等于 round(R · defect_density · L)。允许
        N_defects=0；增加 replicas 只追加任务，不改变已有 replica。

round 用 int(x + 0.5)（正数四舍五入，无 banker's rounding 歧义）。
两个维度可混搭，例如 "l_def": 12, "defect_density": 0.017 表示定长
12 UC、由密度反推缺陷数。same_count 的解析结果与手写
l_def/N_defects 完全等价；balanced_count 会把目标密度和每个 replica
的实际缺陷数写入 workflow config。

解析表由 format_resolution_table(configs) 生成，batch_generate 在
加载配置后打印到 stdout（随 manage_batch.sh 重定向进
batch_generate.log），并写入 record.log 每个批次启动段。

按手性自能缓存（self_energy_cache）
----------------------------------
同一电极结构在相同电子参数下的自能与散射区缺陷类型 / 长度 /
replica 无关。若使用外部 UC，共享缓存还必须区分 UC 内容和 ele_T。

template.self_energy_cache 是全局默认；任一 config 可用同名键覆盖。
save_path 支持以下占位符，在任务展开时解析：

    {m} {n}       手性指数
    {chirality}   "m_n" 形式（如 13_1）
    {data_root}   批次数据根目录（已解析为绝对路径）
    {uc_sha256}   外部 UC 文件哈希；ASE 模式为 "ase"
    {ele_T}       显式电子温度；未设置时为 "template"

相对路径保持相对语义（DPNEGF 按各 leaf 工作目录解析，即每个 leaf
一份独立缓存）；要跨 leaf / 跨 replica 共享，用 {data_root} 组成
绝对路径，例如 "{data_root}/self_energy/{chirality}/"。

注意：save_self_energy=false 时 run_multi 的自动清理只处理 leaf
内部的相对路径缓存；共享的绝对路径缓存由用户自行管理
（如 rm_self_energy.sh 或手动删除）。

batch.py —— 配置项多、需要循环/变量/条件时使用（同名变量赋值即可）：

    configs = [
        {
            "chirality": [5, 5],
            "l_def": l,
            "N_defects": 2,
            "structures": ["5775"],
            "replicas": 20 if l <= 8 else 40,
        }
        for l in (5, 8, 12)
    ]
    temperatures = [500]
    lammps_mode = "opt"
    base_structure_seed = 20260705
    base_lammps_seed = 23456789
    data_root = "."
    max_parallel = 16       # 可选：并发上限，CLI --max-parallel 可覆盖
    template = {
        "r_max": 6.5,
        # 按手性区分自能缓存（同手性的所有任务共享一份）：
        "self_energy_cache": {
            "use_saved": True,
            "save_path": "{data_root}/self_energy/{chirality}/",
        },
    }

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
import math
from pathlib import Path

from uc_source import read_poscar_metadata
import assemble
from defect_angles import normalize_angle_range


# template 层的公共物理默认值。旧版硬编码在 batch_generate.py 的
# TEMPLATE 全局常量中；JSON 的 template 字段可逐项覆盖。
DEFAULT_TEMPLATE = {
    "defect_angle_range_deg": [0, 360],
    "r_max": 6.5,
    "md_steps": 50000,
    "buffer_pl": None,
    # 固定区（电极+缓冲）最靠中间放开不固定的 uc 数（每侧）。
    "release_uc": 1,
    # defect_density -> N_defects 的 replica 分配方式。same_count 保持
    # 旧行为；balanced_count 在整个 replica ensemble 上逼近目标密度。
    "defect_count_mode": "same_count",
    "uc_file": None,
    "save_self_energy": False,
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
    # 每条独立 LAMMPS 轨迹只取最后一帧做 DPNEGF。
    "md_sampling": {"n_samples": 1},
}

LAMMPS_MODES = ("md", "opt")
DEFECT_COUNT_MODES = ("same_count", "balanced_count")

# l_def / N_defects 不再无条件必填：Plan B 允许分别用
# length_angstrom / defect_density 替代（见 validate_config_entry）。
_REQUIRED_CONFIG_KEYS = (
    "chirality",
    "structures",
    "replicas",
)


def uc_length(m, n, bond_angstrom=1.42):
    """
    最小 UC 轴向长度/Å：3·bond·√(m²+mn+n²) / gcd(2m+n, 2n+m)。

    与 stage/cnt_geometry.calculate_unit_cell_length 同公式
    （根号3 倍键长的石墨烯晶格常数），这里用纯 math 重写，
    batch_config 保持只依赖 json/math/pathlib。
    """
    return (
        3.0
        * float(bond_angstrom)
        * math.sqrt(m * m + m * n + n * n)
        / math.gcd(2 * m + n, 2 * n + m)
    )


def _round_positive(x):
    """正数四舍五入到整数（不用 round()，避免 banker's rounding 歧义）。"""
    return int(x + 0.5)


def balanced_defect_count(expected_per_replica, replica):
    """返回 balanced_count 模式下指定 replica（从 1 开始）的缺陷数。

    累计额外缺陷数采用 floor(R*f + 0.5)，其中 f 是期望缺陷数的小数
    部分。这样任意前缀都执行 half-up 平衡取整，扩大 replicas 时不会
    改变已有 replica 的分配。
    """
    expected = float(expected_per_replica)
    if not math.isfinite(expected) or expected < 0:
        raise ValueError("expected_per_replica 必须为有限非负数")
    if isinstance(replica, bool) or not isinstance(replica, int) or replica < 1:
        raise ValueError("replica 必须为从 1 开始的正整数")

    base = math.floor(expected)
    fraction = expected - base
    extras_before = _round_positive((replica - 1) * fraction)
    extras_after = _round_positive(replica * fraction)
    return base + (extras_after - extras_before)


def balanced_total_defects(expected_per_replica, replicas):
    """返回 balanced_count 模式下前 replicas 个任务的缺陷总数。"""
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("replicas 必须为正整数")
    return _round_positive(float(expected_per_replica) * replicas)


def _require_positive_finite(value, name, path):
    """Plan B 目标值（length_angstrom / defect_density）为有限正数。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        _fail(path, f"{name} 必须为有限正数，当前值: {value!r}")
    return float(value)


class BatchConfigError(ValueError):
    """batch.json 内容不合法。"""


def _fail(path, msg):
    raise BatchConfigError(f"{path}: {msg}")


def _require_int(value, name, path, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"{name} 必须为不小于 {minimum} 的整数，当前值: {value!r}")
    return value


def _require_ele_T(value, name, path):
    """ele_T 为可选的有限非负温度（K），允许 int/float。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        _fail(path, f"{name} 必须为有限非负数（K），当前值: {value!r}")
    return float(value)


def _require_bool(value, name, path):
    if not isinstance(value, bool):
        _fail(path, f"{name} 必须为 true/false，当前值: {value!r}")
    return value


def _require_defect_count_mode(value, name, path):
    if value not in DEFECT_COUNT_MODES:
        _fail(
            path,
            f"{name} 必须为 {' / '.join(DEFECT_COUNT_MODES)}，当前值: {value!r}",
        )
    return value


def conductance_mode_for_chirality(m, n):
    """Infer transport mode from CNT chirality."""
    return "fermi" if (int(m) - int(n)) % 3 == 0 else "band_edge_bias"


def _validate_self_energy_cache(cache, name, path):
    """校验 self_energy_cache 配置，返回标准 dict。"""
    if not isinstance(cache, dict):
        _fail(path, f"{name} 必须为对象，当前值: {cache!r}")
    use_saved = cache.get("use_saved")
    save_path = cache.get("save_path")
    _require_bool(use_saved, f"{name}.use_saved", path)
    if not isinstance(save_path, str) or not save_path:
        _fail(path, f"{name}.save_path 必须为非空字符串，当前值: {save_path!r}")
    return {"use_saved": use_saved, "save_path": save_path}


def resolve_self_energy_cache(cache, *, m, n, data_root,
                              uc_sha256=None, ele_T=None):
    """
    解析 self_energy_cache.save_path 中的按手性占位符。

    支持的占位符：{m} {n} {chirality}（"m_n" 形式）{data_root}
    {uc_sha256} {ele_T}。
    其余字段（use_saved）原样保留。相对路径不改动，保持 DPNEGF
    按 leaf 工作目录解析的语义。
    """
    if cache is None:
        return None
    resolved = dict(cache)
    save_path = resolved.get("save_path")
    if isinstance(save_path, str):
        try:
            resolved["save_path"] = save_path.format(
                m=int(m),
                n=int(n),
                chirality=f"{int(m)}_{int(n)}",
                data_root=str(data_root),
                uc_sha256=uc_sha256 or "ase",
                ele_T=f"{float(ele_T):g}" if ele_T is not None else "template",
            )
        except KeyError as exc:
            raise BatchConfigError(
                f"self_energy_cache.save_path 含未知占位符 {exc}；"
                "仅支持 {{m}} {{n}} {{chirality}} {{data_root}} "
                "{{uc_sha256}} {{ele_T}}，"
                f"当前值: {save_path!r}"
            )
    return resolved


def validate_config_entry(
    entry,
    index,
    path,
    default_defect_count_mode="same_count",
    default_uc_file=None,
):
    """校验单个物理配置条目，返回补齐 chirality 后的新 dict。"""
    if not isinstance(entry, dict):
        _fail(path, f"configs[{index}] 必须为对象，当前值: {entry!r}")

    # 手性接受两种写法：chirality: [m, n]，或分开的 m: .., n: ..（多手性
    # 批次用循环生成时常用后者，见仓库 batch.py 模板）。
    if "chirality" not in entry and "m" in entry and "n" in entry:
        entry = {**entry, "chirality": [entry["m"], entry["n"]]}

    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in entry]
    if missing:
        _fail(path, f"configs[{index}] 缺少必需字段: {', '.join(missing)}")

    chirality = entry["chirality"]
    # 手性约定 m>=n>=0（锯齿管 n=0 合法），只有 m 必须为正整数。
    if (
        not isinstance(chirality, list)
        or len(chirality) != 2
        or not all(isinstance(x, int) and not isinstance(x, bool)
                   for x in chirality)
        or chirality[0] < 1
        or chirality[1] < 0
    ):
        _fail(
            path,
            f"configs[{index}].chirality 必须为 [正整数, 非负整数]，"
            f"如 [5, 5] 或 [9, 0]，当前值: {chirality!r}",
        )
    m, n = chirality

    # Plan B：长度维度 l_def / length_angstrom 二选一，
    # 缺陷维度 N_defects / defect_density 二选一，可跨维度混搭。
    has_l_def = "l_def" in entry
    has_length = "length_angstrom" in entry
    if has_l_def == has_length:
        _fail(
            path,
            f"configs[{index}] 长度维度必须且只能提供 "
            f"l_def / length_angstrom 之一",
        )
    has_n_defects = "N_defects" in entry
    has_density = "defect_density" in entry
    if has_n_defects == has_density:
        _fail(
            path,
            f"configs[{index}] 缺陷维度必须且只能提供 "
            f"N_defects / defect_density 之一",
        )

    # 显式 N_defects 已经给出了每个 replica 的缺陷数，因此固定使用
    # same_count。defect_density 才允许选择 ensemble 平衡分配。
    entry_defect_count_mode = None
    if "defect_count_mode" in entry:
        entry_defect_count_mode = _require_defect_count_mode(
            entry["defect_count_mode"],
            f"configs[{index}].defect_count_mode",
            path,
        )

    if has_n_defects:
        if entry_defect_count_mode == "balanced_count":
            _fail(
                path,
                f"configs[{index}].defect_count_mode=balanced_count 只适用于 "
                "defect_density；显式 N_defects 请使用 same_count 或省略该字段",
            )
        defect_count_mode = "same_count"
    else:
        defect_count_mode = (
            entry_defect_count_mode
            if entry_defect_count_mode is not None
            else default_defect_count_mode
        )

    resolution = {}  # 记录目标值，供 format_resolution_table 打印
    uc_file = entry.get("uc_file", default_uc_file)
    uc_sha256 = None
    if uc_file is not None:
        if not isinstance(uc_file, str) or not uc_file:
            _fail(path, f"configs[{index}].uc_file 必须为非空 POSCAR 路径")
        try:
            uc_file = uc_file.format(m=m, n=n, chirality=f"{m}_{n}")
        except (KeyError, ValueError, AttributeError) as exc:
            _fail(path, f"configs[{index}].uc_file 占位符无效: {exc}")
        uc_path = Path(uc_file).expanduser()
        if not uc_path.is_absolute():
            uc_path = path.parent / uc_path
        try:
            uc_meta = read_poscar_metadata(uc_path)
        except (OSError, UnicodeError, ValueError) as exc:
            _fail(path, f"configs[{index}].uc_file 无效: {exc}")
        expected_atoms = 4 * (m * m + m * n + n * n) // math.gcd(2*m+n, 2*n+m)
        if uc_meta["atom_count"] != expected_atoms:
            _fail(path, f"configs[{index}].uc_file 应包含一个 ({m},{n}) UC 的 "
                 f"{expected_atoms} 个 C 原子，实际 {uc_meta['atom_count']}")
        uc_file = uc_meta["path"]
        uc_sha256 = uc_meta["sha256"]
        uc = uc_meta["length_angstrom"]
    else:
        uc = uc_length(m, n)

    if has_length:
        target_length = _require_positive_finite(
            entry["length_angstrom"], f"configs[{index}].length_angstrom", path
        )
        l_def = max(1, _round_positive(target_length / uc))
        resolution["length_angstrom"] = target_length
    else:
        l_def = _require_int(entry["l_def"], f"configs[{index}].l_def", path)

    actual_length = l_def * uc
    if has_density:
        target_density = _require_positive_finite(
            entry["defect_density"], f"configs[{index}].defect_density", path
        )
        expected_defects = target_density * actual_length
        if defect_count_mode == "balanced_count":
            # balanced_count 的 N_defects 是 replica 级属性，在 build_tasks
            # 展开时按 replica 序号计算；这里不伪造单一整数值。
            n_defects = None
        else:
            n_defects = max(1, _round_positive(expected_defects))
        resolution["defect_density"] = target_density
    else:
        n_defects = _require_int(
            entry["N_defects"], f"configs[{index}].N_defects", path
        )
    buffer_pl = entry.get("buffer_pl")
    if buffer_pl is not None:
        buffer_pl = _require_int(
            buffer_pl, f"configs[{index}].buffer_pl", path, minimum=0
        )
    release_uc = entry.get("release_uc")
    if release_uc is not None:
        release_uc = _require_int(
            release_uc, f"configs[{index}].release_uc", path, minimum=0
        )
    # ele_T 可选：per-config 覆盖 template；None 表示沿用 input.json 模板值。
    ele_T = entry.get("ele_T")
    if ele_T is not None:
        ele_T = _require_ele_T(ele_T, f"configs[{index}].ele_T", path)

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

    if "conductance_mode" in entry:
        _fail(path, f"configs[{index}] 不再接受 conductance_mode，请由 chirality 自动判断")
    conductance_mode = conductance_mode_for_chirality(m, n)

    if "self_energy_cache" in entry:
        # per-config 覆盖：不同手性 / 配置可用不同的自能缓存路径。
        entry_cache = _validate_self_energy_cache(
            entry["self_energy_cache"],
            f"configs[{index}].self_energy_cache",
            path,
        )
    else:
        entry_cache = None

    out = dict(entry)
    out["m"] = m
    out["n"] = n
    # 目标值保留在内部 _resolution 供任务展开和解析表使用；
    # make_task_config 只在 balanced_count 时把目标密度写入下游配置。
    out.pop("length_angstrom", None)
    out.pop("defect_density", None)
    out["_resolution"] = resolution
    out["l_def"] = l_def
    out["buffer_pl"] = buffer_pl
    if "defect_angle_range_deg" in entry:
        out["defect_angle_range_deg"] = normalize_angle_range(entry["defect_angle_range_deg"])
    out["release_uc"] = release_uc
    out["ele_T"] = ele_T
    out["N_defects"] = n_defects
    out["defect_count_mode"] = defect_count_mode
    out["uc_file"] = uc_file
    out["uc_sha256"] = uc_sha256
    out["uc_length_angstrom"] = uc
    out["structures"] = list(structures)
    out["conductance_mode"] = conductance_mode
    # replicas 是每个物理配置的必填显式字段（不接受全局默认），
    # 杜绝"忘了写就静默用全局值"的歧义。
    out["replicas"] = _require_int(
        entry["replicas"], f"configs[{index}].replicas", path
    )
    # per-config 自能缓存覆盖；None 表示沿用 template。
    out["self_energy_cache"] = entry_cache
    return out


# batch.py 加载时注入的白名单命名空间：只放数据生成用到的工具，
# 不给 import / open / exec 等能力。配置文件是数据不是脚本。
_PY_BUILTINS_ALLOW = (
    "dict", "list", "tuple", "set", "int", "float", "str", "bool",
    "len", "range", "enumerate", "zip", "sorted", "min", "max",
    "sum", "abs", "round",
)

# batch.py 中会被收集的顶层变量名（其余变量如循环临时量自动忽略）。
# 配置列表接受 configs 或 CHIRAL_CONFIGS（后者为多手性批次的惯用名，
# 见仓库 batch.py 模板），二选一。
_BATCH_PY_KEYS = (
    "structure_mode",
    "template",
    "temperatures",
    "lammps_mode",
    "base_structure_seed",
    "base_lammps_seed",
    "data_root",
    "max_parallel",
    "CHIRAL_CONFIGS",
    "configs",
)


def validate_assemble_entry(entry, index, path, template):
    """Validate a separate module-assembly configuration without UC rounding."""
    label = f"configs[{index}]"
    if not isinstance(entry, dict):
        _fail(path, f"{label} must be an object")
    if any(key in entry for key in ("uc_file", "l_def", "N_defects", "structures", "defect_density", "electrode_length_angstrom", "scatter_length_angstrom", "perfect_uc_count")):
        _fail(path, f"{label}: UC/ASE defect fields cannot be mixed with assembly mode")
    chirality = entry.get("chirality", [entry.get("m"), entry.get("n")])
    if (not isinstance(chirality, (list, tuple)) or len(chirality) != 2
            or any(type(x) is not int for x in chirality)
            or chirality[0] < 1 or chirality[1] < 0):
        _fail(path, f"{label}.chirality must be [m, n]")
    try:
        perfect_path = Path(entry["perfect_module"])
        defect_paths = entry["defect_modules"]
        mix_mode = entry.get("defect_mix_mode", "fixed_count")
        if mix_mode not in ("fixed_count", "balanced_count"):
            _fail(path, f"{label}.defect_mix_mode must be fixed_count or balanced_count")
        if mix_mode == "balanced_count":
            if "defect_counts" in entry:
                _fail(path, f"{label}: balanced mixture cannot specify defect_counts")
            total = _require_int(entry["total_defects"], f"{label}.total_defects", path, minimum=0)
            if not isinstance(defect_paths, dict) or len(defect_paths) != 2:
                _fail(path, f"{label}: balanced mixture requires exactly two defect modules")
            if any(not isinstance(k, str) or not k for k in defect_paths):
                _fail(path, f"{label}: invalid defect names")
            names = sorted(defect_paths)
            first = balanced_defect_count(total / 2, 1)
            counts = {names[0]: first, names[1]: total - first}
            if entry.get("module_sequence") is not None:
                _fail(path, f"{label}: balanced mixture cannot specify module_sequence")
        else:
            if "total_defects" in entry:
                _fail(path, f"{label}: total_defects requires defect_mix_mode=balanced_count")
            counts = entry["defect_counts"]
        scatter_uc = _require_int(entry["scatter_uc_count"], f"{label}.scatter_uc_count", path)
        angles = entry["rotation_angles_deg"]
    except (KeyError, TypeError, ValueError) as exc:
        _fail(path, f"{label}: missing or invalid assembly field: {exc}")
    if not isinstance(defect_paths, dict) or not isinstance(counts, dict) or set(defect_paths) != set(counts):
        _fail(path, f"{label}: defect_modules and defect_counts must have identical keys")
    if ("perfect" in defect_paths or any(not isinstance(k, str) or not k for k in defect_paths)
            or any(type(v) is not int or v < 0 for v in counts.values())):
        _fail(path, f"{label}: invalid defect names or counts")
    if (not isinstance(angles, list) or not angles
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in angles)):
        _fail(path, f"{label}.rotation_angles_deg must be a finite angle list")
    if 0 not in angles or len({x % 360 for x in angles}) != len(angles):
        _fail(path, f"{label}.rotation_angles_deg must include 0 once and have no duplicates modulo 360")
    def metadata(raw_path):
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = path.parent / p
        return assemble.module_metadata(p)
    try:
        perfect = metadata(perfect_path)
        if perfect["symbols"] != ["C"]:
            raise ValueError("perfect module must contain only C")
        defects = {}
        for name, value in defect_paths.items():
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or not isinstance(value[0], str) or not value[0]
                    or type(value[1]) is not int or value[1] < 1):
                raise ValueError(f"defect_modules.{name} must be (POSCAR path, positive UC count)")
            defects[name] = metadata(value[0])
            defects[name]["uc_count"] = value[1]
            expected_length = value[1] * perfect["length_angstrom"]
            if not math.isclose(defects[name]["length_angstrom"], expected_length,
                                rel_tol=0, abs_tol=1e-5):
                raise ValueError(
                    f"defect module {name} cell length {defects[name]['length_angstrom']} Å "
                    f"does not equal {value[1]} × perfect UC length ({expected_length} Å)"
                )
        if mix_mode == "balanced_count" and len({m["uc_count"] for m in defects.values()}) != 1:
            raise ValueError("balanced mixture requires equal module UC counts")
        for name, module in defects.items():
            if any(abs(a - b) > 1e-5 for va, vb in zip(module["cell"][:2], perfect["cell"][:2])
                   for a, b in zip(va, vb)):
                raise ValueError(f"defect module {name} transverse cell differs from perfect module")
        buffer_pl = entry.get("buffer_pl", template["buffer_pl"])
        if buffer_pl is None:
            buffer_pl = 2 if (chirality[0] - chirality[1]) % 3 == 0 else 4
        release_uc = template["release_uc"]
        if release_uc is None:
            release_uc = 1
        layout = assemble.lead_layout(
            perfect, chirality[0], chirality[1], template["r_max"],
            buffer_pl, release_uc,
        )
        normalized = {
            "m": chirality[0], "n": chirality[1],
            "modules": {"perfect": perfect, "defects": defects},
            "defect_counts": dict(counts),
            "scatter_uc_count": scatter_uc,
            "layout": layout,
            "buffer_pl": buffer_pl,
            "rotation_angles_deg": list(angles),
            "module_sequence": entry.get("module_sequence"),
            "replicas": _require_int(entry["replicas"], f"{label}.replicas", path),
            "ele_T": entry.get("ele_T"),
            "self_energy_cache": entry.get("self_energy_cache"),
        }
        if normalized["module_sequence"] is not None and not isinstance(normalized["module_sequence"], list):
            raise ValueError("module_sequence must be a list")
        if mix_mode == "balanced_count":
            normalized["defect_mix_mode"] = mix_mode
            normalized["total_defects"] = total
        assemble.plan(normalized, 0)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _fail(path, f"{label}: {exc}")
    if normalized["ele_T"] is not None:
        normalized["ele_T"] = _require_ele_T(normalized["ele_T"], f"{label}.ele_T", path)
    if normalized["self_energy_cache"] is not None:
        normalized["self_energy_cache"] = _validate_self_energy_cache(normalized["self_energy_cache"], f"{label}.self_energy_cache", path)
    return normalized


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
          "max_parallel": int | None, # 并发上限；None = 未配置，用 CLI/默认值
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
    template["defect_angle_range_deg"] = normalize_angle_range(template["defect_angle_range_deg"])
    _require_bool(template["save_self_energy"], "template.save_self_energy", path)
    template["defect_count_mode"] = _require_defect_count_mode(
        template["defect_count_mode"], "template.defect_count_mode", path
    )
    if template["buffer_pl"] is not None:
        template["buffer_pl"] = _require_int(
            template["buffer_pl"], "template.buffer_pl", path, minimum=0
        )
    if template["release_uc"] is not None:
        template["release_uc"] = _require_int(
            template["release_uc"], "template.release_uc", path, minimum=0
        )
    # ele_T 可选：缺省/None 表示沿用 input.json 模板中的 ele_T，
    # 不由工作流覆写 DPNEGF 电极温度。
    ele_T = template.get("ele_T")
    if ele_T is not None:
        template["ele_T"] = _require_ele_T(ele_T, "template.ele_T", path)
    template["self_energy_cache"] = _validate_self_energy_cache(
        template["self_energy_cache"], "template.self_energy_cache", path
    )

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

    # 并发上限是运行时参数，不进入 template / configs，不参与
    # provenance hash：改并行度不影响已有结果的确定性约定。
    max_parallel = raw.get("max_parallel")
    if max_parallel is not None:
        max_parallel = _require_int(max_parallel, "max_parallel", path)

    configs_raw = raw.get("configs")
    if not isinstance(configs_raw, list) or not configs_raw:
        _fail(path, "configs 必须为非空列表")

    structure_mode = raw.get("structure_mode", "defects")
    if structure_mode not in ("defects", "assemble"):
        _fail(path, "structure_mode must be 'defects' or 'assemble'")
    if structure_mode == "assemble":
        if template.get("uc_file") is not None:
            _fail(path, "assembly mode cannot use template.uc_file")
        configs = [validate_assemble_entry(entry, i, path, template) for i, entry in enumerate(configs_raw)]
    else:
        configs = [
            validate_config_entry(
                entry, i, path,
                default_defect_count_mode=template["defect_count_mode"],
                default_uc_file=template["uc_file"],
            ) for i, entry in enumerate(configs_raw)
        ]

    return {
        "structure_mode": structure_mode,
        "template": template,
        "temperatures": list(temperatures),
        "lammps_mode": lammps_mode,
        "base_structure_seed": base_structure_seed,
        "base_lammps_seed": base_lammps_seed,
        "data_root": data_root,
        "max_parallel": max_parallel,
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
        "__name__": "_batch_config_",
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
    if "configs" not in raw and "CHIRAL_CONFIGS" in raw:
        raw["configs"] = raw.pop("CHIRAL_CONFIGS")
    return _validate_batch(raw, path)


def format_resolution_table(configs):
    """
    把 Plan B 解析结果格式化成对照表（多行字符串）。

    只列出用了 length_angstrom / defect_density 的 config；纯手写
    l_def/N_defects 的批次返回 None（调用方跳过打印）。
    """
    rows = []
    for cfg in configs:
        resolution = cfg.get("_resolution") or {}
        if not resolution:
            continue
        uc = cfg.get("uc_length_angstrom") or uc_length(cfg["m"], cfg["n"])
        actual_length = cfg["l_def"] * uc
        target_length = resolution.get("length_angstrom")
        target_density = resolution.get("defect_density")
        defect_count_mode = cfg.get("defect_count_mode", "same_count")

        if target_density is not None and defect_count_mode == "balanced_count":
            expected = target_density * actual_length
            total_defects = balanced_total_defects(expected, cfg["replicas"])
            actual_density = total_defects / (cfg["replicas"] * actual_length)
            counts = sorted(
                {
                    balanced_defect_count(expected, replica)
                    for replica in range(1, cfg["replicas"] + 1)
                }
            )
            count_text = "/".join(str(value) for value in counts)
            density_text = (
                f"密度 {target_density:.6f} Å^-1 -> balanced_count "
                f"N={count_text}, ensemble 实际 {actual_density:.6f} Å^-1 "
                f"({(actual_density / target_density - 1) * 100:+.1f}%)"
            )
        else:
            actual_density = cfg["N_defects"] / actual_length
            density_text = (
                f"密度 {target_density:.6f} Å^-1 -> "
                f"same_count N_defects={cfg['N_defects']}, "
                f"实际 {actual_density:.6f} Å^-1 "
                f"({(actual_density / target_density - 1) * 100:+.1f}%)"
                if target_density is not None
                else f"N_defects={cfg['N_defects']}（手写，{actual_density:.6f} Å^-1）"
            )
        rows.append(
            f"  ({cfg['m']},{cfg['n']}) UC={uc:.4f} Å: "
            + (
                f"长度 {target_length:.1f} Å -> l_def={cfg['l_def']}, "
                f"实际 {actual_length:.1f} Å "
                f"({(actual_length / target_length - 1) * 100:+.1f}%); "
                if target_length is not None
                else f"l_def={cfg['l_def']}（手写，{actual_length:.1f} Å）; "
            )
            + density_text
        )
    if not rows:
        return None
    header = "[INFO] Plan B 解析表（长度/密度 -> l_def/N_defects）:"
    return "\n".join([header, *rows])


def load(path):
    """按文件后缀分派加载 batch 配置（.py / 其他一律按 JSON）。"""
    path = Path(path)
    if path.suffix == ".py":
        return load_batch_py(path)
    return load_batch_config(path)
