#!/usr/bin/env python3
"""CNT transport summary: a full-width violin panel above three mean curves.

Usage:
    python transport_four_panels_v4.py /path/to/5_5 --linear --length-unit nm
    python transport_four_panels_v4.py /path/to/5_5 --band-edge -o transport_summary.pdf

The original log-first / PTH-fallback collection workflow is preserved.
Each collected record carries equal weight, as in the original script.
Top: original violin, transformed by --linear and --R.
Bottom: mean(g), mean(ln(g)) on a linear axis, and mean(1/g), g=G/G0.
Each replica represents one structural frame and one NEGF conductance.
The last panel uses R0=1/G0; it is NOT 1/mean(g).
v3: PTH input is the conductance G/G0 precomputed by the NEGF workflow
(finite-temperature thermal integration already applied), read directly
from the "conductance" list in negf.out.pth; T_avg is no longer used.
With --exclude-outliers, apply a single 1.5*IQR filter to raw g in each
length group. All distributions and means use the same retained samples.
This is a trimmed ensemble; statistically extreme values may be physical.
v4: with --band-edge, no longer read the conductance at the Fermi level.
Instead read the two entries labeled "Ec" and "Ev" in the conductance
list of negf.out.pth (the band-edge chemical potentials computed by the
NEGF workflow). Every negf.out.pth therefore yields twice as many
records, one per band edge. Each (length, band-edge) pair is drawn as
its own violin with a small horizontal offset and its own color, and the
three mean panels show one curve per band edge. replica_conductance.log
files have no band-edge information and are ignored in --band-edge mode.
Only load trusted PTH files (torch.load uses weights_only=False).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BAND_ORDER = {"Ec": 0, "Ev": 1}
BAND_STYLES = {
    "Ec": {"violin": "#9ACDCD", "accent": "#008080"},
    "Ev": {"violin": "#F2C1A0", "accent": "#D35400"},
}
DEFAULT_STYLE = {"violin": "#9ACDCD", "accent": "#008080"}


def style_for_band(band):
    return BAND_STYLES.get(band, DEFAULT_STYLE)


def load_conductance(pth_path, fermi_energy=0.0):
    """
    直接读取 PTH 文件中 NEGF 流程预先算好的电导。

    negf.out.pth 的 conductance 列表中每个条目对应一个化学势
    (mu_eV, 相对 E_ref),G_over_G0 是对 T(E) 做有限温度热积分
    后的电导。这里取 mu_eV 与 fermi_energy 最接近的条目,
    不再对 T_avg 做插值或积分。
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("读取 PTH 文件需要安装 torch；只读取日志时不需要") from exc

    data = torch.load(
        pth_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, dict):
        raise TypeError(f"文件内容不是字典: {pth_path}")

    if "conductance" not in data:
        raise KeyError(f"缺少 conductance: {pth_path}")

    entries = data["conductance"]
    if not entries:
        raise ValueError(f"conductance 列表为空: {pth_path}")

    entry = min(
        entries,
        key=lambda e: abs(float(e.get("mu_eV", 0.0))) - fermi_energy
    )

    if "G_over_G0" not in entry:
        raise KeyError(f"conductance 条目缺少 G_over_G0: {pth_path}")

    conductance = float(entry["G_over_G0"])
    if not np.isfinite(conductance):
        raise ValueError(
            f"G_over_G0 非有限值: {conductance}, file={pth_path}"
        )

    mu = float(entry.get("mu_eV", 0.0))
    temperature = float(entry.get("temperature_K", float("nan")))

    return conductance, mu, temperature


def _absolute_mu(entry, e_ref):
    """conductance 条目的绝对化学势;优先 mu_absolute_eV。"""
    if entry.get("mu_absolute_eV") is not None:
        return float(entry["mu_absolute_eV"])
    return float(entry.get("mu_eV", 0.0)) + e_ref


def load_conductance_band_edges(pth_path):
    """
    读取 PTH 文件中带边(Ec / Ev)处的电导。

    NEGF 流程在 conductance 列表中为每个带边写入一个条目,
    条目的 label 字段标记 "Ec" 或 "Ev"。对没有 label 的旧文件,
    退回到用顶层 E_c / E_v(lead_L/lead_R 的平均)匹配绝对化学势
    最接近的条目。

    返回 {"Ec": (G/G0, mu_eV, T), "Ev": (G/G0, mu_eV, T)}。
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("读取 PTH 文件需要安装 torch") from exc

    data = torch.load(
        pth_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, dict):
        raise TypeError(f"文件内容不是字典: {pth_path}")

    entries = data.get("conductance")
    if not entries:
        raise ValueError(f"conductance 列表为空: {pth_path}")

    e_ref = float(data.get("E_ref", 0.0))
    results = {}
    for band in ("Ec", "Ev"):
        labeled = [
            e for e in entries
            if str(e.get("label", "")).strip().lower() == band.lower()
        ]
        if labeled:
            entry = labeled[0]
        else:
            edge = data.get("E_c" if band == "Ec" else "E_v")
            if edge is None:
                raise KeyError(
                    f"conductance 条目缺少 label={band},且顶层没有 "
                    f"{'E_c' if band == 'Ec' else 'E_v'} 可回退: {pth_path}"
                )
            if isinstance(edge, dict):
                target = float(np.mean([float(v) for v in edge.values()]))
            else:
                target = float(edge)
            entry = min(
                entries,
                key=lambda e: abs(_absolute_mu(e, e_ref) - target)
            )

        if "G_over_G0" not in entry:
            raise KeyError(f"conductance 条目缺少 G_over_G0: {pth_path}")

        conductance = float(entry["G_over_G0"])
        if not np.isfinite(conductance):
            raise ValueError(
                f"{band} 处 G_over_G0 非有限值: {conductance}, file={pth_path}"
            )

        results[band] = (
            conductance,
            float(entry.get("mu_eV", 0.0)),
            float(entry.get("temperature_K", float("nan"))),
        )

    return results

def get_group_name(pth_path):
    """
    同时兼容新旧目录结构。

    新目录：
        .../配置名/replica_001/dpnegf/200000/output/negf.out.pth

    旧目录：
        .../配置名/dpnegf/40000/output/negf.out.pth

    返回配置名，例如：
        5775_L008_0.1016A-1
    """
    parts = pth_path.parts

    dpnegf_indices = [
        i for i, part in enumerate(parts)
        if part == "dpnegf"
    ]

    if not dpnegf_indices:
        return "unknown"

    dpnegf_index = dpnegf_indices[-1]

    if dpnegf_index == 0:
        return "unknown"

    parent_name = parts[dpnegf_index - 1]

    if parent_name.startswith("replica_"):
        if dpnegf_index < 2:
            return "unknown"
        return parts[dpnegf_index - 2]

    return parent_name


def get_sample_name(pth_path):
    """
    返回样本名。

    新目录返回：
        replica_001/200000

    旧目录返回：
        40000
    """
    step = pth_path.parent.parent.name
    dpnegf_dir = pth_path.parent.parent.parent
    replica_dir = dpnegf_dir.parent

    if replica_dir.name.startswith("replica_"):
        return f"{replica_dir.name}/{step}"

    return step


REPLICA_CONDUCTANCE_PATTERN = re.compile(
    r"^(?P<sample>replica_\d+)\s+.*?\bG/G0=(?P<value>[-+0-9.eE]+)\b",
    re.MULTILINE,
)
L_DEF_PATTERN = re.compile(r"(?:^|_)L(?P<l_def>\d+(?:\.\d+)?)(?:_|$)", re.I)
CHIRALITY_PATTERN = re.compile(r"^(?P<n>\d+)_(?P<m>\d+)$")


def find_chirality(path):
    """Find the nearest CNT chirality directory such as ``5_5``."""
    for parent in (path, *path.parents):
        match = CHIRALITY_PATTERN.fullmatch(parent.name)
        if match:
            return int(match.group("n")), int(match.group("m"))
    raise ValueError(f"路径中找不到手性目录 n_m: {path}")


def cnt_unit_cell_length_angstrom(n, m, graphene_lattice_constant=2.46):
    """Return the axial CNT translational unit-cell length in Å.

    T = a * sqrt(3 (n² + nm + m²)) / gcd(2n+m, 2m+n), where a=2.46 Å.
    For an armchair (5,5) tube, this gives 2.46 Å.
    """
    if n <= 0 or m < 0 or m > n:
        raise ValueError(f"无效手性: ({n}, {m})")
    divisor = np.gcd(2 * n + m, 2 * m + n)
    return float(
        graphene_lattice_constant
        * np.sqrt(3.0 * (n * n + n * m + m * m))
        / divisor
    )


def length_from_config_dir(config_dir):
    """Calculate the defect-region length L = l_def * T from its directory."""
    match = L_DEF_PATTERN.search(config_dir.name)
    if match is None:
        raise ValueError(f"配置名中找不到 L### 字段: {config_dir.name}")
    l_def = float(match.group("l_def"))
    n, m = find_chirality(config_dir)
    unit_cell_length = cnt_unit_cell_length_angstrom(n, m)
    return l_def * unit_cell_length, l_def, n, m, unit_cell_length


def parse_replica_conductance_log(log_path):
    """Read replica-level G/G0 values written by collect_replica_conductance.

    A valid log contains one or more lines such as:
        replica_001  seed=...  G/G0=1.9278  ln=...  file=...
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    config_dir = log_path.parent.resolve()
    length, l_def, n, m, unit_cell_length = length_from_config_dir(config_dir)
    records = []
    for match in REPLICA_CONDUCTANCE_PATTERN.finditer(text):
        value = float(match.group("value"))
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"日志中的 G/G0 必须为有限非负数: {match.group(0)}"
            )
        records.append({
            "value": value,
            "sample": match.group("sample"),
            "length_A": length,
            "l_def": l_def,
            "chirality": (n, m),
            "unit_cell_length_A": unit_cell_length,
            "band": None,
        })

    if not records:
        raise ValueError("未找到 replica 级 G/G0 行")
    return records


def config_dir_from_pth(pth_path):
    """Return the configuration directory for an accepted negf.out.pth path."""
    dpnegf_index = max(
        i for i, part in enumerate(pth_path.parts) if part == "dpnegf"
    )
    parent = pth_path.parts[dpnegf_index - 1]
    if parent.startswith("replica_"):
        return Path(*pth_path.parts[:dpnegf_index - 1])
    return Path(*pth_path.parts[:dpnegf_index])


def collect_results(root, fermi_energy=0.0, band_edge=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root 目录不存在: {root}")

    grouped_data = {}
    logged_config_dirs = set()
    print(f"[INFO] root: {root}")

    if band_edge:
        # replica_conductance.log 只有费米能级处的 G/G0,没有带边信息。
        print("[INFO] --band-edge 模式: 忽略 replica_conductance.log,"
              "只读取 negf.out.pth 中 label 为 Ec/Ev 的条目")
    else:
        logs = sorted(root.rglob("replica_conductance.log"))
        print(f"[INFO] 找到 {len(logs)} 个 replica_conductance.log")

        # Priority 1: use the collector's replica-level results wherever available.
        for log_path in logs:
            try:
                records = parse_replica_conductance_log(log_path)
                config_dir = log_path.parent.resolve()
                group = config_dir.name
                grouped_data.setdefault(group, []).extend(records)
                logged_config_dirs.add(config_dir)
                print(
                    f"[LOG] {group:<30} n={len(records):3d} "
                    f"L={records[0]['length_A']:.6g} Å "
                    f"source={log_path}"
                )
            except Exception as exc:
                print(f"[LOG-SKIP] {log_path}")
                print(f"           {type(exc).__name__}: {exc}")

    # Priority 2: only configurations without a valid log fall back to negf.out.pth.
    result_files = sorted(
        path for path in root.rglob("negf.out.pth") if is_dpnegf_result(path)
    )
    print(f"[INFO] 找到 {len(result_files)} 个严格匹配的 dpnegf 结果")
    for pth_path in result_files:
        try:
            config_dir = config_dir_from_pth(pth_path).resolve()
            if config_dir in logged_config_dirs:
                continue

            base_group = get_group_name(pth_path)
            sample = get_sample_name(pth_path)
            length, l_def, n, m, unit_cell_length = length_from_config_dir(config_dir)

            if band_edge:
                band_results = load_conductance_band_edges(pth_path)
                for band in ("Ec", "Ev"):
                    conductance, mu, temperature = band_results[band]
                    if conductance < 0:
                        raise ValueError(
                            f"{band} 处读取到的电导为负: {conductance}"
                        )
                    group = f"{base_group} {band}"
                    grouped_data.setdefault(group, []).append(
                        {
                            "value": conductance,
                            "sample": sample,
                            "length_A": length,
                            "l_def": l_def,
                            "chirality": (n, m),
                            "unit_cell_length_A": unit_cell_length,
                            "band": band,
                        }
                    )
                    print(
                        f"[PTH] {group:<36} sample={sample:<10} "
                        f"G/G0({band}, mu={mu:g} eV)={conductance:.12g}  "
                        f"T={temperature:g} K"
                    )
            else:
                conductance, mu, temperature = load_conductance(
                    pth_path, fermi_energy=fermi_energy
                )
                if conductance < 0:
                    raise ValueError(f"读取到的电导为负: {conductance}")

                grouped_data.setdefault(base_group, []).append(
                    {
                        "value": conductance,
                        "sample": sample,
                        "length_A": length,
                        "l_def": l_def,
                        "chirality": (n, m),
                        "unit_cell_length_A": unit_cell_length,
                        "band": None,
                    }
                )
                print(
                    f"[PTH] {base_group:<30} sample={sample:<10} "
                    f"G/G0(mu={mu:g} eV)={conductance:.12g}  "
                    f"T={temperature:g} K"
                )
        except Exception as exc:
            print(f"[SKIP] {pth_path}")
            print(f"       {type(exc).__name__}: {exc}")

    return grouped_data

def is_dpnegf_result(path):
    """
    只接受：
    .../结构名/dpnegf/采样步/output/negf.out.pth
    """
    return (
        path.name == "negf.out.pth"
        and path.parent.name == "output"
        and path.parent.parent.parent.name == "dpnegf"
    )

def plot_transport_summary(
    grouped_data,
    output_path,
    log_scale=True,
    resistance=False,
    random_seed=20260731,
    length_unit="A",
    exclude_outliers=False,
):
    if not grouped_data:
        raise RuntimeError("没有读取到有效的 dpnegf 数据")

    def group_length(group):
        records = grouped_data[group]
        if not records or not isinstance(records[0], dict):
            raise ValueError(
                "旧格式数据不含手性和 l_def，无法计算物理横轴长度"
            )
        lengths = {float(record["length_A"]) for record in records}
        if len(lengths) != 1:
            raise ValueError(f"{group} 内部的物理长度不一致: {sorted(lengths)}")
        return lengths.pop()

    def group_band(group):
        records = grouped_data[group]
        bands = {record.get("band") for record in records}
        if len(bands) != 1:
            raise ValueError(f"{group} 内部的带边标记不一致: {sorted(bands)}")
        return bands.pop()

    if length_unit not in {"A", "nm"}:
        raise ValueError("length_unit 必须为 A 或 nm")
    length_factor = 0.1 if length_unit == "nm" else 1.0
    unit_label = "nm" if length_unit == "nm" else "Å"

    # 按 (物理长度, 带边) 排序;同一长度的 Ec/Ev 组相邻放置。
    groups = sorted(
        grouped_data,
        key=lambda g: (group_length(g), BAND_ORDER.get(group_band(g), 0)),
    )
    base_positions_map = {}
    for group in groups:
        base_positions_map[group] = group_length(group) * length_factor
    unique_positions = np.asarray(
        sorted(set(base_positions_map.values())), dtype=float
    )

    def l_def_label(group):
        """目录名中的 L### 标签,如 MVH_L048_... -> L048。"""
        l_defs = {float(record["l_def"]) for record in grouped_data[group]}
        if len(l_defs) != 1:
            raise ValueError(f"{group} 内部的 l_def 不一致: {sorted(l_defs)}")
        l_def = l_defs.pop()
        if l_def.is_integer():
            return f"L{int(l_def):03d}"
        return f"L{l_def:g}"

    length_labels = {}
    for group in groups:
        length_labels[base_positions_map[group]] = l_def_label(group)
    if not np.all(np.isfinite(unique_positions)) or np.any(unique_positions <= 0):
        raise ValueError("物理长度必须为有限正数")
    seen = set()
    for group in groups:
        key = (base_positions_map[group], group_band(group))
        if key in seen:
            raise ValueError(
                f"存在多个配置对应相同物理长度和带边 ({key[0]:g} {unit_label}, "
                f"{key[1]})，无法区分"
            )
        seen.add(key)
    min_spacing = (
        float(np.min(np.diff(unique_positions)))
        if unique_positions.size > 1 else 1.0
    )

    # 同一物理长度有多个带边组时,给每组一个对称的水平偏移并收窄小提琴。
    groups_by_length = defaultdict(list)
    for group in groups:
        groups_by_length[base_positions_map[group]].append(group)
    max_shared = max(len(v) for v in groups_by_length.values())
    positions = []
    for group in groups:
        siblings = groups_by_length[base_positions_map[group]]
        index = siblings.index(group)
        offset = (index - (len(siblings) - 1) / 2.0) * 0.3 * min_spacing
        positions.append(base_positions_map[group] + offset)
    positions = np.asarray(positions, dtype=float)
    violin_width = min(0.45 * min_spacing,
                       0.9 * min_spacing / max_shared)

    datasets = []
    group_stats = []

    for group, position in zip(groups, positions):
        records = grouped_data[group]
        band = group_band(group)
        style = style_for_band(band)

        # 兼容旧格式 grouped_data[group] = [value, ...]。
        if records and isinstance(records[0], dict):
            transmissions = np.asarray(
                [record["value"] for record in records],
                dtype=float,
            )
        else:
            transmissions = np.asarray(records, dtype=float)

        # All four panels use the same samples. Never drop/clip zero values:
        # a zero conductance gives infinite resistance and cannot be drawn here.
        if (transmissions.size == 0 or not np.all(np.isfinite(transmissions))
                or np.any(transmissions <= 0.0)):
            raise ValueError(
                f"{group}: 四联图要求 G/G0 为有限正数；"
                "零电导对应无限电阻，不能静默删除或截断。"
            )
        # Apply one common mask in raw g space, before any log/reciprocal.
        # Do not iterate the filter or redefine it for individual panels.
        total_count = transmissions.size
        if exclude_outliers:
            q1, q3 = np.percentile(transmissions, [25, 75])
            iqr = q3 - q1
            keep = ((transmissions >= q1 - 1.5 * iqr)
                    & (transmissions <= q3 + 1.5 * iqr))
            transmissions = transmissions[keep]
            if transmissions.size == 0:
                raise ValueError(f"{group}: outlier 筛选后无剩余样本")
            print(f"[FILTER] {group}: total={total_count}, "
                  f"kept={transmissions.size}, "
                  f"excluded={total_count - transmissions.size}; "
                  "1.5*IQR on raw g")
        with np.errstate(over="ignore", divide="ignore"):
            sample_resistances = 1.0 / transmissions
            mean_g = float(np.mean(transmissions))
            mean_r = float(np.mean(sample_resistances))
        if not np.isfinite(mean_g) or not np.isfinite(mean_r):
            raise ValueError(f"{group}: 均值或倒数溢出，无法绘图")

        if resistance:
            if np.any(transmissions <= 0.0):
                raise ValueError(
                    f"{group} 含有 G/G0 <= 0 的数据，无法计算 R/R0=1/(G/G0)"
                )
            physical_values = sample_resistances
        else:
            physical_values = transmissions

        if log_scale:
            plot_values = np.log(physical_values)
        else:
            plot_values = physical_values

        datasets.append(plot_values)
        group_stats.append({
            "group": group,
            "band": band,
            "style": style,
            "base_position": base_positions_map[group],
            "position": position,
            "n": int(transmissions.size),
            "mean_g": mean_g,
            "mean_ln_g": float(np.mean(np.log(transmissions))),
            "mean_r": mean_r,
        })

        print(
            f"[GROUP] {group}: "
            f"L={group_length(group):.6g} Å, "
            f"n={len(transmissions)}, "
            f"mean={np.mean(transmissions):.6e}, "
            f"median={np.median(transmissions):.6e}"
        )

    fig_width = max(13.5, 1.25 * len(groups))
    fig = plt.figure(figsize=(fig_width, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0])
    ax = fig.add_subplot(grid[0, :])
    mean_ax = fig.add_subplot(grid[1, 0])
    log_ax = fig.add_subplot(grid[1, 1])
    resistance_ax = fig.add_subplot(grid[1, 2])

    # A KDE cannot represent a singleton or a constant dataset. Such groups
    # retain their mean markers, but have no artificial violin distribution.
    regular_indices = [
        i for i, values in enumerate(datasets)
        if values.size >= 2 and np.ptp(values) > 0
    ]
    if regular_indices:
        violin = ax.violinplot(
            [datasets[i] for i in regular_indices],
            positions=positions[regular_indices],
            widths=violin_width,
            showmeans=False, showmedians=False, showextrema=False,
        )
        areas = []
        for i, body in zip(regular_indices, violin["bodies"]):
            vertices = body.get_paths()[0].vertices
            # Center coordinates to avoid cancellation at large L or R.
            x = vertices[:, 0] - positions[i]
            y = vertices[:, 1] - vertices[0, 1]
            areas.append(0.5 * abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            ))
        positive_areas = [area for area in areas if area > 0 and np.isfinite(area)]
        target_area = min(positive_areas) if positive_areas else None
        for i, body, area in zip(regular_indices, violin["bodies"], areas):
            if target_area is not None and area > 0 and np.isfinite(area):
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = positions[i] + (
                    vertices[:, 0] - positions[i]
                ) * (target_area / area)
            style = group_stats[i]["style"]
            body.set_facecolor(style["violin"])
            body.set_edgecolor(style["violin"])
            body.set_linewidth(0.8)
            body.set_alpha(1.0)

    # 只显示 1.5×IQR 规则判定出的离群点。
    # 固定随机种子使离群点的横向位置每次一致。
    rng = np.random.default_rng(random_seed)
    for stats, values in zip(group_stats, datasets):
        position = stats["position"]
        style = stats["style"]
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = values[
            (values < lower_fence) | (values > upper_fence)
        ]

        if outliers.size and not exclude_outliers:
            jitter = rng.uniform(
                -0.08 * min_spacing,
                0.08 * min_spacing,
                size=outliers.size,
            )
            ax.scatter(
                position + jitter,
                outliers,
                s=18,
                color=style["accent"],
                alpha=0.72,
                linewidth=0,
                zorder=6,
            )

        # 均值：彩色短线穿过白心圆,标注数值;Ec 标在右侧,Ev 标在左侧。
        mean_value = float(np.mean(values))
        ax.hlines(
            mean_value,
            position - 0.35 * violin_width,
            position + 0.35 * violin_width,
            colors=style["accent"],
            linewidth=2.0,
            zorder=7,
        )
        ax.scatter(
            [position],
            [mean_value],
            s=42,
            facecolor="white",
            edgecolor=style["accent"],
            linewidth=2.0,
            zorder=8,
        )
        if stats["band"] == "Ev":
            xytext, ha = (-13, 0), "right"
        else:
            xytext, ha = (13, 0), "left"
        ax.annotate(
            f"{mean_value:.2f}",
            xy=(position, mean_value),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va="center",
            color=style["accent"],
            fontsize=9,
            zorder=9,
        )

        # 样本数标在坐标系顶部,避免与小提琴和均值标注重叠。
        ax.annotate(
            f"n={stats['n']}",
            xy=(position, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=style["accent"],
            fontsize=8,
            annotation_clip=False,
            zorder=9,
        )

    ax.margins(y=0.06)

    ax.set_xticks(unique_positions)
    ax.set_xticklabels(
        [f"{length_labels[position]}\n{position:g}"
         for position in unique_positions]
    )
    ax.set_xlabel(f"Defect-region length L ({unit_label})")

    if resistance and log_scale:
        ax.set_ylabel(r"$\ln(R/R_0)$")
    elif resistance:
        ax.set_ylabel(r"$R/R_0 = 1/(G/G_0)$")
    elif log_scale:
        ax.set_ylabel(r"$\ln(G/G_0)$")
    else:
        ax.set_ylabel(r"$G/G_0$")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
        zorder=0
    )

    bands_present = sorted(
        {stats["band"] for stats in group_stats if stats["band"] is not None},
        key=lambda b: BAND_ORDER.get(b, 0),
    )
    if bands_present:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=style_for_band(b)["violin"],
                      edgecolor=style_for_band(b)["accent"],
                      label=b)
                for b in bands_present
            ],
            loc="best",
            framealpha=0.9,
        )

    total_samples = sum(stats["n"] for stats in group_stats)
    ax.set_title(
        f"(a) Sample distribution and means "
        f"(L count={unique_positions.size}, total n={total_samples})",
        loc="left",
        pad=16,
    )
    if exclude_outliers:
        fig.suptitle("Outliers excluded: 1.5 × IQR on raw g (same samples in all panels)",
                     fontsize=12)
    # Take the natural logarithm of each frame conductance BEFORE averaging.
    # The middle panel uses a linear axis because its values are already logs.
    series_bands = bands_present if bands_present else [None]
    for band in series_bands:
        style = style_for_band(band)
        series = [stats for stats in group_stats if stats["band"] == band]
        series.sort(key=lambda s: s["base_position"])
        xs = [s["base_position"] for s in series]
        label = band if band is not None else None
        for panel, values, title, ylabel in (
            (mean_ax, [s["mean_g"] for s in series], "(b) Mean conductance",
             r"$\langle G/G_0\rangle$"),
            (log_ax, [s["mean_ln_g"] for s in series], "(c) Mean log conductance",
             r"$\langle\ln(G/G_0)\rangle$"),
            (resistance_ax, [s["mean_r"] for s in series], "(d) Mean resistance",
             r"$\langle R/R_0\rangle=\langle 1/g\rangle$"),
        ):
            panel.plot(
                xs, values, "o", color=style["accent"],
                markerfacecolor="white", markeredgewidth=1.6,
                markersize=5.5, linewidth=1.5, label=label,
            )
    for panel, title, ylabel in (
        (mean_ax, "(b) Mean conductance", r"$\langle G/G_0\rangle$"),
        (log_ax, "(c) Mean log conductance", r"$\langle\ln(G/G_0)\rangle$"),
        (resistance_ax, "(d) Mean resistance",
         r"$\langle R/R_0\rangle=\langle 1/g\rangle$"),
    ):
        panel.set_title(title, loc="left", fontsize=11, pad=10)
        panel.set_xlabel(f"Defect-region length L ({unit_label})")
        panel.set_ylabel(ylabel)
        panel.grid(axis="y", linestyle="--", alpha=0.3)
        panel.set_axisbelow(True)
        # Use automatic ticks in the small panels to prevent crowding.
        panel.margins(x=0.06, y=0.10)
        if bands_present:
            panel.legend(fontsize=9)
    log_ax.set_yscale("linear")
    mean_ax.set_ylim(bottom=0)
    resistance_ax.set_ylim(bottom=0)


    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"[DONE] 图片已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="读取 dpnegf 结果：上排小提琴图，下排电导均值、自然对数电导均值、电阻均值"
    )

    parser.add_argument(
        "root",
        help="需要递归搜索的根目录"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dpnegf_transport_summary.png",
        help="输出图片路径，默认 dpnegf_transport_summary.png；也支持 PDF/SVG"
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=0.0,
        help="化学势位置,默认 0.0 eV;取 conductance 列表中 mu_eV 最接近该值的条目"
             "(--band-edge 模式下忽略)",
    )
    parser.add_argument(
        "--band-edge",
        action="store_true",
        help="半导体管带边模式:读取 conductance 列表中 label 为 Ec 和 Ev 的条目,"
             "每个 negf.out.pth 产生两条记录,按带边分组绘图",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="仅顶部小提琴图使用原始值；默认顶部绘制自然对数"
    )
    parser.add_argument(
        "--R",
        action="store_true",
        help="仅顶部改为电阻小提琴图；默认取 ln，配合 --linear 画线性电阻",
    )
    parser.add_argument(
        "--length-unit", choices=["A", "nm"], default="A",
        help="所有子图的长度单位，默认 A（Å），可选 nm",
    )
    parser.add_argument(
        "--exclude-outliers", action="store_true",
        help="按各长度组原始 g 的 1.5×IQR 排除离群样本；四图共用筛选结果，默认不排除",
    )
    args = parser.parse_args()

    grouped_data = collect_results(
        root=args.root,
        fermi_energy=args.fermi,
        band_edge=args.band_edge,
    )

    plot_transport_summary(
        grouped_data=grouped_data,
        output_path=args.output,
        log_scale=not args.linear,
        resistance=args.R,
        length_unit=args.length_unit,
        exclude_outliers=args.exclude_outliers,
    )


if __name__ == "__main__":
    main()
