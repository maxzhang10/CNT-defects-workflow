#!/usr/bin/env python3
"""CNT transport summary: a full-width violin panel above three mean curves.

Usage:
    python transport_four_panels.py /path/to/5_5 --linear --length-unit nm
    python transport_four_panels.py /path/to/5_5 -o transport_summary.pdf

The original log-first / PTH-fallback collection workflow is preserved.
Each collected record carries equal weight, as in the original script.
Top: original violin, transformed by --linear and --R.
Bottom: mean(g), mean(ln(g)) on a linear axis, and mean(1/g), g=G/G0.
Each replica represents one structural frame and one NEGF conductance.
The last panel uses R0=1/G0; it is NOT 1/mean(g).
PTH input remains T(E_F), without finite-temperature energy integration.
Only load trusted PTH files (torch.load uses weights_only=False).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def to_numpy(value):
    """兼容 torch.Tensor、numpy.ndarray 和普通列表。"""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_conductance(pth_path, fermi_energy=0.0):
    """
    读取 T_avg，并在 fermi_energy 处线性插值。

    自动识别 T_avg 中长度与 uni_grid 相同的能量轴，
    其他维度才进行平均。
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

    if "uni_grid" not in data:
        raise KeyError(f"缺少 uni_grid: {pth_path}")

    if "T_avg" not in data:
        raise KeyError(f"缺少 T_avg: {pth_path}")

    energy = np.asarray(
        to_numpy(data["uni_grid"]),
        dtype=float
    ).squeeze()

    transmission = np.asarray(
        to_numpy(data["T_avg"]),
        dtype=float
    ).squeeze()

    energy = np.ravel(energy)

    if energy.ndim != 1 or energy.size < 2:
        raise ValueError(
            f"uni_grid 形状异常: {data['uni_grid'].shape}"
        )

    # 一维 T_avg：最常见情况
    if transmission.ndim == 1:
        if transmission.size != energy.size:
            raise ValueError(
                f"长度不一致: energy={energy.size}, "
                f"T_avg={transmission.size}, file={pth_path}"
            )

    else:
        # 找出长度等于能量点数的轴
        candidate_axes = [
            axis
            for axis, size in enumerate(transmission.shape)
            if size == energy.size
        ]

        if len(candidate_axes) != 1:
            raise ValueError(
                f"无法唯一确定能量轴: "
                f"uni_grid.shape={energy.shape}, "
                f"T_avg.shape={transmission.shape}, "
                f"candidate_axes={candidate_axes}, "
                f"file={pth_path}"
            )

        energy_axis = candidate_axes[0]

        # 把能量轴移到最后
        transmission = np.moveaxis(
            transmission,
            energy_axis,
            -1
        )

        # 只平均非能量维度
        if transmission.ndim > 1:
            transmission = transmission.mean(
                axis=tuple(range(transmission.ndim - 1))
            )

    transmission = np.ravel(transmission)

    if transmission.size != energy.size:
        raise ValueError(
            f"处理后长度仍不一致: "
            f"energy={energy.size}, "
            f"T={transmission.size}, file={pth_path}"
        )

    valid = (
        np.isfinite(energy)
        & np.isfinite(transmission)
    )

    energy = energy[valid]
    transmission = transmission[valid]

    if energy.size < 2:
        raise ValueError(f"有效能量点不足: {pth_path}")

    # np.interp 要求能量升序
    order = np.argsort(energy)
    energy = energy[order]
    transmission = transmission[order]

    if not energy[0] <= fermi_energy <= energy[-1]:
        raise ValueError(
            f"E_F={fermi_energy} eV 超出能量范围 "
            f"[{energy[0]}, {energy[-1]}]: {pth_path}"
        )

    conductance = float(
        np.interp(
            fermi_energy,
            energy,
            transmission
        )
    )

    nearest_index = int(
        np.argmin(np.abs(energy - fermi_energy))
    )
    nearest_energy = float(energy[nearest_index])
    nearest_value = float(transmission[nearest_index])

    return conductance, nearest_energy, nearest_value

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


def collect_results(root, fermi_energy=0.0):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root 目录不存在: {root}")

    grouped_data = {}
    logged_config_dirs = set()
    logs = sorted(root.rglob("replica_conductance.log"))
    print(f"[INFO] root: {root}")
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

            transmission, nearest_energy, nearest_value = load_conductance(
                pth_path, fermi_energy=fermi_energy
            )
            if transmission < 0:
                raise ValueError(f"插值后的透射系数为负: {transmission}")

            group = get_group_name(pth_path)
            sample = get_sample_name(pth_path)
            length, l_def, n, m, unit_cell_length = length_from_config_dir(config_dir)
            grouped_data.setdefault(group, []).append(
                {
                    "value": transmission,
                    "sample": sample,
                    "length_A": length,
                    "l_def": l_def,
                    "chirality": (n, m),
                    "unit_cell_length_A": unit_cell_length,
                }
            )
            print(
                f"[PTH] {group:<30} sample={sample:<10} "
                f"T_interp({fermi_energy:g})={transmission:.12g}  "
                f"T_nearest({nearest_energy:.6g})={nearest_value:.12g}"
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

    groups = sorted(grouped_data, key=group_length)
    if length_unit not in {"A", "nm"}:
        raise ValueError("length_unit 必须为 A 或 nm")
    length_factor = 0.1 if length_unit == "nm" else 1.0
    unit_label = "nm" if length_unit == "nm" else "Å"
    positions = np.asarray(
        [group_length(group) * length_factor for group in groups], dtype=float
    )
    if not np.all(np.isfinite(positions)) or np.any(positions <= 0):
        raise ValueError("物理长度必须为有限正数")
    if np.unique(positions).size != positions.size:
        raise ValueError(
            "存在多个配置对应相同物理长度，无法在同一横坐标位置绘制独立大提琴"
        )
    min_spacing = float(np.min(np.diff(positions))) if positions.size > 1 else 1.0
    violin_width = 0.45 * min_spacing
    datasets = []
    mean_conductances = []
    mean_log_conductances = []
    mean_resistances = []

    for group in groups:
        records = grouped_data[group]

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
        with np.errstate(over="ignore", divide="ignore"):
            sample_resistances = 1.0 / transmissions
            mean_g = float(np.mean(transmissions))
            mean_r = float(np.mean(sample_resistances))
        if not np.isfinite(mean_g) or not np.isfinite(mean_r):
            raise ValueError(f"{group}: 均值或倒数溢出，无法绘图")
        mean_conductances.append(mean_g)
        mean_log_conductances.append(float(np.mean(np.log(transmissions))))
        mean_resistances.append(mean_r)

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

    # 颜色直接取自参考图。
    violin_color = "#9ACDCD"
    accent_color = "#008080"

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
            body.set_facecolor(violin_color)
            body.set_edgecolor(violin_color)
            body.set_linewidth(0.8)
            body.set_alpha(1.0)

    # 只显示 1.5×IQR 规则判定出的离群点。
    # 固定随机种子使离群点的横向位置每次一致。
    rng = np.random.default_rng(random_seed)
    for position, values in zip(
        positions,
        datasets,
    ):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = values[
            (values < lower_fence) | (values > upper_fence)
        ]

        if outliers.size:
            jitter = rng.uniform(
                -0.08 * min_spacing,
                0.08 * min_spacing,
                size=outliers.size,
            )
            ax.scatter(
                position + jitter,
                outliers,
                s=18,
                color=accent_color,
                alpha=0.72,
                linewidth=0,
                zorder=6,
            )

        # 均值：青绿色短线穿过白心圆，右侧标注数值。
        mean_value = float(np.mean(values))
        ax.hlines(
            mean_value,
            position - 0.10 * min_spacing,
            position + 0.10 * min_spacing,
            colors=accent_color,
            linewidth=2.0,
            zorder=7,
        )
        ax.scatter(
            [position],
            [mean_value],
            s=42,
            facecolor="white",
            edgecolor=accent_color,
            linewidth=2.0,
            zorder=8,
        )
        ax.annotate(
            f"{mean_value:.2f}",
            xy=(position, mean_value),
            xytext=(13, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=accent_color,
            fontsize=9,
            zorder=9,
        )

    ax.margins(y=0.06)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{position:g}" for position in positions])
    ax.set_xlabel(f"Defect-region length L ({unit_label})")

    if resistance and log_scale:
        ax.set_ylabel(r"$\ln(R/R_0)$")
    elif resistance:
        ax.set_ylabel(r"$R/R_0 = 1/[T(E_F)]$")
    elif log_scale:
        ax.set_ylabel(r"$\ln(G/G_0)$")
    else:
        ax.set_ylabel(r"$G/G_0=T(E_F)$")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
        zorder=0
    )

    ax.set_title("(a) Sample distribution and means", loc="left", pad=12)
    # Take the natural logarithm of each frame conductance BEFORE averaging.
    # The middle panel uses a linear axis because its values are already logs.
    for panel, values, title, ylabel in (
        (mean_ax, mean_conductances, "(b) Mean conductance",
         r"$\langle G/G_0\rangle$"),
        (log_ax, mean_log_conductances, "(c) Mean log conductance",
         r"$\langle\ln(G/G_0)\rangle$"),
        (resistance_ax, mean_resistances, "(d) Mean resistance",
         r"$\langle R/R_0\rangle=\langle 1/g\rangle$"),
    ):
        panel.plot(
            positions, values, "o", color=accent_color,
            markerfacecolor="white", markeredgewidth=1.6,
            markersize=5.5, linewidth=1.5,
        )
        panel.set_title(title, loc="left", fontsize=11, pad=10)
        panel.set_xlabel(f"Defect-region length L ({unit_label})")
        panel.set_ylabel(ylabel)
        panel.grid(axis="y", linestyle="--", alpha=0.3)
        panel.set_axisbelow(True)
        # Use automatic ticks in the small panels to prevent crowding.
        panel.margins(x=0.06, y=0.10)
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
        help="费米能级位置，默认 0.0 eV"
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
    args = parser.parse_args()

    grouped_data = collect_results(
        root=args.root,
        fermi_energy=args.fermi
    )

    plot_transport_summary(
        grouped_data=grouped_data,
        output_path=args.output,
        log_scale=not args.linear,
        resistance=args.R,
        length_unit=args.length_unit,
    )


if __name__ == "__main__":
    main()
