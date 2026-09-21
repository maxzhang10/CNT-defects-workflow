#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_conductance(pth_path, fermi_energy=0.0):
    """
    从新版 DPNEGF 的 negf.out.pth 读取预计算电导。

    新版 DPNEGF 把电导结果预计算后写入文件顶层的 ``conductance`` 列表，
    每项含 ``label``（"Ef" / "Ev" / "Ec"）、``G_over_G0``、``mu_absolute_eV``。
    这里按标签读取 ``G_over_G0``，不再自己用 T_avg 插值——后者在新版格式里
    是相对 E_ref 的能量网格，直接拿外部 fermi_energy 插值既不对也不需要。

    标签选择优先级：
      1. 若 conductance 列表里存在 "Ef" 项，用它（费米模式）；
      2. 否则若存在 "Ec" 和 "Ev"（带边模式），取两者 G_over_G0 的算术平均，
         作为该 replica 的代表电导（与 collect_md_conductance.py 的
         band_edge_bias 口径一致）；
      3. 都没有则报错，附文件路径与找到的标签。

    fermi_energy 参数仅为保持调用接口兼容保留，不再用于插值；
    返回的 nearest_energy / nearest_value 取自匹配条目的 mu_absolute_eV /
    G_over_G0，便于日志中沿用既有输出格式。
    """
    data = torch.load(
        pth_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, dict):
        raise TypeError(f"文件内容不是字典: {pth_path}")

    entries = data.get("conductance")
    if not isinstance(entries, list) or not entries:
        raise KeyError(
            f"缺少或为空的 conductance 列表: {pth_path}"
        )

    by_label = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if isinstance(label, str) and "G_over_G0" in entry:
            by_label[label] = entry

    if "Ef" in by_label:
        entry = by_label["Ef"]
        conductance = float(entry["G_over_G0"])
        nearest_energy = float(entry.get("mu_absolute_eV", fermi_energy))
        nearest_value = conductance
    elif "Ec" in by_label and "Ev" in by_label:
        gc = float(by_label["Ec"]["G_over_G0"])
        gv = float(by_label["Ev"]["G_over_G0"])
        conductance = (gc + gv) / 2.0
        # 带边模式没有单一 Ef；用 Ec 的 mu 作代表能量，G_over_G0 取均值。
        nearest_energy = float(by_label["Ec"].get("mu_absolute_eV", fermi_energy))
        nearest_value = conductance
    else:
        found = sorted(by_label.keys())
        raise KeyError(
            f"未找到 Ef 或 (Ec+Ev) 电导条目: {pth_path}; 找到的标签: {found}"
        )

    return conductance, nearest_energy, nearest_value

def get_group_name(pth_path):
    """
    同时兼容新旧目录结构。

    新目录：
        .../温度/手性/配置名/replica_001/dpnegf/200000/output/negf.out.pth

    旧目录：
        .../温度/手性/配置名/dpnegf/40000/output/negf.out.pth

    P2-10：分组键必须包含温度和手性，否则不同温度/手性的同名配置
    （如 50K/5_5/DV_DV 与 300K/5_5/DV_DV）会进入同一把 violin。

    返回分组键，例如：
        500K_5_5_5775_L008_0.1016A-1
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
        config_name = parts[dpnegf_index - 2]
        # 向上找温度（NNNK）和手性（M_N）。
        prefix_start = dpnegf_index - 2
    else:
        config_name = parent_name
        prefix_start = dpnegf_index - 1

    temperature = None
    chirality = None
    for j in range(prefix_start - 1, -1, -1):
        part = parts[j]
        if temperature is None and _looks_like_temperature(part):
            temperature = part
        if chirality is None and _looks_like_chirality(part):
            chirality = part
        if temperature is not None and chirality is not None:
            break

    prefix = ""
    if temperature:
        prefix += temperature
    if chirality:
        prefix += "_" + chirality if prefix else chirality
    if prefix:
        return f"{prefix}_{config_name}"
    return config_name


def _looks_like_temperature(name: str) -> bool:
    """匹配形如 '300K'、'500K' 的目录名。"""
    return name.endswith("K") and name[:-1].isdigit()


def _looks_like_chirality(name: str) -> bool:
    """匹配形如 '5_5'、'9_0' 的手性目录名。"""
    parts = name.split("_")
    return len(parts) == 2 and all(p.isdigit() for p in parts)


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


def collect_results(root, fermi_energy=0.0):
    root = Path(root).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"root 目录不存在: {root}")

    result_files = sorted(
        path
        for path in root.rglob("negf.out.pth")
        if is_dpnegf_result(path)
    )

    print(f"[INFO] root: {root}")
    print(f"[INFO] 找到 {len(result_files)} 个严格匹配的 dpnegf 结果")

    grouped_data = {}

    for pth_path in result_files:
        try:
            transmission, nearest_energy, nearest_value = (
                load_conductance(
                    pth_path,
                    fermi_energy=fermi_energy
                )
            )

            if transmission < 0:
                raise ValueError(
                    f"插值后的透射系数为负: {transmission}"
                )

            group = get_group_name(pth_path)
            grouped_data.setdefault(group, []).append(transmission)

            sample = get_sample_name(pth_path)

            print(
                f"[OK] {group:<30} "
                f"sample={sample:<10} "
                f"T_interp({fermi_energy:g})={transmission:.12g}  "
                f"T_nearest({nearest_energy:.6g})="
                f"{nearest_value:.12g}"
            )

        except Exception as exc:
            print(f"[SKIP] {pth_path}")
            print(f"       {type(exc).__name__}: {exc}")

    return grouped_data

def is_dpnegf_result(path):
    """
    同时接受两种合法布局：
      有 MD 步长：.../配置名/dpnegf/采样步/output/negf.out.pth
      md_steps=0：.../配置名/dpnegf/output/negf.out.pth

    P2-15：旧版只接受带采样步的布局（向上三级为 dpnegf），
    无 MD 路径把结果放在 dpnegf/output/negf.out.pth（向上两级为 dpnegf），
    被静默排除。
    """
    if path.name != "negf.out.pth":
        return False
    if path.parent.name != "output":
        return False
    # 有采样步：parent.parent.parent = dpnegf
    if path.parent.parent.parent.name == "dpnegf":
        return True
    # md_steps=0：parent.parent = dpnegf
    if path.parent.parent.name == "dpnegf":
        return True
    return False

def plot_violin(grouped_data, output_path, log_scale=True):
    if not grouped_data:
        raise RuntimeError("没有读取到有效的 dpnegf 数据")

    groups = sorted(grouped_data)
    datasets = []

    for group in groups:
        transmissions = np.asarray(grouped_data[group], dtype=float)

        if log_scale:
            transmissions = np.clip(
                transmissions,
                np.finfo(float).tiny,
                None
            )
            plot_values = np.log(transmissions)
        else:
            plot_values = transmissions

        datasets.append(plot_values)

        print(
            f"[GROUP] {group}: "
            f"n={len(transmissions)}, "
            f"median={np.median(transmissions):.6e}"
        )

    positions = np.arange(1, len(groups) + 1)
    fig_width = max(8, 1.25 * len(groups))

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # 大提琴概率密度
    violin = ax.violinplot(
        datasets,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # 计算每把大提琴当前的几何面积
    areas = []

    for body in violin["bodies"]:
        vertices = body.get_paths()[0].vertices
        x = vertices[:, 0]
        y = vertices[:, 1]

        area = 0.5 * np.abs(
            np.dot(x, np.roll(y, 1))
            - np.dot(y, np.roll(x, 1))
        )
        areas.append(area)

    # 以最小面积为目标，避免放大后相互重叠
    target_area = min(areas)

    # 沿横向缩放，使所有大提琴面积相同
    for position, body, area in zip(
        positions,
        violin["bodies"],
        areas
    ):
        vertices = body.get_paths()[0].vertices

        scale = target_area / area
        vertices[:, 0] = (
            position
            + (vertices[:, 0] - position) * scale
        )

        body.set_facecolor("#72A7D8")
        body.set_edgecolor("#356B9A")
        body.set_linewidth(1.0)
        body.set_alpha(0.80)

    for body in violin["bodies"]:
        body.set_facecolor("#72A7D8")
        body.set_edgecolor("#356B9A")
        body.set_linewidth(1.0)
        body.set_alpha(0.80)

    # 计算中位数、四分位数和完整数据范围
    medians = np.array([
        np.median(values) for values in datasets
    ])
    quartile1 = np.array([
        np.percentile(values, 25) for values in datasets
    ])
    quartile3 = np.array([
        np.percentile(values, 75) for values in datasets
    ])
    data_min = np.array([
        np.min(values) for values in datasets
    ])
    data_max = np.array([
        np.max(values) for values in datasets
    ])

    # 最小值—最大值范围线
    ax.vlines(
        positions,
        data_min,
        data_max,
        color="black",
        linewidth=1.2,
        zorder=3
    )

    # Q1—Q3 黑色四分位箱
    ax.vlines(
        positions,
        quartile1,
        quartile3,
        color="black",
        linewidth=8,
        zorder=4
    )

    # 白色中位数点
    ax.scatter(
        positions,
        medians,
        s=28,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=5
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_xlabel("Structure")

    if log_scale:
        ax.set_ylabel(r"$\ln(G/G_0)$")
    else:
        ax.set_ylabel(r"$G/G_0=T(E_F)$")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
        zorder=0
    )

    fig.tight_layout()

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
        description="读取 dpnegf 结果并绘制费米能级电导大提琴图"
    )

    parser.add_argument(
        "root",
        help="需要递归搜索的根目录"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dpnegf_violin.png",
        help="输出图片路径，默认 dpnegf_violin.png"
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
        help="使用线性 G/G0，而不是 ln(G/G0)"
    )

    args = parser.parse_args()

    grouped_data = collect_results(
        root=args.root,
        fermi_energy=args.fermi
    )

    plot_violin(
        grouped_data=grouped_data,
        output_path=args.output,
        log_scale=not args.linear
    )


if __name__ == "__main__":
    main()