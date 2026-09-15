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


def to_numpy(value):
    """兼容 torch.Tensor、numpy.ndarray 和普通列表。"""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_conductance(pth_path, fermi_energy=0.0):
    """
    读取 T_avg，并在 fermi_energy 处线性插值。

    自动识别 T_avg 中长度与 uni_grid 相同的能量轴，
    其他维度才进行平均。
    """
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