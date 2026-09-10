#!/usr/bin/env python3
"""
递归读取 replica_conductance.log，提取 <ln(G/G0)>-L 拟合点。

本脚本只整理数据和绘制误差棒散点，不执行局域化长度拟合。

横坐标：
    L = N_defects / density，单位为 Å

纵坐标：
    y = mean[ln(G/G0)]

误差棒：
    SEM = std[ln(G/G0)] / sqrt(N_valid)

电导模式（由日志中的 conductance_mode 字段决定）：
    fermi          —— 使用通用 mean ln(G/G0) / std ln(G/G0)。
    band_edge_bias —— 不使用通用 Fermi 字段（它对应 Fermi 能级线性响应），
                      而是分别读取 <ln Gc>/std ln Gc 和 <ln Gv>/std ln Gv，
                      为导带 (conduction) 与价带 (valence) 各产生一个数据点。
                      两个通道始终独立，不合并。
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def extract_number(text, label, value_pattern=FLOAT_RE):
    """从日志中提取指定标签后的数值。"""
    match = re.search(
        rf"(?m)^\s*{re.escape(label)}\s*:\s*({value_pattern})\s*$",
        text,
    )
    if match is None:
        raise ValueError(f"缺少字段: {label}")
    return match.group(1)


def extract_text(text, label):
    """从日志中提取指定标签后的文本。"""
    match = re.search(
        rf"(?m)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
        text,
    )
    if match is None:
        raise ValueError(f"缺少字段: {label}")
    return match.group(1)


def _base_point(log_path, text):
    """提取日志中与电导通道无关的公共字段。"""
    configuration_dir = extract_text(text, "configuration_dir")
    temperature_K = float(extract_number(text, "temperature_K"))
    structures = extract_text(text, "structures")
    l_def = float(extract_number(text, "l_def"))
    n_defects = int(float(extract_number(text, "N_defects")))
    density = float(extract_number(text, "density_A^-1"))
    n_valid = int(float(extract_number(text, "valid replicas")))
    conductance_mode = extract_text(text, "conductance_mode")

    if density <= 0:
        raise ValueError(f"density_A^-1 必须大于 0，当前为 {density}")
    if n_valid <= 0:
        raise ValueError(f"valid replicas 必须大于 0，当前为 {n_valid}")

    length_A = n_defects / density

    return {
        "temperature_K": temperature_K,
        "structures": structures,
        "l_def": l_def,
        "N_defects": n_defects,
        "density_A^-1": density,
        "length_A": length_A,
        "length_nm": length_A / 10.0,
        "valid_replicas": n_valid,
        "configuration_dir": configuration_dir,
        "log_file": str(log_path.resolve()),
        "conductance_mode": conductance_mode,
    }


def _make_point(base, text, channel, mean_ln_label, std_ln_label):
    """由公共字段 + 指定通道的 <ln G>/std ln G 构造一个拟合数据点。"""
    mean_ln = float(extract_number(text, mean_ln_label))
    std_ln = float(extract_number(text, std_ln_label))
    n_valid = base["valid_replicas"]
    point = dict(base)
    point["channel"] = channel
    point["mean_ln_G_over_G0"] = mean_ln
    point["std_ln_G_over_G0"] = std_ln
    point["sem_ln_G_over_G0"] = std_ln / math.sqrt(n_valid)
    return point


def parse_log(log_path):
    """将一个 replica_conductance.log 转换为拟合数据点列表。

    返回列表是因为 band_edge_bias 模式下导带、价带各产生一个点。
    fermi 模式返回单个点（channel="fermi"），仍使用通用 mean/std ln(G/G0)。
    band_edge_bias 模式不读取通用 Fermi 字段，分别读取 <ln Gc>/<ln Gv> 及
    其 std，两个通道独立不合并。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    base = _base_point(log_path, text)
    mode = base["conductance_mode"]

    if mode == "band_edge_bias":
        gc_point = _make_point(base, text, "conduction", "<ln Gc>", "std ln Gc")
        gv_point = _make_point(base, text, "valence", "<ln Gv>", "std ln Gv")
        return [gc_point, gv_point]

    if mode == "fermi":
        return [_make_point(base, text, "fermi", "mean ln(G/G0)", "std ln(G/G0)")]

    raise ValueError(f"未知 conductance_mode: {mode}")


def collect_points(root, temperatures=None, structure=None, channels=None):
    """递归查找日志，并按筛选条件返回拟合点。

    channels: 可选通道筛选集合，例如 {"fermi"} 或 {"conduction", "valence"}。
    默认不筛选，返回日志中出现的所有通道。
    band_edge_bias 日志产生 conduction + valence 两个点；fermi 日志产生 fermi 点。
    """
    points = []
    failures = []

    for log_path in sorted(root.rglob("replica_conductance.log")):
        try:
            log_points = parse_log(log_path)
        except (OSError, ValueError) as exc:
            failures.append((log_path, str(exc)))
            continue

        for point in log_points:
            if temperatures is not None:
                if not any(
                    math.isclose(point["temperature_K"], t, abs_tol=1e-8)
                    for t in temperatures
                ):
                    continue

            if structure is not None and point["structures"] != structure:
                continue

            if channels is not None and point["channel"] not in channels:
                continue

            points.append(point)

    points.sort(key=lambda p: (p["channel"], p["temperature_K"], p["length_A"]))
    return points, failures


def write_csv(points, output_path):
    """写出可直接导入 Origin 或供后续 Python 拟合使用的表格。"""
    fieldnames = [
        "channel",
        "conductance_mode",
        "temperature_K",
        "structures",
        "l_def",
        "N_defects",
        "density_A^-1",
        "length_A",
        "length_nm",
        "mean_ln_G_over_G0",
        "std_ln_G_over_G0",
        "sem_ln_G_over_G0",
        "valid_replicas",
        "configuration_dir",
        "log_file",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def plot_points(points, output_path):
    """按 (通道, 温度) 绘制 <ln(G/G0)>-L 误差棒散点，不进行拟合。

    不同通道用不同标记区分，温度用颜色区分，使导带/价带/Fermi 三个
    通道的散点在同一图中可辨。
    """
    channels = sorted({p["channel"] for p in points})
    channel_markers = {"fermi": "o", "conduction": "s", "valence": "^"}
    temperatures = sorted({p["temperature_K"] for p in points})
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    color_index = {temperature: i for i, temperature in enumerate(temperatures)}
    for channel in channels:
        marker = channel_markers.get(channel, "o")
        for temperature in temperatures:
            group = [
                p for p in points
                if p["channel"] == channel
                and math.isclose(p["temperature_K"], temperature, abs_tol=1e-8)
            ]
            if not group:
                continue

            x = [p["length_nm"] for p in group]
            y = [p["mean_ln_G_over_G0"] for p in group]
            yerr = [p["sem_ln_G_over_G0"] for p in group]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=marker,
                markersize=5,
                capsize=3,
                linestyle="none",
                color=colors(color_index[temperature] % 10),
                label=f"{channel} {temperature:g} K",
            )

    ax.set_xlabel(r"Defective-region length $L$ (nm)")
    ax.set_ylabel(r"$\langle\ln(G/G_0)\rangle$")
    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_points(points):
    """在终端列出最关键的拟合输入。"""
    print(
        f"{'channel':>11}  {'T (K)':>8}  {'L (nm)':>12}  "
        f"{'<ln(G/G0)>':>14}  {'SEM':>12}  {'N':>5}"
    )
    print("-" * 71)
    for point in points:
        print(
            f"{point['channel']:>11}  "
            f"{point['temperature_K']:8.1f}  "
            f"{point['length_nm']:12.6f}  "
            f"{point['mean_ln_G_over_G0']:14.8f}  "
            f"{point['sem_ln_G_over_G0']:12.8f}  "
            f"{point['valid_replicas']:5d}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="提取 <ln(G/G0)>-L 拟合点，不执行拟合。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="包含各温度和长度目录的数据根目录",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[50.0, 300.0, 700.0],
        help="需要保留的温度，默认: 50 300 700",
    )
    parser.add_argument(
        "--structure",
        default=None,
        help="只保留指定缺陷类型，例如 5775；默认不筛选",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        choices=["fermi", "conduction", "valence"],
        help=(
            "只保留指定电导通道。默认不筛选，返回日志中出现的所有通道"
            "（fermi 模式: fermi；band_edge_bias 模式: conduction + valence）。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("localization_points.csv"),
        help="输出 CSV，默认: localization_points.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("localization_points.png"),
        help="输出散点图，默认: localization_points.png",
    )
    args = parser.parse_args()

    points, failures = collect_points(
        root=args.root,
        temperatures=args.temperatures,
        structure=args.structure,
        channels=set(args.channels) if args.channels else None,
    )

    if not points:
        raise SystemExit(
            "没有提取到数据点，请检查 --root、温度、缺陷类型和日志文件名。"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    write_csv(points, args.output)
    plot_points(points, args.figure)
    print_points(points)

    print(f"\n[DONE] points = {len(points)}")
    print(f"[DONE] csv    = {args.output.resolve()}")
    print(f"[DONE] figure = {args.figure.resolve()}")

    if failures:
        print(f"\n[WARNING] 有 {len(failures)} 个日志解析失败：")
        for log_path, reason in failures:
            print(f"  {log_path}: {reason}")


if __name__ == "__main__":
    main()
