#!/usr/bin/env python3
"""
统计电导：遍历 data_root 下所有 DPNEGF 结果，抽取费米能级处的透射作为电导，
汇总成一张 tidy CSV。

设计要点
--------
- **只读**：绝不改动、删除任何计算结果，可反复重跑。
- **独立于主流水线**：electron 输运结果 `output/negf.out.pth` 是 torch pickle，
  必须用 DeePTB 的 venv 跑（base 环境没有 torch）：
      source /personal/soft/DeePTB/.venv/bin/activate
      python collect_conductance.py --root <data_root>
  因此本步骤不 wire 进 run.py（venv 冲突 + 昂贵计算与廉价分析解耦）。
- **电导定义**：G = T_avg 在 uni_grid 的 E = E_F（默认 0，能量轴已相对费米能）
  处的值，单位 G0 = 2e^2/h。格点未必正好落在 0，用线性插值。
- **只依赖 torch + numpy**：单位胞长度公式内联，不 import ASE / cnt_geometry，
  以免污染 DeePTB venv。

目录契约（与 run.py 一致）
--------------------------
    <data_root>/<T>K/<m>_<n>/<STRUCTURE>/dpnegf/<timestep>/output/negf.out.pth

STRUCTURE 命名两种：
- 单/双缺陷：如 MVH_DV、DV_DV、5775_5775、P —— 缺陷数 = 下划线分段数（P=0）。
  线密度需要 l_def（从 --config 读）与手性，density = n_defects / (l_def * T)。
- 多缺陷：如 MVH_DV_5775_Dens_0.49Å-1 —— 线密度直接内嵌在名字里（`_Dens_<x>Å-1`），
  此时 Dens 前的 token 是缺陷"类型列表"而非个数，故 n_defects 置空。
"""
import os
import re
import csv
import math
import argparse
from math import gcd
from pathlib import Path

import numpy as np
import torch

import logkit as L


# ------------------------------------------------------------------
# 几何：单位胞长度（内联 cnt_geometry.calculate_unit_cell_length，避免依赖 ASE）
# ------------------------------------------------------------------
def unit_cell_length(m, n):
    a = math.sqrt(3) * 1.42  # 石墨烯晶格常数（√3 × C-C 键长），Å
    d_R = gcd(2 * m + n, 2 * n + m)
    return math.sqrt(3) * a * math.sqrt(m * m + n * n + m * n) / d_R


# ------------------------------------------------------------------
# 电导：T_avg 在 E_F 处插值
# ------------------------------------------------------------------
def conductance_from_pth(pth_path, e_fermi=0.0):
    """
    读 negf.out.pth，返回 (G/G0, E_F 处透射)。

    negf.out.pth 结构（实测）：
        uni_grid : Tensor(N,)  能量轴，已相对 E_F，单位 eV
        T_avg    : Tensor(N,)  平均透射谱
    电导取 E = e_fermi 处的透射（线性插值）。单位即 G0 = 2e^2/h。
    """
    d = torch.load(pth_path, map_location="cpu", weights_only=False)

    if "uni_grid" not in d or "T_avg" not in d:
        raise KeyError(
            f"negf.out.pth 缺少 uni_grid / T_avg 键：{pth_path}（现有键 {list(d.keys())}）"
        )

    grid = np.asarray(d["uni_grid"], dtype=float)
    t_avg = np.asarray(d["T_avg"], dtype=float)

    order = np.argsort(grid)  # np.interp 要求 x 递增
    grid = grid[order]
    t_avg = t_avg[order]

    if e_fermi < grid[0] or e_fermi > grid[-1]:
        raise ValueError(
            f"E_F={e_fermi} 超出能量轴范围 [{grid[0]:.3f}, {grid[-1]:.3f}]：{pth_path}"
        )

    g = float(np.interp(e_fermi, grid, t_avg))
    return g, g


# ------------------------------------------------------------------
# 路径元信息解析（以 'dpnegf' 为锚点做位置解析，绝不正则手性）
# ------------------------------------------------------------------
_DENS_RE = re.compile(r"_Dens_([0-9]*\.?[0-9]+)")


def parse_leaf_meta(pth_path):
    """
    从 .../<T>K/<m>_<n>/<STRUCTURE>/dpnegf/<timestep>/output/negf.out.pth
    解析元信息。以 'dpnegf' 段为锚点定位，避免把结构名当手性。
    """
    parts = Path(pth_path).resolve().parts

    if "dpnegf" not in parts:
        raise ValueError(f"路径中缺少 dpnegf 段：{pth_path}")

    i = parts.index("dpnegf")
    # 锚点前：... <T>K / <m>_<n> / <STRUCTURE> / dpnegf / <timestep> / output / negf.out.pth
    structure = parts[i - 1]
    chir_seg = parts[i - 2]
    temp_seg = parts[i - 3]
    timestep = parts[i + 1] if i + 1 < len(parts) else ""

    # 温度：'300K' -> 300
    m_temp = re.fullmatch(r"(\d+)K", temp_seg)
    temperature = int(m_temp.group(1)) if m_temp else None

    # 手性：'6_4' -> (6, 4)。仅接受纯 <int>_<int>，否则置空（不猜）。
    m_chir = re.fullmatch(r"(\d+)_(\d+)", chir_seg)
    chirality = (int(m_chir.group(1)), int(m_chir.group(2))) if m_chir else (None, None)

    # 线密度：多缺陷目录名内嵌 '_Dens_<x>Å-1'
    m_dens = _DENS_RE.search(structure)
    dens_from_name = float(m_dens.group(1)) if m_dens else None

    # 结构短名（去掉 _Dens_ 尾巴）
    structure_base = structure[: m_dens.start()] if m_dens else structure

    return {
        "temperature_K": temperature,
        "m": chirality[0],
        "n": chirality[1],
        "structure": structure,
        "structure_base": structure_base,
        "timestep": timestep,
        "dens_from_name": dens_from_name,
    }


def count_defects_from_base(structure_base):
    """
    单/双缺陷目录：缺陷数 = 下划线分段数；'P'（完美）= 0。
    仅在目录名没有 _Dens_ 内嵌密度时才用得上。
    """
    s = structure_base.strip("_")
    if s == "" or s.upper() == "P":
        return 0
    return len(s.split("_"))


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def find_negf_outputs(root):
    root = Path(root).resolve()
    return sorted(root.rglob("dpnegf/*/output/negf.out.pth"))


def main():
    ap = argparse.ArgumentParser(
        description="遍历 data_root，抽取每个 DPNEGF 结果 E_F 处电导，汇总成 CSV（只读）。"
    )
    ap.add_argument(
        "--root",
        default=None,
        help="数据根目录。缺省时从 --config 的 data_root 读。",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="config.json 路径（可选）。用于给单/双缺陷目录补算线密度所需的 l_def；"
        "缺省时按 CNT_CONFIG / ../config.json 解析。",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出 CSV 路径。缺省写到 <root>/conductance_summary.csv。",
    )
    ap.add_argument(
        "--e-fermi",
        type=float,
        default=0.0,
        help="费米能（相对 uni_grid，默认 0.0 eV）。",
    )
    args = ap.parse_args()

    # 解析 config（尽力而为，用于 l_def 补算线密度；失败不致命）
    config = None
    l_def = None
    try:
        from fdf2xyz import load_config

        config = load_config(args.config)
        l_def = int(config.get("l_def")) if config.get("l_def") is not None else None
    except Exception as e:  # noqa: BLE001
        L.warn(f"未能加载 config（单/双缺陷线密度将留空）：{e}")

    # 确定 root
    if args.root is not None:
        root = Path(args.root).resolve()
    elif config is not None and config.get("data_root"):
        root = Path(config["data_root"]).resolve()
    else:
        raise SystemExit("必须指定 --root，或让 --config 提供 data_root。")

    if not root.exists():
        raise SystemExit(f"root 不存在：{root}")

    out_path = Path(args.out).resolve() if args.out else (root / "conductance_summary.csv")

    L.info("[CONFIG]")
    L.info(f"root     = {root}")
    L.info(f"e_fermi  = {args.e_fermi}")
    L.info(f"l_def    = {l_def}")
    L.info(f"out      = {out_path}")

    pth_list = find_negf_outputs(root)
    L.info(f"找到 negf.out.pth: {len(pth_list)} 个")

    rows = []
    n_ok = 0
    n_fail = 0

    for i, pth in enumerate(pth_list, 1):
        L.item(i, len(pth_list), str(pth))
        try:
            meta = parse_leaf_meta(pth)
            g, t_ef = conductance_from_pth(pth, e_fermi=args.e_fermi)

            # 线密度：优先目录内嵌；否则用 l_def + 手性补算（单/双缺陷）
            linear_density = meta["dens_from_name"]
            n_defects = None
            scattering_len = None

            if linear_density is None:
                n_defects = count_defects_from_base(meta["structure_base"])
                if l_def is not None and meta["m"] is not None:
                    T = unit_cell_length(meta["m"], meta["n"])
                    scattering_len = l_def * T
                    linear_density = (
                        n_defects / scattering_len if scattering_len > 0 else None
                    )

            rows.append(
                {
                    "temperature_K": meta["temperature_K"],
                    "m": meta["m"],
                    "n": meta["n"],
                    "structure": meta["structure"],
                    "structure_base": meta["structure_base"],
                    "n_defects": n_defects,
                    "scattering_length_A": (
                        round(scattering_len, 4) if scattering_len else None
                    ),
                    "linear_density_A-1": (
                        round(linear_density, 4) if linear_density is not None else None
                    ),
                    "timestep": meta["timestep"],
                    "G_over_G0": g,
                    "transmission_at_EF": t_ef,
                    "negf_path": str(pth),
                }
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            L.error(f"抽取失败: {pth}")
            L.error(f"原因: {e}")
            n_fail += 1

    if not rows:
        raise SystemExit("没有成功抽取到任何电导数据。")

    # 排序：温度 -> 手性 -> 结构 -> timestep，便于阅读
    def _key(r):
        return (
            r["temperature_K"] if r["temperature_K"] is not None else -1,
            r["m"] if r["m"] is not None else -1,
            r["n"] if r["n"] is not None else -1,
            r["structure"],
            r["timestep"],
        )

    rows.sort(key=_key)

    fieldnames = [
        "temperature_K",
        "m",
        "n",
        "structure",
        "structure_base",
        "n_defects",
        "scattering_length_A",
        "linear_density_A-1",
        "timestep",
        "G_over_G0",
        "transmission_at_EF",
        "negf_path",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    L.ok(f"电导汇总完成  成功={n_ok}  失败={n_fail}")
    L.info(f"CSV -> {out_path}")


if __name__ == "__main__":
    main()
