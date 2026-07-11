# %%
import os
import re
import numpy as np
import json
from ase import Atoms
from ase.io import write
from ase.units import Bohr
import cnt_geometry
import argparse
from pathlib import Path

import logkit as L

def load_config(config_path=None):
    """
    优先级：
    1. 命令行传入 --config
    2. 环境变量 CNT_CONFIG
    3. 默认 work_flow_test/config.json
    """
    if config_path is None:
        env_config = os.environ.get("CNT_CONFIG")

        if env_config is not None:
            config_path = env_config
        else:
            # fdf2xyz.py 位于 work_flow_test/stage/
            # parents[1] 是 work_flow_test/
            config_path = Path(__file__).resolve().parents[1] / "config.json"

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_chirality_from_config(config):
    if "chirality" not in config:
        raise KeyError("config.json 中缺少 chirality 字段，例如: \"chirality\": [5, 5]")

    chirality = config["chirality"]

    if not isinstance(chirality, (list, tuple)) or len(chirality) != 2:
        raise ValueError(f"config['chirality'] 格式错误，应为 [m, n]，实际为: {chirality}")

    m, n = map(int, chirality)

    return m, n

# %%

def build_output_dir(fdf_path, root_dir, outroot, chirality):
    """
    根据输入 STRUCT.fdf 路径构造输出目录。

    支持两种模式：

    1. root 是 data 总目录：
       data/300K/5_5/5775_5775/dpnegf/40000/STRUCT.fdf
       -> outroot/300K/5_5/5775_5775/dpnegf/40000/

    2. root 已经是某个结构的 dpnegf 目录：
       5775_5775/dpnegf/40000/STRUCT.fdf
       -> outroot/40000/

    关键：
    手性 m,n 从 config.json 读，不再从 STRUCT.fdf 第一行读。
    同时只删除路径中第一个手性目录 5_5，不删除结构名 5775_5775。
    """

    fdf_path = Path(fdf_path).resolve()
    root_dir = Path(root_dir).resolve()
    outroot = Path(outroot).resolve()

    m, n = chirality
    chirality_dir = f"{m}_{n}"

    rel_dir = fdf_path.parent.relative_to(root_dir)
    rel_parts = list(rel_dir.parts)

    # 找温度目录，例如 300K
    temp_idx = None
    for i, part in enumerate(rel_parts):
        if re.fullmatch(r"\d+K", part):
            temp_idx = i
            break

    # 如果 root 本身就是某个结构的 dpnegf 目录，例如：
    # root = .../5775_5775/dpnegf
    # rel_parts = ["40000"]
    # 这种情况下不重建 300K/5_5/结构名，直接保持相对路径
    if temp_idx is None:
        return outroot / rel_dir

    temperature_dir = rel_parts[temp_idx]

    # 温度目录后面的部分：
    # ["5_5", "5775_5775", "dpnegf", "40000"]
    rest_parts = rel_parts[temp_idx + 1:]

    # 只去掉第一个手性目录 5_5
    # 千万不要用正则删除所有 \d+_\d+，否则 5775_5775 会被误删
    if rest_parts and rest_parts[0] == chirality_dir:
        rest_parts = rest_parts[1:]

    output_dir = outroot / temperature_dir / chirality_dir

    if rest_parts:
        output_dir = output_dir / Path(*rest_parts)

    return output_dir

def swap_left_two_pl_order(tube, n_atoms_per_pl):
    """
    只交换左端前两个 PL 的原子编号顺序，不改变原子坐标。
    原始顺序：
        [PL1][PL2][rest]
    新顺序：
        [PL2][PL1][rest]
    """
    n_total = len(tube)

    if n_total < 2 * n_atoms_per_pl:
        raise ValueError("体系原子数不足，无法交换前两个 PL。")
    
    order = np.concatenate([
        np.arange(n_atoms_per_pl, 2 * n_atoms_per_pl),  # 原 PL2 放到前面
        np.arange(0, n_atoms_per_pl),                   # 原 PL1 放到后面
        np.arange(2 * n_atoms_per_pl, n_total)          # 剩余原子顺序不变
    ])

    return tube[order]

# %%
def clean_line(line):
    """
    去掉 FDF 行中的注释和首尾空格。
    """
    return line.split("#", 1)[0].strip()

# %%
def get_chirality_from_fdf(fdf_path):
    """
    从 STRUCT.fdf 第一行提取 CNT 手性指数 m, n。

    例如：
    # from /data/run01/scvj270/lmp4data/TB/tutorial_DFT+NEGF/9_0/300K/...
    """
    with open(fdf_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    match = re.search(r"/(\d+)_(\d+)(?:/|$)", first_line)

    if match is None:
        raise ValueError(f"无法从第一行提取 m, n：\n{first_line}")

    return int(match.group(1)), int(match.group(2))

# %%
def read_fdf_value(lines, key):
    """
    读取单行参数，例如：
        LatticeConstant 1.0 Ang
        AtomicCoordinatesFormat Ang
    """
    key = key.lower()

    for line in lines:
        line = clean_line(line)

        if not line:
            continue

        parts = line.split()

        if parts[0].lower() == key:
            return parts[1:]

    raise ValueError(f"未找到参数：{key}")


# %%
def read_fdf_block(lines, block_name):
    """
    读取 block，例如：
        %block LatticeVectors
        ...
        %endblock LatticeVectors
    """
    block_name = block_name.lower()
    inside = False
    data = []

    for line in lines:
        line = clean_line(line)

        if not line:
            continue

        parts = line.split()
        lower = [x.lower() for x in parts]

        if len(lower) >= 2 and lower[0] == "%block" and lower[1] == block_name:
            inside = True
            continue

        if inside and lower[0] == "%endblock":
            return data

        if inside:
            data.append(line)

    raise ValueError(f"未找到 block：{block_name}")

# %%
def parse_struct_fdf(fdf_path):
    """
    直接从 STRUCT.fdf 读取：
        - 晶格
        - 元素类型
        - 原子坐标

    返回 ASE Atoms 对象。
    """

    with open(fdf_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # =========================
    # 1. 读取晶格常数
    # =========================
    lattice_info = read_fdf_value(lines, "LatticeConstant")

    lattice_constant = float(lattice_info[0])
    lattice_unit = lattice_info[1].lower() if len(lattice_info) > 1 else "bohr"

    if lattice_unit.startswith("ang"):
        lattice_scale = lattice_constant
    elif lattice_unit.startswith("bohr"):
        lattice_scale = lattice_constant * Bohr
    else:
        raise ValueError(f"不支持的 LatticeConstant 单位：{lattice_unit}")

    # =========================
    # 2. 读取晶格向量
    # =========================
    lattice_block = read_fdf_block(lines, "LatticeVectors")

    if len(lattice_block) != 3:
        raise ValueError(
            f"LatticeVectors 应为 3 行，实际读取到 {len(lattice_block)} 行。"
        )

    cell = np.array(
        [[float(x) for x in row.split()[:3]] for row in lattice_block],
        dtype=float
    )

    cell *= lattice_scale

    # =========================
    # 3. 读取元素编号对应关系
    # =========================
    species_block = read_fdf_block(lines, "ChemicalSpeciesLabel")

    species_map = {}

    for row in species_block:
        parts = row.split()

        if len(parts) < 3:
            raise ValueError(f"ChemicalSpeciesLabel 行格式异常：{row}")

        species_id = int(parts[0])
        symbol = parts[2]

        species_map[species_id] = symbol

    # =========================
    # 4. 读取坐标格式
    # =========================
    coordinate_info = read_fdf_value(lines, "AtomicCoordinatesFormat")
    coordinate_format = coordinate_info[0].lower()

    # =========================
    # 5. 读取原子坐标
    # =========================
    coordinate_block = read_fdf_block(lines, "AtomicCoordinatesAndAtomicSpecies")

    positions = []
    symbols = []

    for row in coordinate_block:
        parts = row.split()

        if len(parts) < 4:
            raise ValueError(f"坐标行格式异常：{row}")

        x, y, z = map(float, parts[:3])
        species_id = int(parts[3])

        if species_id not in species_map:
            raise ValueError(f"未定义的元素编号：{species_id}")

        positions.append([x, y, z])
        symbols.append(species_map[species_id])

    positions = np.array(positions, dtype=float)

    # =========================
    # 6. 将坐标统一转为 Å
    # =========================
    if coordinate_format in ["ang", "angstrom", "notscaledcartesianang"]:
        pass

    elif coordinate_format in ["bohr", "notscaledcartesianbohr"]:
        positions *= Bohr

    elif coordinate_format == "scaledcartesian":
        positions *= lattice_scale

    elif coordinate_format in ["fractional", "scaledbylatticevectors"]:
        positions = positions @ cell

    else:
        raise ValueError(
            f"暂不支持 AtomicCoordinatesFormat：{coordinate_format}"
        )

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=False
    )

    return atoms


# %%
def FDF_to_xyz(
    fdf_path,
    output_dir=None,
    r_max=6.5,
    swap_left_pl=True,
    chirality=None,
    l_def=5
):
    """
    读取 leaf 文件夹中的 STRUCT.fdf，
    可选地交换左端前两个 PL 的原子编号顺序，
    并在同一文件夹下生成 m_n.xyz。

    手性 m,n 从 config.json 传入。
    """

    if chirality is None:
        raise ValueError("FDF_to_xyz 需要传入 chirality=(m, n)，现在不再从 STRUCT.fdf 读取手性。")

    m, n = chirality

    if output_dir is None:
        output_dir = os.path.dirname(fdf_path)

    os.makedirs(output_dir, exist_ok=True)



    xyz_path = os.path.join(output_dir, f"{m}_{n}.xyz")

    # 2. 直接解析 STRUCT.fdf
    atoms = parse_struct_fdf(fdf_path)

    # 3. 根据 m, n 计算一个 PL 中的原子数，并交换左端两个 PL
    if swap_left_pl:
        T, N_uc, l_PL, length = cnt_geometry.geo_info(m, n, r_max,l_def)

        n_atoms_per_pl = l_PL * N_uc

        atoms = swap_left_two_pl_order(atoms, n_atoms_per_pl)

        L.debug(
            f"Swap left PLs: {fdf_path} | "
            f"m={m}, n={n}, N_uc={N_uc}, l_PL={l_PL}, "
            f"atoms_per_PL={n_atoms_per_pl}"
        )

    # 4. 输出普通 xyz
    write(xyz_path, atoms, format="xyz")

    # 5. 构造 NanoTCAD-ViDES 需要的第二行
    lattice = " ".join(
        f"{x:.8f}" for x in atoms.cell.array.flatten()
    )

    xyz_header = (
        f'Lattice="{lattice}" '
        'Properties=species:S:1:pos:R:3 '
        'Generation=T from=T NanoTCAD-ViDES=T pbc="F F F"'
    )

    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines[1] = xyz_header + "\n"

    with open(xyz_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    L.ok(f"转换完成: {fdf_path} -> {xyz_path} | natoms={len(atoms)}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert STRUCT.fdf files to xyz files for DPNEGF."
    )

    parser.add_argument(
        "--root",
        default="siesta_structs_batch",
        help="Input root directory containing STRUCT.fdf files."
    )

    parser.add_argument(
        "--outroot",
        required=True,
        help="Output root directory for xyz files. 当前版本主要保留参数兼容，实际原地输出到 STRUCT.fdf 所在目录。"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="config.json path. If not given, use CNT_CONFIG or work_flow_test/config.json."
    )

    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="r_max used for CNT geometry information. If not given, read from config.json."
    )

    parser.add_argument(
        "--no-swap-left-pl",
        action="store_true",
        help="Do not swap the first two left PLs."
    )

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    outroot = Path(args.outroot).resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Input root does not exist: {root_dir}")

    # =========================
    # 从 config 读取 chirality 和 r_max
    # =========================
    config = load_config(args.config)
    chirality = get_chirality_from_config(config)
    l_def = int(config.get("l_def", 5))
    if args.r_max is None:
        r_max = float(config.get("r_max", 6.5))
    else:
        r_max = float(args.r_max)

    L.info("[CONFIG]")
    L.info(f"chirality = {chirality}")
    L.info(f"r_max     = {r_max}")

    n_success = 0
    n_failed = 0

    for current_dir, subdirs, files in os.walk(root_dir):
        if "STRUCT.fdf" not in files:
            continue

        fdf_path = Path(current_dir) / "STRUCT.fdf"

        try:
            # 关键：原地输出 xyz，不再重建目录
            output_dir = Path(current_dir)

            FDF_to_xyz(
                fdf_path=str(fdf_path),
                output_dir=str(output_dir),
                r_max=r_max,
                swap_left_pl=(not args.no_swap_left_pl),
                chirality=chirality,
                l_def=l_def
            )

            n_success += 1

        except Exception as e:
            n_failed += 1
            L.error(f"转换失败: {fdf_path}")
            L.error(f"原因: {e}")

    L.ok(f"fdf2xyz 完成  成功={n_success}  失败={n_failed}")
    L.info(f"输入根目录 = {root_dir}")
    L.info(f"输出根目录 = {outroot}")

    if n_failed > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()


