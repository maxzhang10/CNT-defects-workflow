"""Read the axial period and atom count of a single-CNT-UC POSCAR.

This module deliberately has no ASE dependency, so batch configuration can be
validated before a compute node starts the structure-generation stage.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def read_poscar_metadata(path):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    lines = data.decode("utf-8-sig").splitlines()
    if len(lines) < 8:
        raise ValueError(f"UC POSCAR 内容不足: {path}")
    try:
        scale_fields = [float(x) for x in lines[1].split()]
        if (len(scale_fields) != 1 or not math.isfinite(scale_fields[0])
                or scale_fields[0] <= 0):
            raise ValueError("只支持一个正的 POSCAR 缩放因子")
        scale = scale_fields[0]
        lattice = [
            [float(x) * scale for x in lines[i].split()]
            for i in (2, 3, 4)
        ]
        if any(len(vector) != 3 for vector in lattice):
            raise ValueError("晶格向量必须各有三个分量")
        if not all(math.isfinite(value) for vector in lattice for value in vector):
            raise ValueError("晶格向量必须为有限数值")
        # The transport and defect-placement code assumes a straight z axis.
        tolerance = 1e-6
        if (
            abs(lattice[0][2]) > tolerance
            or abs(lattice[1][2]) > tolerance
            or abs(lattice[2][0]) > tolerance
            or abs(lattice[2][1]) > tolerance
            or lattice[2][2] <= 0
        ):
            raise ValueError("CNT 轴必须沿正 z，轴向晶格向量不得倾斜")
        if abs(lattice[0][0] * lattice[1][1]
               - lattice[0][1] * lattice[1][0]) <= tolerance:
            raise ValueError("横向两个晶格向量不能共线")
        length = lattice[2][2]
        if lines[5].split() != ["C"]:
            raise ValueError("只接受元素行为 C 的 VASP5 POSCAR")
        counts_line = lines[6].split()
        if len(counts_line) != 1 or not counts_line[0].isdigit():
            raise ValueError("C 原子数必须为单个正整数")
        counts = [int(counts_line[0])]
        if counts[0] < 1:
            raise ValueError("C 原子数必须为正整数")
        if not math.isfinite(length):
            raise ValueError("轴向晶格常数必须为有限正数")
        mode_line = 7
        if lines[mode_line].strip().lower().startswith("s"):
            mode_line += 1
        mode = lines[mode_line].strip().lower()
        if mode.startswith("d"):
            z_factor = length
        elif mode.startswith(("c", "k")):
            z_factor = scale
        else:
            raise ValueError("坐标类型必须为 Direct 或 Cartesian")
        first_atom_line = mode_line + 1
        if len(lines) < first_atom_line + counts[0]:
            raise ValueError("原子坐标行数不足")
        z_positions = []
        for line in lines[first_atom_line:first_atom_line + counts[0]]:
            coordinate = [float(value) for value in line.split()[:3]]
            if len(coordinate) != 3 or not all(math.isfinite(x) for x in coordinate):
                raise ValueError("原子坐标必须为有限三维数值")
            z_positions.append((coordinate[2] * z_factor) % length)
        z_positions.sort()
        axial_gaps = [b - a for a, b in zip(z_positions, z_positions[1:])]
        axial_gaps.append(z_positions[0] + length - z_positions[-1])
        if max(axial_gaps) > 2.5:
            raise ValueError(
                f"轴向存在 {max(axial_gaps):.3f} Å 空白；"
                "UC 需要沿 z 周期连接，不能在轴向加入真空层"
            )
    except (IndexError, ValueError) as exc:
        raise ValueError(f"UC POSCAR 格式或晶格无效: {path}: {exc}") from exc
    return {
        "path": str(path),
        "length_angstrom": length,
        "atom_count": sum(counts),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
