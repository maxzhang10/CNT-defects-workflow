# POSCAR / LAMMPS data / 元数据输出
import numpy as np
import os
from ase.io import write
#将 tube 中的 H 原子移动到中心区、右电极之前，保持 C 原子分布不变
def reposition_hydrogens(tube, N):
    """
    将由于按元素排序而落在结构末尾的 H 原子，
    移动到中心散射区末尾、右电极之前。
    移动距离为N
    重新排序后：
    左电极 C | 中心区 C 、 H | 右电极 C
    """
    symbols = np.array(tube.get_chemical_symbols())

    c_indices = np.where(symbols == "C")[0]
    h_indices = np.where(symbols == "H")[0]

    if len(h_indices) == 0:
        return tube.copy()

    if len(c_indices) < 2 * N:
        raise ValueError("C 原子数量不足，无法划分左右电极。")

    left_c = c_indices[:N]
    middle_c = c_indices[N:-N]
    right_c = c_indices[-N:]

    new_order = np.concatenate([
        left_c,
        middle_c,
        h_indices,
        right_c
    ])

    return tube[new_order]

# 将原子坐标写入 POSCAR 文件
def write_poscar(path, atoms):
    write(path, atoms, format="vasp", vasp5=True)

def write_lammps(path, atoms):
    write(path, atoms, format="lammps-data", units="metal", atom_style="atomic")

