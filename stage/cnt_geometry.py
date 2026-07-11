from ase.build import nanotube
import numpy as np
from ase.build import sort
import math
from math import gcd

import logkit as L

def clean_cnt_by_shift_wrap_anchor(tube, z_shift=0.1, anchor_index=0):
    """
    对完整 CNT 做 z 方向 clean：

    1. 所有原子 z 坐标加一个小量 z_shift；
    2. wrap 回周期盒子；
    3. 取 anchor_index 号原子的 z 坐标作为参考；
    4. 所有原子 z 减去这个参考值，使该原子 z = 0；
    5. 再按 z, y, x 排序。

    参数
    ----
    tube : ase.Atoms
    z_shift : float
        z 方向平移量，默认 0.1 Å。
    anchor_index : int
        作为 z 归零参考的原子编号，Python 从 0 开始。
        anchor_index=0 对应“一号原子”。
    """

    tube = tube.copy()

    # 1. z 方向整体平移
    tube.positions[:, 2] += z_shift

    # 2. wrap 回 cell
    tube.wrap()

    # 3. 取一号原子的 z 作为参考
    z0 = tube.positions[anchor_index, 2]

    # 4. z 归零
    tube.positions[:, 2] -= z0

    # 5. 再 wrap 一次，避免出现负 z
    tube.wrap()

    # 6. 排序：z, y, x
    p = tube.positions
    order = np.lexsort((p[:, 0], p[:, 1], p[:, 2]))
    tube = tube[order]

    return tube

#构建 CNT 的函数，输入参数为 m、n、单元格数量和真空层厚度，输出为构建好的 C NT 结构。
def build_unit_cnt(m, n,vacuum=10.0):
    tube = nanotube(m, n, 1)
    tube.center(vacuum=vacuum, axis=(0, 1))

    tube = sort(tube, tags=tube.positions[:, 0])
    tube = sort(tube, tags=tube.positions[:, 1])
    tube = sort(tube, tags=tube.positions[:, 2])

    tube.positions[:, 2] -= tube.positions[:, 2].min()
    
    return tube


#固定柱坐标中心轴
def set_reference_cyl(tube):
    global cx, cy, R0
    pos = tube.positions
    cx = pos[:,0].mean()
    cy = pos[:,1].mean()
    r = np.sqrt((pos[:,0]-cx)**2 + (pos[:,1]-cy)**2)
    R0 = r.mean()

#用公式计算 P 管原子数
def calculate_atom_count(m, n, unit):
    d_R = gcd(2*m + n, 2*n + m)
    N_uc = 4 * (m**2 + n**2 + m*n) // d_R
    return N_uc * unit


#unit cell的长度计算函数
def calculate_unit_cell_length(m, n):
    a = np.sqrt(3) * 1.42  # Graphene lattice constant in Å，根号3倍的碳-碳键长
    d_R = gcd(2*m + n, 2*n + m)
    T = np.sqrt(3)*a * np.sqrt((m**2 + n**2 + m*n) ) / d_R
    return T

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def tube_to_cyl(tube):
    pos = tube.positions
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    theta = np.arctan2(y - cy, x - cx)
    
    return {"theta": theta, "z": z,  }

#给定 tube 和目标柱坐标 (theta_t, z_t)，返回最接近该位置的原子索引（1-based）
def coord_to_index(tube, theta_t, z_t,only_C=True):
    cyl = tube_to_cyl(tube)
    # 1. 预计算柱坐标
    theta_all = cyl["theta"]
    z_all = cyl["z"]
    
     # 3. 计算与目标点的差值
    dtheta = wrap_to_pi(theta_all - theta_t)
    dz = z_all - z_t    
    ds = R0 * dtheta
    # 5. 定义距离度量并选最小
    score = ds**2 + dz**2

    if only_C: #only_C=True,打开开关，可以排除H原子的干扰
        symbols = np.array(tube.get_chemical_symbols())
        score[symbols != "C"] = np.inf

    idx0 = int(np.argmin(score))
    return idx0 + 1


def cyl_distance_2d(theta1, z1, theta2, z2):
    """
    两个缺陷在 CNT 表面展开后的二维柱面距离。

    用途：
    判断两个缺陷在表面拓扑上是否太近、是否可能重叠。

    距离定义：
        d = sqrt((R0 * dtheta)^2 + dz^2)

    其中 dtheta 经过 wrap_to_pi 处理，自动考虑 theta = -pi / pi 的接缝。
    """
    dtheta = wrap_to_pi(theta1 - theta2)
    ds = R0 * dtheta
    dz = z1 - z2

    return np.sqrt(ds**2 + dz**2)

def geo_info(m, n, r_max, l_def):
    T = calculate_unit_cell_length(m, n)
    N_uc = calculate_atom_count(m, n, 1)
    l_PL = math.floor(r_max / T) + 1

    # 2PL-2PL-defects-2PL-2PL
    length = l_PL * 8 + l_def

    L.debug("结构为 2PL-2PL-defects-2PL-2PL")
    L.debug(f"defects区 为 {l_def} 个uc")
    L.debug(f"unit cell长度为 T = {T}")
    L.debug(f"unit cell 原子数N_uc = {N_uc}")
    L.debug(f"PL由 l_PL = {l_PL} 个unit cell 组成")
    L.debug(f"总体由 length = {length} 个unit cell 组成")
    L.debug(f"总体原子数 = {N_uc * length}")

    return T, N_uc, l_PL, length