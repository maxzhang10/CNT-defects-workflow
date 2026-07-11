# %%
from ase import Atom
from ase.build import nanotube
from numpy.linalg import norm
from ase.io import write
import numpy as np
from ase.build import sort
import os

from math import gcd
import math
import re
import cnt_geometry
import defects
import exporters
import lammps_io
import json
from pathlib import Path

# %%
def multi_defects_ele(tube, coords, type_list, seed=None):
    rng = np.random.default_rng(seed)

    new_tube = tube.copy()
    defect_log = []

    for k, (theta, z) in enumerate(coords, start=1):

        current_index = cnt_geometry.coord_to_index(
            new_tube,
            theta,
            z,
            only_C=True
        )

        defect_type = str(rng.choice(type_list))

        if defect_type == "MVH":
            new_tube = defects.CNT_MVH(new_tube, current_index)
        elif defect_type == "DV":
            new_tube = defects.CNT_DV(new_tube, current_index)
        elif defect_type == "5775":
            new_tube = defects.CNT_5775(new_tube, current_index)
        else:
            raise ValueError(f"未知缺陷类型: {defect_type}")

        defect_log.append({
            "no": k,
            "type": defect_type,
            "theta": theta,
            "z": z,
            "index_when_created": current_index,
        })

    return new_tube, defect_log

# %%
def generate_defect_coords(
    tube,
    N,
    T,
    l_PL,
    l_def,
    min_sep=4.26,
    edge_margin=1.42,
    seed=None,
    max_trials=20000,
):
    """
    在缺陷区内随机生成 N 个缺陷坐标。

    约束：
    1. 只在中间 l_def 个 unit cell 的 defects 区内采样；
    2. defects 区左右边界各缩小 edge_margin，默认 1.42 Å；
    3. 不再使用 ele_mask_checker；
    4. 缺陷之间二维柱面展开距离大于 min_sep；
    5. 最终坐标吸附到真实 C 原子上；
    6. 完全由 seed 控制随机性。
    """

    rng = np.random.default_rng(seed)
    cyl = cnt_geometry.tube_to_cyl(tube)

    coords = []
    used_indices = []

    # 名义 defects 区域：
    # 2PL-2PL-defects-2PL-2PL
    z_def_low = 4 * l_PL * T
    z_def_high = (4 * l_PL + l_def) * T

    # 实际允许造缺陷的区域：
    # 左右各缩小一个 C-C 键长，避免缺陷贴近边界
    z_low = z_def_low + edge_margin
    z_high = z_def_high - edge_margin

    if z_high <= z_low:
        raise ValueError(
            f"可造缺陷区间为空：z_low={z_low:.6f}, z_high={z_high:.6f}。"
            f"请增大 l_def 或减小 edge_margin。"
        )

    print(f"名义 defects 区间: z = [{z_def_low:.6f}, {z_def_high:.6f}] Å")
    print(f"实际采样区间:     z = [{z_low:.6f}, {z_high:.6f}] Å")
    print(f"边界缩进:         {edge_margin:.2f} Å")

    trial = 0

    while len(coords) < N and trial < max_trials:
        trial += 1

        theta_rand = rng.uniform(-np.pi, np.pi)
        z_rand = rng.uniform(z_low, z_high)

        index = cnt_geometry.coord_to_index(
            tube,
            theta_rand,
            z_rand,
            only_C=True
        )

        if index in used_indices:
            continue

        # 吸附到真实 C 原子位置
        theta_atom = cnt_geometry.wrap_to_pi(cyl["theta"][index - 1])
        z_atom = cyl["z"][index - 1]

        # 吸附到最近 C 原子后，仍然要求在缩小后的 z 范围内
        if not (z_low <= z_atom <= z_high):
            continue

        candidate = [theta_atom, z_atom]

        if not far_enough_from_existing(candidate, coords, min_sep):
            continue

        coords.append(candidate)
        used_indices.append(index)

    if len(coords) < N:
        raise RuntimeError(
            f"只生成了 {len(coords)} 个缺陷，目标是 {N} 个。"
            f"可以降低 N、减小 min_sep、减小 edge_margin，或者增大 l_def。"
        )

    return coords, used_indices

# %%
def far_enough_from_existing(candidate, existing_coords, min_sep):
    theta, z = candidate

    for theta_old, z_old in existing_coords:
        dist = cnt_geometry.cyl_distance_2d(
            theta, z,
            theta_old, z_old
        )

        if dist < min_sep:
            return False

    return True

# %%
def load_config(config_path="/personal/CNT_defects_package/work_flow_test/config_multi.json"):
    if config_path is None:
        config_path = os.environ.get("CNT_CONFIG", "../config.json")

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config




# %%
config = load_config()


temperature = int(config["temperature"])
m, n = map(int, config["chirality"])
r_max = float(config["r_max"])
data_root = Path(config.get("data_root", "../data")).resolve()
l_def = int(config["l_def"])
T, N_uc, l_PL, length = cnt_geometry.geo_info(m, n, r_max,l_def) 


# %%
tube_unit = cnt_geometry.build_unit_cnt(m, n, vacuum=10.0)
tube_clean = cnt_geometry.clean_cnt_by_shift_wrap_anchor(tube_unit) 
tube = tube_clean * (1, 1, length) 

# %%
cnt_geometry.set_reference_cyl(tube)   # 规定中心轴

# %% [markdown]
# 生成结构

# %%
# =========================
# 生成 N 个缺陷坐标
# =========================

N = int(config["N_defects"])
seed = 20260705

# 二维柱面展开距离屏蔽半径
# 目的：避免两个缺陷核心拓扑重叠
min_defect_sep = 4.26

# defects 区左右各缩小一个 C-C 键长
edge_margin = 1.42

defects_cood_ind, pristine_indices = generate_defect_coords(
    tube=tube,
    N=N,
    T=T,
    l_PL=l_PL,
    l_def=l_def,
    min_sep=min_defect_sep,
    edge_margin=edge_margin,
    seed=seed,
)

Dens = len(defects_cood_ind) / (l_def * T)

print(f"目标缺陷数量: {N}")
print(f"实际缺陷数量: {len(defects_cood_ind)}")
print(f"散射区长度: {l_def * T:.2f} Å")
print(f"线密度: {Dens:.4f} 缺陷/Å")
print(f"二维柱面最小间距: {min_defect_sep:.2f} Å")

print("缺陷坐标：")
for coord, idx in zip(defects_cood_ind, pristine_indices):
    print(
        f"  index={idx:5d}, "
        f"theta={coord[0]: .6f}, "
        f"z={coord[1]: .6f}"
    )






# %%
# =========================
# 生成多缺陷结构
# =========================

# type_list = ["MVH", "DV", "5775"]
type_list = config["structures"]

tube_multi_defects, defect_log = multi_defects_ele(
    tube,
    defects_cood_ind,
    type_list,
    seed=seed,
)

print("实际生成缺陷记录：")
for item in defect_log:
    print(
        f"  #{item['no']:02d} "
        f"type={item['type']:>4s}, "
        f"index={item['index_when_created']:5d}, "
        f"theta={item['theta']: .6f}, "
        f"z={item['z']: .6f}"
    )

# %%
type_name = "_".join(type_list)
folder_name = f"{type_name}_Dens_{Dens:.2f}Å-1"
structures = {
    folder_name: tube_multi_defects
}



for folder, atoms in structures.items():
    atoms_pos = atoms.copy()
    
    atoms_lmp = exporters.reposition_hydrogens(atoms, 4*l_PL*N_uc)  # 计算左右电极的原子数，调整 H 原子位置


    output_dir = data_root / f"{temperature}K" / f"{m}_{n}" / folder / "lammps"
    output_dir.mkdir(parents=True, exist_ok=True)

    poscar_path = output_dir / "POSCAR"
    lammps_path = output_dir / "data.lmp"

    exporters.write_poscar(poscar_path, atoms_pos)
    exporters.write_lammps(lammps_path, atoms_lmp)

    print(f"[WRITE] {folder}")
    print(f"  POSCAR   -> {poscar_path}")
    print(f"  data.lmp -> {lammps_path}")

# %%


from itertools import chain

from numpy import char


lammps_io.prepare_lammps_inputs(
    temperature=temperature,
    chirality=(m, n),
    structures=structures,
    l_PL=l_PL,
    N_uc=N_uc,
    data_root=data_root
)


