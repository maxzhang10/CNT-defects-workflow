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

import logkit as L


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("CNT_CONFIG", "../config.json")

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


config = load_config()
# %% [markdown]
# 单、双缺陷构建函数，均来自于CNT_defecr.py

# %% [markdown]
# 管的几何参数

# %%
import importlib
importlib.reload(cnt_geometry)

# %%
temperature = int(config["temperature"])
m, n = map(int, config["chirality"])
r_max = float(config["r_max"])
data_root = Path(config.get("data_root", "../data")).resolve()
l_def = int(config["l_def"])
md_steps = int(config.get("md_steps", 40000))

L.info("[CONFIG]")
L.info(f"temperature = {temperature}")
L.info(f"chirality   = {(m, n)}")
L.info(f"r_max       = {r_max}")
L.info(f"data_root   = {data_root}")

T, N_uc, l_PL, length = cnt_geometry.geo_info(m, n, r_max, l_def)

# %% [markdown]
# 坐标排序

# %%

tube_unit = cnt_geometry.build_unit_cnt(m, n, vacuum=10.0)
tube_clean = cnt_geometry.clean_cnt_by_shift_wrap_anchor(tube_unit) 
tube = tube_clean * (1, 1, length) 



# %%
cnt_geometry.set_reference_cyl(tube)   # 规定中心轴

# %% [markdown]
# 缺陷位置定义

# %%


index   = cnt_geometry.calculate_atom_count(m,n,length)//2
cyl     = cnt_geometry.tube_to_cyl(tube) 
theta1  =cyl["theta"][index-1]
z1      =cyl["z"][index-1]
dtheta  = np.deg2rad(0.0)   # 例如 0 度
dz      = 3                   # 例如 3 Å
theta_t = cnt_geometry.wrap_to_pi(theta1 + dtheta)
z_t     = z1 + dz

# %% [markdown]
# 生成单、双缺陷管

# %%
#生成单缺陷结构
tube_MVH   = defects.CNT_MVH(tube, index)
tube_DV    = defects.CNT_DV(tube, index)
tube_5775  = defects.CNT_5775(tube, index)


# 先只算一次第二个缺陷的 index
idx2_MVH       = cnt_geometry.coord_to_index(tube_MVH, theta_t, z_t)
tube_MVH_MVH   = defects.CNT_MVH(tube_MVH, idx2_MVH)
tube_MVH_DV    = defects.CNT_DV(tube_MVH, idx2_MVH)
tube_MVH_5775  = defects.CNT_5775(tube_MVH, idx2_MVH)
# ===== 下面开始 DV 为第一个缺陷 =====
idx2_DV        = cnt_geometry.coord_to_index(tube_DV, theta_t, z_t)
tube_DV_DV     = defects.CNT_DV(tube_DV, idx2_DV)
tube_DV_5775   = defects.CNT_5775(tube_DV, idx2_DV)
# ===== 5775 为第一个缺陷 =====
idx2_5775      = cnt_geometry.coord_to_index(tube_5775, theta_t, z_t)
tube_5775_5775 = defects.CNT_5775(tube_5775, idx2_5775)

# %% [markdown]
# 输出管结构文件

# %%

all_structures = {
    # Perfect structure
    "P": tube,

    # Single-defect structures
    "MVH": tube_MVH,
    "DV": tube_DV,
    "5775": tube_5775,

    # Double-defect structures
    "MVH_MVH": tube_MVH_MVH,
    "MVH_DV": tube_MVH_DV,
    "MVH_5775": tube_MVH_5775,
    "DV_DV": tube_DV_DV,
    "DV_5775": tube_DV_5775,
    "5775_5775": tube_5775_5775,
}

selected_structure_names = config.get("structures", list(all_structures.keys()))

unknown = [name for name in selected_structure_names if name not in all_structures]
if unknown:
    raise ValueError(
        f"config 中存在未知结构名: {unknown}\n"
        f"可选结构名为: {list(all_structures.keys())}"
    )

structures = {
    name: all_structures[name]
    for name in selected_structure_names
}

L.info("[CONFIG] selected structures:")
for i, name in enumerate(structures, 1):
    L.item(i, len(structures), name)
# structures = {
#     # Perfect structure
#     #"P": tube,

#     # Single-defect structures
#     #"MVH": tube_MVH,
#     #"DV": tube_DV,
#     #"5775": tube_5775,

#     # Double-defect structures
#     #"MVH_MVH": tube_MVH_MVH,
#     #"MVH_DV": tube_MVH_DV,
#     #"MVH_5775": tube_MVH_5775,
#     "DV_DV": tube_DV_DV,
#     "DV_5775": tube_DV_5775,
#     "5775_5775": tube_5775_5775,
# }


for folder, atoms in structures.items():
    atoms_pos = atoms.copy()
    
    atoms_lmp = exporters.reposition_hydrogens(atoms, 4*l_PL*N_uc)  # 计算左右电极的原子数，调整 H 原子位置


    output_dir = data_root / f"{temperature}K" / f"{m}_{n}" / folder / "lammps"
    output_dir.mkdir(parents=True, exist_ok=True)

    poscar_path = output_dir / "POSCAR"
    lammps_path = output_dir / "data.lmp"

    exporters.write_poscar(poscar_path, atoms_pos)
    exporters.write_lammps(lammps_path, atoms_lmp)

    L.info(f"[WRITE] {folder}")
    L.info(f"  POSCAR   -> {poscar_path}")
    L.info(f"  data.lmp -> {lammps_path}")

# %%
from itertools import chain

from numpy import char


lammps_io.prepare_lammps_inputs(
    temperature=temperature,
    chirality=(m, n),
    structures=structures,
    l_PL=l_PL,
    N_uc=N_uc,
    data_root=data_root,
    md_steps=md_steps
)

