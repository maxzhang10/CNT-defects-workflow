# batch.py —— (8,2) CNT 批任务配置
#
# 外部 UC 从本文件同目录的 POSCAR 读取。POSCAR 必须是单个 (8,2) 原胞；
# batch_config 会校验 56 个 C 原子、轴向周期并记录文件 SHA-256。

temperatures = [100]
lammps_mode = "md"             # "md" NVT 退火 / "opt" 几何优化
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "."                # 相对路径按本文件所在目录解析
max_parallel = 16

template = {
    "defect_angle_range_deg": [0, 360],  # 缺陷锚点周向范围（度）；例如 [0, 90]
    "r_max": 6.5,
    "md_steps": 15000,
    "buffer_pl": 4,
    "release_uc": 1,
    # defect_density 接口可选 same_count / balanced_count；下方显式
    # N_defects 配置始终使用 same_count。
    "defect_count_mode": "balanced_count",
    "ele_T": 100,
    "save_self_energy": False,
    # 相对路径按每个 DPNEGF leaf 解析，各 replica 使用自己的缓存。
    # 若改成跨 replica 共享的绝对路径，外部 UC 要加入
    # {uc_sha256} 和 {ele_T}，并考虑模型、r_max 等电子参数。
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
    "md_sampling": {"n_samples": 1},
}

# (8,2): 外部 POSCAR 的 UC = 6.50725749 Å，56 C/UC。
# 沿用原先同轴向 UC 长度的 (12,3) 长度、缺陷数和 replica 计划。
CHIRAL_CONFIGS = [
    {
        "m": 8,
        "n": 2,
        # 本文件同目录的优化后 (8,2) UC；轴向周期 6.50725749 Å。
        "uc_file": "./input_files/uc/8_2/POSCAR",
        "buffer_pl": 4,
        "l_def": l_def,
        "N_defects": n_defects,
        "structures": ["MVH"],
        "replicas": replicas,
    }
    for l_def, n_defects, replicas in [
        # UC 数、缺陷数、replica 数；散射区长度
        ( 3,  1, 20),  # 1.952 nm
        ( 6,  2, 20),  # 3.904 nm
        ( 9,  3, 20),  # 5.857 nm
        (12,  4, 40),  # 7.809 nm
        (15,  5, 40),  # 9.761 nm
        (18,  6, 60),  # 11.713 nm
        (21,  7, 60),  # 13.665 nm
        (24,  8, 60),  # 15.617 nm
        (30, 10, 60),  # 19.522 nm
        (36, 12, 60),  # 23.426 nm
        (42, 14, 60),  # 27.330 nm
        (48, 16, 60),  # 31.235 nm
        (54, 18, 60),  # 35.139 nm
        (60, 20, 60),  # 39.044 nm
        (66, 22, 60),  # 42.948 nm
    ]
]
