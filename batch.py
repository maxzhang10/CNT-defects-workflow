# batch.py —— 批任务配置（Python 格式）
#
# 与 batch.json 等价，但可以用循环 / 变量 / 条件表达式生成 configs。
# 命名空间为白名单（无 import / open），只写数据生成逻辑。
# 校验规则与 batch.json 完全一致，见 stage/batch_config.py docstring。

temperatures = [100]
lammps_mode = "md"             # "md" NVT 退火 / "opt" 几何优化
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "."                # 相对路径按本文件所在目录解析

template = {
    "r_max": 6.5,
    "md_steps": 15000,
    "save_self_energy": False,
    # 自能缓存路径，可用占位符按手性区分（电极自能只依赖手性，
    # 同手性的所有长度 / replica 可共享一份缓存）：
    #   {m} {n} {chirality}（"13_1" 形式） {data_root}
    # 相对路径 = 每个 leaf 一份独立缓存；要跨 replica 共享用
    # {data_root} 组成绝对路径，例如：
    #   "save_path": "{data_root}/self_energy/{chirality}/",
    # 也可在任一 config 内加同名 "self_energy_cache" 键单独覆盖。
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
    "md_sampling": {"n_samples": 1},
}

# 每个物理配置显式声明自己的 replicas（必填）。
# 示例：三个散射区长度，短管 20 个 replica，长管 40 个。
# F = 2*m + n = 27; l_def = scattering-region UC count
CHIRAL_CONFIGS = [
    # (13, 1): UC = 19.209404 Å, 244 C/UC
    *[
        {
            "m": 13,
            "n": 1,
            "buffer_pl": 4,
            "l_def": l_def,
            "N_defects": N_defects,
            "structures": ["MVH"],
            "replicas": replicas,
        }
        for l_def, N_defects, replicas in [
            # UC数, 缺陷数, replicas    散射区长度
            ( 3,  1, 20),  # 5.763 nm
            ( 6,  2, 60),  # 11.526 nm
            ( 9,  3, 60),  # 17.288 nm
            (12,  4, 60),  # 23.051 nm
            (15,  5, 60),  # 28.814 nm
            (18,  6, 60),  # 34.577 nm
            (21,  7, 60),  # 40.340 nm
            (24,  8, 60),  # 46.103 nm
            (30, 10, 60),  # 57.628 nm
            (36, 12, 60),  # 69.154 nm
            (42, 14, 60),  # 80.679 nm
            (48, 16, 60),  # 92.205 nm
            (54, 18, 60),  # 103.731 nm
            (60, 20, 60),  # 115.256 nm
            (66, 22, 60),  # 126.782 nm
        ]
    ],
    # (12, 3): UC = 6.507257 Å, 84 C/UC
    *[
        {
            "m": 12,
            "n": 3,
            "buffer_pl": 4,
            "l_def": l_def,
            "N_defects": N_defects,
            "structures": ["MVH"],
            "replicas": replicas,
        }
        for l_def, N_defects, replicas in [
            # UC数, 缺陷数, replicas    散射区长度
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
    ],
    # (11, 5): UC = 20.131975 Å, 268 C/UC
    *[
        {
            "m": 11,
            "n": 5,
            "buffer_pl": 4,
            "l_def": l_def,
            "N_defects": N_defects,
            "structures": ["MVH"],
            "replicas": replicas,
        }
        for l_def, N_defects, replicas in [
            # UC数, 缺陷数, replicas    散射区长度
            ( 3,  1, 40),  # 6.040 nm
            ( 6,  2, 60),  # 12.079 nm
            ( 9,  3, 60),  # 18.119 nm
            (12,  4, 60),  # 24.158 nm
            (15,  5, 60),  # 30.198 nm
            (18,  6, 60),  # 36.238 nm
            (21,  7, 60),  # 42.277 nm
            (24,  8, 60),  # 48.317 nm
            (30, 10, 60),  # 60.396 nm
            (36, 12, 60),  # 72.475 nm
            (42, 14, 60),  # 84.554 nm
            (48, 16, 60),  # 96.633 nm
            (54, 18, 60),  # 108.713 nm
            (60, 20, 60),  # 120.792 nm
            (66, 22, 60),  # 132.871 nm
        ]
    ],
    # (10, 7): UC = 21.014081 Å, 292 C/UC
    *[
        {
            "m": 10,
            "n": 7,
            "buffer_pl": 4,
            "l_def": l_def,
            "N_defects": N_defects,
            "structures": ["MVH"],
            "replicas": replicas,
        }
        for l_def, N_defects, replicas in [
            # UC数, 缺陷数, replicas    散射区长度
            ( 3,  1, 40),  # 6.304 nm
            ( 6,  2, 60),  # 12.608 nm
            ( 9,  3, 60),  # 18.913 nm
            (12,  4, 60),  # 25.217 nm
            (15,  5, 60),  # 31.521 nm
            (18,  6, 60),  # 37.825 nm
            (21,  7, 60),  # 44.130 nm
            (24,  8, 60),  # 50.434 nm
            (30, 10, 60),  # 63.042 nm
            (36, 12, 60),  # 75.651 nm
            (42, 14, 60),  # 88.259 nm
            (48, 16, 60),  # 100.868 nm
            (54, 18, 60),  # 113.476 nm
            (60, 20, 60),  # 126.084 nm
            (66, 22, 60),  # 138.693 nm
        ]
    ],
    # (9, 9): UC = 2.459512 Å, 36 C/UC
    *[
        {
            "m": 9,
            "n": 9,
            "buffer_pl": 4,
            "l_def": l_def,
            "N_defects": N_defects,
            "structures": ["MVH"],
            "replicas": replicas,
        }
        for l_def, N_defects, replicas in [
            # UC数, 缺陷数, replicas    散射区长度
            ( 3,  1, 20),  # 0.738 nm
            ( 6,  2, 20),  # 1.476 nm
            ( 9,  3, 20),  # 2.214 nm
            (12,  4, 20),  # 2.951 nm
            (15,  5, 20),  # 3.689 nm
            (18,  6, 20),  # 4.427 nm
            (21,  7, 20),  # 5.165 nm
            (24,  8, 20),  # 5.903 nm
            (30, 10, 40),  # 7.379 nm
            (36, 12, 40),  # 8.854 nm
            (42, 14, 60),  # 10.330 nm
            (48, 16, 60),  # 11.806 nm
            (54, 18, 60),  # 13.281 nm
            (60, 20, 60),  # 14.757 nm
            (66, 22, 60),  # 16.233 nm
        ]
    ],
]

# 提交顺序：优先运行 9_9、12_3、13_1；其余手性保持上面的原顺序。
_priority = {(9, 9): 0, (12, 3): 1, (13, 1): 2}
CHIRAL_CONFIGS.sort(
    key=lambda config: _priority.get((config["m"], config["n"]), 3)
)
