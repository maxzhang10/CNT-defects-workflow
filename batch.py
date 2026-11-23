# scattering 批配置。可将此文件和 manage.sh 一起复制到数据目录并独立修改。
MAX_PARALLEL = 50

CHIRAL_CONFIGS = [
    {"m": 7,  "n": 0, "l_def": 2, "N_defects": 0, "structures": ["P"]},
    {"m": 9,  "n": 0, "l_def": 2, "N_defects": 0, "structures": ["P"]},
    {"m": 11, "n": 0, "l_def": 2, "N_defects": 0, "structures": ["P"]},
    {"m": 8,  "n": 2, "l_def": 2, "N_defects": 0, "structures": ["P"]},
    {"m": 12, "n": 3, "l_def": 1, "N_defects": 0, "structures": ["P"]},
    {"m": 6,  "n": 3, "l_def": 1, "N_defects": 0, "structures": ["P"]},
    {"m": 8,  "n": 4, "l_def": 1, "N_defects": 0, "structures": ["P"]},
    {"m": 7,  "n": 4, "l_def": 1, "N_defects": 0, "structures": ["P"]},
    {"m": 4,  "n": 4, "l_def": 4, "N_defects": 0, "structures": ["P"]},
    {"m": 6,  "n": 6, "l_def": 3, "N_defects": 0, "structures": ["P"]},
    {"m": 8,  "n": 8, "l_def": 2, "N_defects": 0, "structures": ["P"]},
]


TEMPERATURES = [500]


# 每个物理配置独立运行的 LAMMPS 轨迹数。
# 可由命令行 --lammps-repeats 覆盖。
DEFAULT_LAMMPS_REPEATS = 2

# 两类随机种子分开管理：
# seed 控制缺陷结构；同一物理配置的多个 replica 共用同一个 seed。
# lammps_seed 控制热运动轨迹；每个 replica 使用不同的 seed。
BASE_STRUCTURE_SEED = 20260705
BASE_LAMMPS_SEED = 23456789


TEMPLATE = {
    "temperature": 300,
    "chirality": [5, 5],
    "r_max": 6.50,
    "data_root": ".",
    "structures": ["5775"],
    "N_defects": 3,
    "l_def": 5,
    "md_steps": 50000,
    # 每条独立 LAMMPS 轨迹只取最后一帧做 DPNEGF。
    "md_sampling": {
        "n_samples": 1,
    },
    # DPNEGF 透射谱能量网格步长（eV）。
    "espacing": 0.1,
    # DPNEGF 透射谱的计算范围 [emin, emax]（eV）。
    "negf_energy_window": [-0.5, 0.5],
    # False 保持既有行为：DPNEGF 成功后清理 output/self_energy。
    "save_self_energy": False,
    # 覆盖 DPNEGF input.json 的 self_energy_options.cache 配置。
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
}
