# batch.py —— 批任务配置（Python 格式）
#
# 与 batch.json 等价，但可以用循环 / 变量 / 条件表达式生成 configs。
# 命名空间为白名单（无 import / open），只写数据生成逻辑。
# 校验规则与 batch.json 完全一致，见 stage/batch_config.py docstring。

temperatures = [500]
lammps_mode = "opt"            # "md" NVT 退火 / "opt" 几何优化
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "."                # 相对路径按本文件所在目录解析

template = {
    "r_max": 6.5,
    "md_steps": 50000,
    "save_self_energy": False,
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
    },
    "md_sampling": {"n_samples": 1},
}

# 每个物理配置显式声明自己的 replicas（必填）。
# 示例：三个散射区长度，短管 20 个 replica，长管 40 个。
configs = [
    {
        "chirality": [5, 5],
        "l_def": l,
        "N_defects": 2,
        "structures": ["5775"],
        "conductance_mode": "band_edge_bias",
        "replicas": 20 if l <= 8 else 40,
    }
    for l in (5, 8, 12)
]
