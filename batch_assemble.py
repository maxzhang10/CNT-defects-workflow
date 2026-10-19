"""独立的 POSCAR 模块拼接批配置；请先替换下方模块路径。"""

structure_mode = "assemble"
temperatures = [100]
lammps_mode = "md"
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "./assemble_data"
max_parallel = 4

template = {
    "r_max": 6.5,
    "md_steps": 0,  # 0: 跳过 LAMMPS，拼接 POSCAR 直接进入 DPNEGF；正数: MD
    "ele_T": 100,
    "save_self_energy": False,
    "self_energy_cache": {"use_saved": True, "save_path": "./self_energy/"},
}

configs = [
    {
        "chirality": [8, 2],
        "perfect_module": "./input_files/uc_POSCAR/8_2/POSCAR",
        "defect_modules": {"MVH": ("./modules/8_2/MVH_POSCAR", 1)},
        "defect_counts": {"MVH": 2},
        # 两种或更多模块等比例混合时，用以下三项替换上面 defect_modules / defect_counts：
        # "defect_modules": {"MVH1": ("./modules/8_2/MVH1_POSCAR", 1),
        #                    "MVH2": ("./modules/8_2/MVH2_POSCAR", 1),
        #                    "MVH3": ("./modules/8_2/MVH3_POSCAR", 1)},
        # "defect_mix_mode": "balanced_count",
        # "total_defects": 4,  # replica 依次为 (2,1,1)、(1,2,1)、(1,1,2)
        "scatter_uc_count": 6,
        "buffer_pl": 2,  # (8,2) 旧规则：每侧 2PL 电极 + 2PL 缓冲
        "rotation_angles_deg": [0, 180],  # 只填经接口原子配准检验可用的离散角度
        "replicas": 20,
        # "module_sequence": ["perfect", "MVH", "perfect", "MVH", ...],
    },
]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from batch_generate import main

    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(Path(__file__).resolve())]
    raise SystemExit(main())
