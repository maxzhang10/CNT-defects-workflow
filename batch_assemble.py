"""独立的 POSCAR 模块拼接批配置；请先替换下方模块路径。"""

structure_mode = "assemble"
temperatures = [100]
lammps_mode = "md"
lammps_device = "gpu"          # "cpu": CH.airebo-m / "gpu": *.pth
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "./assemble_data"
max_parallel = 4

template = {
    # 含 defect_density 的条目继承此设置，也可在 configs 的单个条目覆盖。
    # balanced_count：按 replica 编号平衡取整；same_count：每根取相同整数。
    "defect_count_mode": "balanced_count",
    "r_max": 6.5,
    "md_steps": 0,  # 0: 跳过 LAMMPS，拼接 POSCAR 直接进入 DPNEGF；正数: MD
    "ele_T": 100,
    "e_fermi": None,  # eV；数值覆写费米能级，None/缺省则删除输入键并重新计算
    "save_self_energy": False,
    "self_energy_cache": {"use_saved": True, "save_path": "./self_energy/"},
}

configs = [
    {
        "chirality": [8, 2],
        "perfect_module": "./input_files/uc_POSCAR/8_2/POSCAR",
        "defect_modules": {"MVH": ("./modules/8_2/MVH_POSCAR", 1)},
        "length_angstrom": 78.0,   # 散射区目标长度（Å），也可替换为 "scatter_uc_count": 12
        "defect_density": 0.03,    # 总缺陷数 / 散射区实际长度（Å⁻¹）；允许 0
        # 多类型等比例：将上面的 defect_modules 替换为下一字典，并启用 defect_mix_mode。
        # "defect_modules": {"A": ("./modules/8_2/A_POSCAR", 1),
        #                    "B": ("./modules/8_2/B_POSCAR", 1)},
        # "defect_mix_mode": "balanced_count",
        # 此时先按密度确定每根的总数，再按 A/B 轮换分配；各类型须占相同 UC 数。
        # 固定数量：用 "defect_counts": {"MVH": 2} 替换 defect_density。
        # 多类型均分固定总数：启用上述 A/B 两行配置，用 "total_defects": 4 替换 defect_density。
        "buffer_pl": 2,  # (8,2) 旧规则：每侧 2PL 电极 + 2PL 缓冲
        "rotation_angles_deg": [0, 180],  # 只填经接口原子配准检验可用的离散角度
        "replicas": 20,
        # 固定数量和类型时可选 module_sequence，序列的缺陷数和总 UC 数须匹配。
    },
]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from batch_generate import main

    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(Path(__file__).resolve())]
    raise SystemExit(main())
