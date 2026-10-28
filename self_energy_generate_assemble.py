"""Generate reusable lead self-energies with the assembly structure path.

Run this file directly, or pass it to batch_generate.py with --config.
The perfect_module must be the same POSCAR used by the target assembly jobs.
"""

structure_mode = "assemble"
temperatures = [500]
lammps_mode = "md"
base_structure_seed = 20260705
base_lammps_seed = 23456789
data_root = "/public5/home/t6s008517/dpnegf/uc_opt_self_energy_culculate/5_5/assemble"
max_parallel = 1

template = {
    "r_max": 6.5,
    "md_steps": 0,
    "release_uc": 0,
    "ele_T": 500,
    "save_self_energy": True,
    "self_energy_cache": {
        "use_saved": False,
        "save_path": "./self_energy/",
    },
    "md_sampling": {"n_samples": 1},
}

configs = [
    {
        "chirality": [5, 5],
        "perfect_module": "./input_files/uc_POSCAR/5_5/POSCAR",
        "defect_modules": {},
        "defect_counts": {},
        "scatter_uc_count": 3,
        "buffer_pl": 2,
        "rotation_angles_deg": [0],
        "replicas": 1,
    },
]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from batch_generate import main

    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(Path(__file__).resolve())]
    if "--skip-conductance" not in sys.argv:
        sys.argv.append("--skip-conductance")
    raise SystemExit(main())
