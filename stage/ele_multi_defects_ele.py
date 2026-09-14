#!/usr/bin/env python3

import json
import os
from pathlib import Path

import numpy as np

import cnt_geometry
import defects
import exporters
import lammps_io
import logkit as L


def multi_defects_ele(tube, coords, type_list, seed=None):
    rng = np.random.default_rng(seed)

    new_tube = tube.copy()
    defect_log = []

    for k, (theta, z) in enumerate(coords, start=1):
        current_index = cnt_geometry.coord_to_index(
            new_tube,
            theta,
            z,
            only_C=True,
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
    total_length=None,
):
    """
    在缺陷区内随机生成 N 个缺陷坐标。

    约束：
    1. 只在中间 l_def 个 unit cell 的 defects 区内采样；
    2. defects 区左右边界各缩小 edge_margin，默认 1.42 Å；
    3. 不使用 ele_mask_checker；
    4. 缺陷之间二维柱面展开距离大于 min_sep；
    5. 最终坐标吸附到真实 C 原子上；
    6. 完全由 seed 控制随机性。

    参数
    ----
    total_length : int, optional
        CNT 总长度（晶胞数）。如果 total_length == l_def，则为纯缺陷区
        （散射区）模式：整管只有 l_def 个 uc，缺陷区从 z=0 开始，不拼接
        电极 PL 缓冲区；否则为 2PL-2PL-defects-2PL-2PL 电极模式。
    """
    rng = np.random.default_rng(seed)
    cyl = cnt_geometry.tube_to_cyl(tube)

    coords = []
    used_indices = []

    # 判断是否为纯缺陷区（散射区）模式
    if total_length is not None and total_length == l_def:
        # 纯缺陷区模式：整个管子都是缺陷区，无电极区
        z_def_low = 0
        z_def_high = l_def * T
        L.info("使用纯缺陷区模式（无电极区）")
    else:
        # 正常模式：2PL-2PL-defects-2PL-2PL
        z_def_low = 4 * l_PL * T
        z_def_high = (4 * l_PL + l_def) * T

    # 实际允许造缺陷的区域：左右各缩小一个 C-C 键长
    z_low = z_def_low + edge_margin
    z_high = z_def_high - edge_margin

    if z_high <= z_low:
        raise ValueError(
            f"可造缺陷区间为空：z_low={z_low:.6f}, z_high={z_high:.6f}。"
            f"请增大 l_def 或减小 edge_margin。"
        )

    L.info(f"名义 defects 区间: z = [{z_def_low:.6f}, {z_def_high:.6f}] Å")
    L.info(f"实际采样区间:     z = [{z_low:.6f}, {z_high:.6f}] Å")
    L.info(f"边界缩进:         {edge_margin:.2f} Å")

    trial = 0

    while len(coords) < N and trial < max_trials:
        trial += 1

        theta_rand = rng.uniform(-np.pi, np.pi)
        z_rand = rng.uniform(z_low, z_high)

        index = cnt_geometry.coord_to_index(
            tube,
            theta_rand,
            z_rand,
            only_C=True,
        )

        if index in used_indices:
            continue

        # 吸附到真实 C 原子位置
        theta_atom = cnt_geometry.wrap_to_pi(cyl["theta"][index - 1])
        z_atom = cyl["z"][index - 1]

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


def far_enough_from_existing(candidate, existing_coords, min_sep):
    theta, z = candidate

    for theta_old, z_old in existing_coords:
        dist = cnt_geometry.cyl_distance_2d(
            theta,
            z,
            theta_old,
            z_old,
        )

        if dist < min_sep:
            return False

    return True


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("CNT_CONFIG", "../config.json")

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    config = load_config()

    required_keys = [
        "temperature",
        "chirality",
        "r_max",
        "l_def",
        "N_defects",
        "structures",
        "lammps_seed",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"config 缺少必要字段: {missing_keys}")

    temperature = int(config["temperature"])
    m, n = map(int, config["chirality"])
    r_max = float(config["r_max"])
    data_root = Path(config.get("data_root", "../data")).resolve()
    l_def = int(config["l_def"])
    md_steps = int(config.get("md_steps", 40000))

    # seed：控制缺陷位置和缺陷类型。
    # lammps_seed：只控制这一条 LAMMPS 热运动轨迹。
    seed = int(config.get("seed", 20260705))
    lammps_seed = int(config["lammps_seed"])

    if lammps_seed <= 0:
        raise ValueError(f"lammps_seed 必须为正整数，当前值: {lammps_seed}")

    T, N_uc, l_PL, _ = cnt_geometry.geo_info(
        m,
        n,
        r_max,
        l_def,
    )
    # length 由 config 决定：未指定时回退到 2PL-2PL-defects-2PL-2PL 电极模式；
    # 当 length == l_def 时为纯缺陷区（散射区）模式，整管只有 l_def 个 uc。
    length = config.get("length", l_PL * 8 + l_def)
    scattering_only = (length == l_def)
    L.info(f"length      = {length} (scattering_only={scattering_only})")

    tube_unit = cnt_geometry.build_unit_cnt(m, n, vacuum=10.0)
    tube_clean = cnt_geometry.clean_cnt_by_shift_wrap_anchor(tube_unit)
    tube = tube_clean * (1, 1, length)
    cnt_geometry.set_reference_cyl(tube)

    N = int(config["N_defects"])
    min_defect_sep = 4.26
    edge_margin = 1.42

    defects_coord_ind, pristine_indices = generate_defect_coords(
        tube=tube,
        N=N,
        T=T,
        l_PL=l_PL,
        l_def=l_def,
        min_sep=min_defect_sep,
        edge_margin=edge_margin,
        seed=seed,
        total_length=length,
    )

    density = len(defects_coord_ind) / (l_def * T)

    L.info(f"目标缺陷数量: {N}")
    L.info(f"实际缺陷数量: {len(defects_coord_ind)}")
    L.info(f"散射区长度: {l_def * T:.2f} Å")
    L.info(f"线密度: {density:.4f} 缺陷/Å")
    L.info(f"二维柱面最小间距: {min_defect_sep:.2f} Å")
    L.info(f"缺陷结构 seed: {seed}")
    L.info(f"LAMMPS seed: {lammps_seed}")

    L.info("缺陷坐标：")
    for coord, idx in zip(defects_coord_ind, pristine_indices):
        L.info(
            f"  index={idx:5d}, "
            f"theta={coord[0]: .6f}, "
            f"z={coord[1]: .6f}"
        )

    type_list = list(config["structures"])

    tube_multi_defects, defect_log = multi_defects_ele(
        tube,
        defects_coord_ind,
        type_list,
        seed=seed,
    )

    L.info("实际生成缺陷记录：")
    for item in defect_log:
        L.info(
            f"  #{item['no']:02d} "
            f"type={item['type']:>4s}, "
            f"index={item['index_when_created']:5d}, "
            f"theta={item['theta']: .6f}, "
            f"z={item['z']: .6f}"
        )

    type_name = "_".join(type_list)
    folder_name = (
        f"{type_name}"
        f"_L{l_def:03d}"
        f"_{density:.4f}A-1"
    )

    structures = {
        folder_name: tube_multi_defects,
    }

    # 目录优先级：
    # 1. run_multi.py 通过 CNT_STRUCTURE_ROOT 强制指定当前工作根目录；
    # 2. batch_generate.py 写入配置中的 structure_root；
    # 3. 兼容旧单次工作流的 data_root/温度/手性/结构目录规则。
    env_structure_root = os.environ.get("CNT_STRUCTURE_ROOT")
    configured_structure_root = config.get("structure_root")

    if env_structure_root:
        structure_root = Path(env_structure_root).resolve()

        if configured_structure_root is not None:
            config_root = Path(configured_structure_root).resolve()
            if config_root != structure_root:
                raise ValueError(
                    "结构工作目录不一致：\n"
                    f"  CNT_STRUCTURE_ROOT = {structure_root}\n"
                    f"  config.structure_root = {config_root}"
                )
    elif configured_structure_root is not None:
        structure_root = Path(configured_structure_root).resolve()
    else:
        structure_root = (
            data_root
            / f"{temperature}K"
            / f"{m}_{n}"
            / folder_name
        )

    structure_root.mkdir(parents=True, exist_ok=True)
    L.info(f"结构工作目录: {structure_root}")

    for folder, atoms in structures.items():
        atoms_pos = atoms.copy()

        # 纯缺陷区模式下没有电极区，不需要把末尾 H 原子移回中部散射区。
        if scattering_only:
            atoms_lmp = atoms.copy()
        else:
            atoms_lmp = exporters.reposition_hydrogens(
                atoms,
                4 * l_PL * N_uc,
            )

        output_dir = structure_root / "lammps"
        output_dir.mkdir(parents=True, exist_ok=True)

        poscar_path = output_dir / "POSCAR"
        lammps_path = output_dir / "data.lmp"

        exporters.write_poscar(poscar_path, atoms_pos)
        exporters.write_lammps(lammps_path, atoms_lmp)

        L.info(f"[WRITE] {folder}")
        L.info(f"  POSCAR   -> {poscar_path}")
        L.info(f"  data.lmp -> {lammps_path}")

    # 散射区模式下没有电极原子需要固定，n_fix=0（全部自由）；
    # 电极模式下固定左右各 2PL 的电极原子。
    n_fix = 0 if scattering_only else 4 * l_PL * N_uc

    lammps_io.prepare_lammps_inputs(
        temperature=temperature,
        chirality=(m, n),
        structures=structures,
        l_PL=l_PL,
        N_uc=N_uc,
        data_root=data_root,
        structure_root=structure_root,
        md_steps=md_steps,
        lammps_seed=lammps_seed,
        n_fix=n_fix,
    )


if __name__ == "__main__":
    main()
