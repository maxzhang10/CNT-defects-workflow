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
from defect_angles import normalize_angle_range
from geometry_mode import geometry_mode, uses_aligned_5775


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


def aligned_5775_defects(tube, coords, indices, seed):
    if not indices:
        return tube.copy(), []

    # Plan all rotations on the pristine tube. SW rotations preserve atom indices.
    cyl = cnt_geometry.tube_to_cyl(tube)
    neighbors = np.array([[j for _, j in defects.find_nearest(tube, i)] for i in indices])
    centers = np.asarray(indices)[:, None] - 1
    arc = cnt_geometry.R0 * cnt_geometry.wrap_to_pi(cyl["theta"][neighbors] - cyl["theta"][centers])
    dz = cyl["z"][neighbors] - cyl["z"][centers]
    angles = np.arctan2(arc, dz) % np.pi  # undirected bond in the unrolled tube

    # Distance ranks can exchange places for nearly equal C–C lengths.
    # Keep ranks that select the same bond family as at the first anchor.
    eligible = []
    for rank, reference in enumerate(angles[0]):
        alignment = np.abs(np.cos(angles - reference))
        if np.all(np.argmax(alignment, axis=1) == rank):
            eligible.append(rank)
    if not eligible:
        raise ValueError("这些等分点的三个 C 近邻序号都无法在全部 5775 缺陷上保持同一键取向")
    rank = int(np.random.default_rng(seed).choice(eligible))
    result = tube
    defect_log = []
    for row, (center, (theta, z)) in enumerate(zip(indices, coords)):
        partner = int(neighbors[row, rank]) + 1
        result = defects.add_57(result, center, partner)
        defect_log.append(dict(no=row + 1, type="5775", theta=theta, z=z,
                               index_when_created=center, neighbor_rank=rank,
                               neighbor_index=partner, bond_angle_deg=float(np.rad2deg(angles[row, rank]))))
    return result, defect_log


def generate_defect_coords(
    tube,
    N,
    T,
    l_PL,
    l_def,
    m,
    n,
    min_sep=4.26,
    edge_margin=1.42,
    seed=None,
    max_trials=20000,
    buffer_pl=None,
    defect_angle_range_deg=(0, 360),
    scattering_only=False,
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
    """
    angle_start, angle_end = normalize_angle_range(defect_angle_range_deg)
    angle_width = angle_end - angle_start
    if N < 0:
        raise ValueError(f"N 不能小于 0，当前值: {N}")
    if N == 0:
        L.info("本 replica 分配为 0 个缺陷，跳过缺陷坐标采样")
        return [], []

    rng = np.random.default_rng(seed)
    cyl = cnt_geometry.tube_to_cyl(tube)

    coords = []
    used_indices = []

    # 名义 defects 区域：
    # 金属管 2PL-2PL-defects-2PL-2PL（左侧 4PL 后开始）；
    # 半导体管 2PL-4PL-defects-4PL-2PL（左侧 6PL 后开始）。
    n_fix_pl = 0 if scattering_only else cnt_geometry.fixed_pl_per_side(m, n, buffer_pl)
    z_def_low = n_fix_pl * l_PL * T
    z_def_high = (n_fix_pl * l_PL + l_def) * T

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

        # Keep the legacy random sequence for a full circle.
        theta_rand = (rng.uniform(-np.pi, np.pi) if angle_width == 360
                      else np.deg2rad(rng.uniform(angle_start, angle_end)))
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
        if angle_width < 360 and (np.rad2deg(theta_atom) - angle_start) % 360 > angle_width:
            continue

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
            f"角度范围为 [{angle_start}, {angle_end}] 度。"
            f"可以扩大角度范围、降低 N、减小 min_sep、减小 edge_margin，或者增大 l_def。"
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


def equidistant_defect_coords(tube, count, length, angle_range, edge_margin, min_sep, seed):
    """Snap the interior division points to carbon sites near a common azimuth."""
    if count == 0:
        return [], []
    start, end = normalize_angle_range(angle_range)
    cyl = cnt_geometry.tube_to_cyl(tube)
    theta, z = cyl["theta"], cyl["z"]
    allowed = (tube.numbers == 6) & (z >= edge_margin) & (z <= length - edge_margin)
    if end - start < 360:
        allowed &= (np.rad2deg(theta) - start) % 360 <= end - start
    if count > np.count_nonzero(allowed):
        raise ValueError(f"角度和边距范围内可用的碳位点不足 {count} 个")

    step = length / (count + 1)
    targets = step * np.arange(1, count + 1)
    groups = []
    for target in targets:
        dz = np.abs(z - target)
        # Give each division point its own axial interval. Prefer sites within
        # one C–C bond of the ideal z, or the nearest layer in a sparse sector.
        sites = np.flatnonzero(allowed & (dz <= step / 2))
        if not len(sites):
            raise ValueError(f"等分点 z={target:.4f} Å 附近没有满足角度和边距的碳位点")
        cutoff = max(1.42, dz[sites].min()) + 1e-9
        groups.append(sites[dz[sites] <= cutoff])

    best_indices = None
    best_score = None
    for first in np.random.default_rng(seed).permutation(groups[0]):
        chosen = [first]
        for sites, target in zip(groups[1:], targets[1:]):
            for prev in reversed(chosen):
                # Axial minimum image includes the two ends of the supercell.
                dz = np.abs(z[sites] - z[prev])
                dz = np.minimum(dz, length - dz)
                arc = cnt_geometry.R0 * cnt_geometry.wrap_to_pi(theta[sites] - theta[prev])
                sites = sites[arc**2 + dz**2 >= min_sep**2 - 1e-10]
            if not len(sites):
                break
            angles = np.abs(cnt_geometry.wrap_to_pi(theta[sites] - theta[first]))
            order = np.lexsort((np.abs(z[sites] - target), angles.round(10)))
            chosen.append(sites[order[0]])
        if len(chosen) == count:
            angles = np.sort(theta[chosen] % (2 * np.pi))
            gaps = np.diff(angles, append=angles[0] + 2 * np.pi)
            span = 2 * np.pi - gaps.max()
            score = (round(span, 10), round(float(np.sum((z[chosen] - targets)**2)), 10))
            if best_score is None or score < best_score:
                best_indices, best_score = chosen, score
    if best_indices is None:
        raise ValueError(f"无法在 {count} 个等分点附近以至少 {min_sep} Å 的间距放置缺陷")
    return [[float(theta[i]), float(z[i])] for i in best_indices], [int(i + 1) for i in best_indices]


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("CNT_CONFIG", "../config.json")

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config 文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main_assemble(config):
    """Generate user-supplied modules into the same LAMMPS/DPNEGF layout."""
    import assemble

    root = Path(config["structure_root"]).resolve()
    env_root = os.environ.get("CNT_STRUCTURE_ROOT")
    if env_root and Path(env_root).resolve() != root:
        raise ValueError("CNT_STRUCTURE_ROOT differs from config.structure_root")
    if root.exists():
        unexpected = [p for p in root.iterdir()
                      if p.name not in {"workflow_config.json", "workflow.log"}
                      and not (p.is_file() and p.name.startswith("cnt_") and p.suffix == ".json")]
        if unexpected:
            raise FileExistsError(f"assembly output exists; refusing to overwrite: {unexpected[0]}")
    output_dir = root / "lammps"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"assembly structure exists: {output_dir}")
    seed = int(config["seed"])
    atoms, meta = assemble.build(
        config["assembly"], seed,
        placement_seed=config.get("placement_seed"),
        rotation_seed=config.get("rotation_seed"),
    )
    m, n = map(int, config["chirality"])
    atoms = exporters.reposition_hydrogens(atoms, meta["side_atoms_per_side"])
    output_dir.mkdir(parents=True, exist_ok=False)
    exporters.write_poscar(output_dir / "POSCAR", atoms)
    exporters.write_lammps(output_dir / "data.lmp", atoms)
    with (root / "assembly.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    with (root / "defects.json").open("w", encoding="utf-8") as file:
        json.dump({"seed": seed, "type_list": list(config["assembly"]["defect_counts"]),
                   "N_defects": meta["N_defects"], "actual_density_A_inv": meta["actual_density_A_inv"],
                   "defect_counts": meta["defect_counts"], "module_sequence": meta["module_sequence"]},
                  file, indent=2, ensure_ascii=False)
    md_steps = int(config.get("md_steps", 40000))
    lammps_mode = config.get("lammps_mode", "md")
    if lammps_mode == "opt" or md_steps > 0:
        lammps_io.prepare_lammps_inputs(
            temperature=int(config["temperature"]), chirality=(m, n),
            structures={"assembled": atoms}, l_PL=1, N_uc=meta["atoms_per_pl"],
            data_root=Path(config["data_root"]), structure_root=root,
            md_steps=md_steps, lammps_seed=config.get("lammps_seed"),
            lammps_mode=lammps_mode,
            lammps_device=config.get("lammps_device", "gpu"),
            fixed_atoms_per_side=meta["fixed_atoms_per_side"],
        )
    L.info(f"assembled {meta['module_sequence']} into {root}; scatter={meta['scatter_length_angstrom']} A")


def main():
    config = load_config()
    scattering_only = geometry_mode(config) == "scattering_only"
    if scattering_only and "defect_density" in config:
        from batch_config import resolve_scattering_config
        config = resolve_scattering_config(config)
    if config.get("structure_mode") == "assemble":
        main_assemble(config)
        return

    # lammps_mode: "md"（缺省，热弛豫 MD）或 "opt"（纯几何优化）。
    # opt 模式不做热运动，lammps_seed 无意义，不再强制。
    lammps_mode = config.get("lammps_mode", "md")
    if lammps_mode not in ("md", "opt"):
        raise ValueError(
            f"lammps_mode 必须为 'md' 或 'opt'，当前值: {lammps_mode}"
        )

    required_keys = [
        "temperature",
        "chirality",
        "r_max",
        "l_def",
        "N_defects",
        "structures",
    ]
    if lammps_mode == "md":
        required_keys.append("lammps_seed")
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
    # lammps_seed：只控制这一条 LAMMPS 热运动轨迹（opt 模式忽略）。
    seed = int(config.get("seed", 20260705))
    lammps_seed = int(config.get("lammps_seed", 0)) or None

    if lammps_mode == "md":
        if lammps_seed is None or lammps_seed <= 0:
            raise ValueError(
                f"lammps_seed 必须为正整数，当前值: {config.get('lammps_seed')}"
            )

    buffer_pl = config.get("buffer_pl")
    # 固定区内侧放开不固定的 uc 数；缺省用 cnt_geometry.RELEASE_UC_PER_SIDE
    release_uc = config.get("release_uc")
    T, N_uc, l_PL, length = cnt_geometry.geo_info(
        m,
        n,
        r_max,
        l_def,
        buffer_pl,
        config.get("uc_file"),
        config.get("uc_sha256"),
    )

    if scattering_only:
        length = l_def
    tube_unit = cnt_geometry.build_unit_cnt(
        m, n, vacuum=10.0 if scattering_only else 50.0,
        uc_file=config.get("uc_file"),
        uc_sha256=config.get("uc_sha256"),
    )
    tube_clean = cnt_geometry.clean_cnt_by_shift_wrap_anchor(tube_unit)
    tube = tube_clean * (1, 1, length)
    cnt_geometry.set_reference_cyl(tube)

    N = int(config["N_defects"])
    if N < 0:
        raise ValueError(f"N_defects 不能小于 0，当前值: {N}")
    min_defect_sep = 4.26
    edge_margin = 1.42

    angle_range = config.get("defect_angle_range_deg", [0, 360])
    equidistant = scattering_only and config.get("defect_layout") == "equidistant"
    if equidistant:
        defects_coord_ind, pristine_indices = equidistant_defect_coords(
            tube, N, l_def * T, angle_range, edge_margin, min_defect_sep, seed,
        )
    else:
        defects_coord_ind, pristine_indices = generate_defect_coords(
            tube, N, T, l_PL, l_def, m, n, min_sep=min_defect_sep,
            edge_margin=edge_margin, seed=seed, buffer_pl=buffer_pl,
            defect_angle_range_deg=angle_range, scattering_only=scattering_only,
        )

    density = len(defects_coord_ind) / (l_def * T)

    L.info(f"目标缺陷数量: {N}")
    L.info(f"实际缺陷数量: {len(defects_coord_ind)}")
    L.info(f"散射区长度: {l_def * T:.2f} Å")
    L.info(f"线密度: {density:.4f} 缺陷/Å")
    L.info(f"二维柱面最小间距: {min_defect_sep:.2f} Å")
    L.info(f"缺陷结构 seed: {seed}")
    if lammps_mode == "md":
        L.info(f"LAMMPS seed: {lammps_seed}")
    else:
        L.info("LAMMPS mode: opt（纯几何优化，不使用 lammps_seed）")

    L.info("缺陷坐标：")
    if not defects_coord_ind:
        L.info("  （本 replica 分配为 0 个缺陷，生成 pristine 结构）")
    for coord, idx in zip(defects_coord_ind, pristine_indices):
        L.info(
            f"  index={idx:5d}, "
            f"theta={coord[0]: .6f}, "
            f"z={coord[1]: .6f}"
        )

    type_list = list(config["structures"])

    if uses_aligned_5775(config):
        tube_multi_defects, defect_log = aligned_5775_defects(
            tube, defects_coord_ind, pristine_indices, seed,
        )
    else:
        tube_multi_defects, defect_log = multi_defects_ele(
            tube, defects_coord_ind, type_list, seed=seed,
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
    if equidistant:
        folder_name += f"_equidistant_{config['target_defect_density_A_inv']}"

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

    # P2-13：持久化实际抽中的缺陷组成到 defects.json。
    # 旧版 defect_log 仅输出到终端，同一配置目录会混合不同实际缺陷组成，
    # 后续统计无法从持久化元数据判断每个 replica 的真实类型和位置。
    defects_meta = {
        "defect_angle_range_deg": normalize_angle_range(config.get("defect_angle_range_deg", [0, 360])),
        "seed": seed,
        "type_list": type_list,
        "defect_log": defect_log,
        "folder_name": folder_name,
        "N_defects": N,
        "actual_density_A_inv": density,
        "defect_count_mode": config.get("defect_count_mode", "same_count"),
        "target_defect_density_A_inv": config.get(
            "target_defect_density_A_inv"
        ),
        "uc_sha256": config.get("uc_sha256"),
    }
    defects_json_path = structure_root / "defects.json"
    if scattering_only:
        defects_meta["geometry_mode"] = "scattering_only"
    if equidistant:
        defects_meta.update(defect_layout="equidistant", l_def=l_def,
                            length_angstrom=l_def * T,
                            target_z_angstrom=[i * l_def * T / (N + 1) for i in range(1, N + 1)])
    with open(defects_json_path, "w", encoding="utf-8") as f:
        json.dump(defects_meta, f, indent=2, ensure_ascii=False)
    L.info(f"defects.json -> {defects_json_path}")

    for folder, atoms in structures.items():
        # P2-9：POSCAR 和 data.lmp 都必须使用 reposition_hydrogens 之后的原子顺序。
        # 含氢结构（如 MVH）的 H 由 Atoms.append() 加在数组末尾，若不重排，
        # md_steps=0 路径会直接从 POSCAR 生成 STRUCT.fdf，右电极范围将包含 H
        # 而少含等量 C，破坏同质碳电极假设。POSCAR 和 data.lmp 用同一份重排后的结构。
        atoms_repositioned = atoms if scattering_only else exporters.reposition_hydrogens(
            atoms,
            cnt_geometry.fixed_pl_per_side(m, n, buffer_pl) * l_PL * N_uc,
        )

        output_dir = structure_root / "lammps"
        output_dir.mkdir(parents=True, exist_ok=True)

        poscar_path = output_dir / "POSCAR"
        lammps_path = output_dir / "data.lmp"

        exporters.write_poscar(poscar_path, atoms_repositioned)
        exporters.write_lammps(lammps_path, atoms_repositioned)

        L.info(f"[WRITE] {folder}")
        L.info(f"  POSCAR   -> {poscar_path}")
        L.info(f"  data.lmp -> {lammps_path}")

    if scattering_only and lammps_mode == "md" and md_steps == 0:
        return
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
        lammps_mode=lammps_mode,
        lammps_device=config.get("lammps_device", "gpu"),
        buffer_pl=buffer_pl,
        release_uc=release_uc,
        fixed_atoms_per_side=0 if scattering_only else None,
    )


if __name__ == "__main__":
    main()
