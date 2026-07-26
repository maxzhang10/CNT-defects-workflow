#!/usr/bin/env python3

import re
import shutil
from pathlib import Path

import logkit as L


def prepare_lammps_inputs(
    temperature,
    structures,
    chirality,
    l_PL,
    N_uc,
    data_root="../data",
    structure_root=None,
    input_dir=None,
    md_steps=40000,
    files=None,
    lammps_seed=None,
):
    """
    为不同结构生成 LAMMPS 计算目录，并修改 in.lammps。

    lammps_seed 只控制当前独立 LAMMPS 轨迹，必须由上层显式传入。

    structure_root 可显式指定当前结构/replica 的根目录；未提供时
    保留旧目录规则 data_root/温度/手性/结构名。
    """
    if files is None:
        files = ("in.lammps", "CH.airebo-m", "run.sh")

    if lammps_seed is None:
        raise ValueError("必须显式传入 lammps_seed")

    lammps_seed = int(lammps_seed)
    if lammps_seed <= 0:
        raise ValueError(
            f"lammps_seed 必须为正整数，当前值: {lammps_seed}"
        )

    data_root = Path(data_root).resolve()

    if structure_root is not None:
        structure_root = Path(structure_root).resolve()
        if len(structures) != 1:
            raise ValueError(
                "显式指定 structure_root 时，structures 必须只包含一个结构目录"
            )

    if input_dir is None:
        this_file_dir = Path(__file__).resolve().parent
        input_dir = this_file_dir / "../input_files/lammps"

    input_dir = Path(input_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"源文件夹不存在: {input_dir}")

    n_fix = 4 * l_PL * N_uc
    m, n = chirality

    for folder, atoms in structures.items():
        if structure_root is None:
            current_structure_root = (
                data_root
                / f"{temperature}K"
                / f"{m}_{n}"
                / folder
            )
        else:
            current_structure_root = structure_root

        dst_dir = current_structure_root / "lammps"
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 1. 复制模板文件
        for filename in files:
            src_file = input_dir / filename
            dst_file = dst_dir / filename

            if not src_file.is_file():
                raise FileNotFoundError(f"模板文件不存在: {src_file}")

            shutil.copy(str(src_file), str(dst_file))

        # 2. 修改当前结构的 in.lammps
        lammps_file = dst_dir / "in.lammps"

        with lammps_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            lines = file.readlines()

        has_hydrogen = "H" in atoms.get_chemical_symbols()
        has_mass_2 = any(
            re.match(r"^\s*mass\s+2\s+", line)
            for line in lines
        )

        found_nfix = False
        found_velocity = False
        found_nvt = False
        found_run = False
        new_lines = []

        for line in lines:
            # 修改固定原子数
            if re.match(r"^\s*variable\s+nfix\s+equal\s+", line):
                line = f"variable nfix equal {n_fix}\n"
                found_nfix = True

            # 修改初始速度温度与随机种子：
            # velocity mobile create <temperature> <seed> ...
            if re.match(r"^\s*velocity\s+mobile\s+create\s+", line):
                parts = line.split()
                if len(parts) < 5:
                    raise RuntimeError(
                        f"velocity create 行格式不完整: {line.rstrip()}"
                    )

                parts[3] = str(temperature)
                parts[4] = str(lammps_seed)
                line = " ".join(parts) + "\n"
                found_velocity = True

            # 修改 NVT 热浴温度
            if re.match(r"^\s*fix\s+1\s+mobile\s+nvt\s+temp\s+", line):
                parts = line.split()
                if len(parts) < 7:
                    raise RuntimeError(
                        f"NVT fix 行格式不完整: {line.rstrip()}"
                    )

                parts[5] = str(float(temperature))
                parts[6] = str(float(temperature))
                line = " ".join(parts) + "\n"
                found_nvt = True

            # 修改热弛豫 MD 步数
            if re.match(r"^\s*run\s+\d+\s*(?:#.*)?$", line):
                line = f"run             {int(md_steps)}\n"
                found_run = True

            # 含 H 结构：在 mass 1 后加入 mass 2
            if has_hydrogen and re.match(r"^\s*mass\s+1\s+12", line):
                new_lines.append(line)

                if not has_mass_2:
                    new_lines.append("mass        2 1\n")

                continue

            # 含 H 结构：势函数元素列表由 C 改为 C H
            if has_hydrogen and re.match(
                r"^\s*pair_coeff\s+\*\s+\*\s+\./CH\.airebo-m\s+C\s*$",
                line,
            ):
                line = "pair_coeff * * ./CH.airebo-m  C H\n"

            new_lines.append(line)

        # 这些字段缺失时直接报错，避免模板没有被真正改到。
        missing_edits = []
        if not found_nfix:
            missing_edits.append("variable nfix equal")
        if not found_velocity:
            missing_edits.append("velocity mobile create")
        if not found_nvt:
            missing_edits.append("fix 1 mobile nvt temp")
        if not found_run:
            missing_edits.append("run <md_steps>")

        if missing_edits:
            raise RuntimeError(
                f"{lammps_file} 中未找到以下模板行，无法安全修改: "
                + ", ".join(missing_edits)
            )

        with lammps_file.open("w", encoding="utf-8") as file:
            file.writelines(new_lines)

        L.info(
            f"已生成 {dst_dir}，"
            f"nfix={n_fix}，"
            f"T={temperature} K，"
            f"md_steps={md_steps}，"
            f"lammps_seed={lammps_seed}，"
            f"含H={has_hydrogen}"
        )
