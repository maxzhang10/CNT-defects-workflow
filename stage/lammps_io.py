#!/usr/bin/env python3

import os
import re
import shutil
from pathlib import Path

import logkit as L

import cnt_geometry


def symlink_force(src: Path, dst: Path) -> None:
    """建软连接 dst -> src，已存在则先删后建。

    机器学习力场（.pth）体积大，每个 lammps 工作目录若各拷一份会
    大量重复占用空间；这里统一软连接到 input_dir 下的源文件。
    用绝对路径建链，使 leaf 目录整体迁移后连接仍然有效。
    """
    src_resolved = src.resolve()

    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            # 目录（理论上不会出现），用 rmtree 兜底。
            shutil.rmtree(dst)

    os.symlink(src_resolved, dst)


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
    model_file=None,
    lammps_mode="md",
    buffer_pl=None,
):
    """
    为不同结构生成 LAMMPS 计算目录，并修改 in.lammps。

    lammps_mode:
        "md"  —— 热弛豫 MD。拷贝 in.lammps 模板，改写温度、随机种子、
                 md_steps 等字段；lammps_seed 必填。
        "opt" —— 纯几何优化。拷贝 opt.lammps 模板（落到工作目录时仍命名
                 为 in.lammps，run.sh / sub_lmps.py / 工作目录识别无需
                 改动），只改写固定原子数 nfix（及含 H 时的 mass/pair_coeff），
                 不涉及温度、种子与步数；lammps_seed 忽略。

    lammps_seed 只控制当前独立 LAMMPS 轨迹，md 模式下必须由上层显式传入。

    structure_root 可显式指定当前结构/replica 的根目录；未提供时
    保留旧目录规则 data_root/温度/手性/结构名。
    """
    if lammps_mode not in ("md", "opt"):
        raise ValueError(
            f"lammps_mode 必须为 'md' 或 'opt'，当前值: {lammps_mode}"
        )

    if files is None:
        if lammps_mode == "opt":
            # opt 模板拷贝时重命名为 in.lammps，下游统一只认 in.lammps。
            files = (("opt.lammps", "in.lammps"), ("run.sh", "run.sh"))
        else:
            # 力场（.pth）单独从 input_dir 里 glob，不写死文件名；
            # 这里只列普通模板文件，in.lammps 之后会被就地改写故走拷贝。
            files = (("in.lammps", "in.lammps"), ("run.sh", "run.sh"))

    if lammps_mode == "md":
        if lammps_seed is None:
            raise ValueError("md 模式必须显式传入 lammps_seed")

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

    # 力场文件：input_dir 下任意 *.pth 都视为 DeepMD 机器学习力场。
    # 体积大，leaf 目录用软连接指向它，不写死文件名。
    model_files = sorted(input_dir.glob("*.pth"))

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"未在 input_dir 中找到 .pth 力场文件: {input_dir}"
        )

    if model_file is not None:
        model_src = input_dir / model_file
        if not model_src.is_file():
            raise FileNotFoundError(
                f"指定的力场文件不存在: {model_src}"
            )
        model_src = model_src.resolve()
    else:
        if len(model_files) > 1:
            names = "\n".join(f"  {p.name}" for p in model_files)
            raise ValueError(
                f"input_dir 中找到多个 .pth 力场文件，"
                f"请用 model_file 指定一个:\n{names}"
            )
        model_src = model_files[0].resolve()

    model_filename = model_src.name

    m, n = chirality
    # 每侧固定原子数与结构生成保持一致（2PL电极+buffer_pl缓冲），
    # 但最靠中间的 1 个 uc 放开不固定（左右共 2 个 uc），
    # 见 cnt_geometry.fixed_atoms_per_side / RELEASE_UC_PER_SIDE。
    n_fix = cnt_geometry.fixed_atoms_per_side(m, n, l_PL, N_uc, buffer_pl)

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

        # 1. 分发模板文件
        # in.lammps / run.sh 体积小且 in.lammps 之后会被就地改写，走拷贝；
        # *.pth 机器学习力场体积大，改用软连接避免重复占用空间。
        for src_name, dst_name in files:
            src_file = input_dir / src_name
            dst_file = dst_dir / dst_name

            if not src_file.is_file():
                raise FileNotFoundError(f"模板文件不存在: {src_file}")

            shutil.copy(str(src_file), str(dst_file))

        # 力场文件：软连接到 input_dir 下的源 .pth，保留源文件名。
        symlink_force(model_src, dst_dir / model_filename)

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
            # 修改固定原子数（md / opt 两种模式都需要）
            if re.match(r"^\s*variable\s+nfix\s+equal\s+", line):
                line = f"variable nfix equal {n_fix}\n"
                found_nfix = True

            if lammps_mode == "md":
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

            # 把 pair_style deepmd <任意>.pth 改写成实际的力场文件名
            # （input_dir 里 glob 出的 .pth 文件名不固定）。
            # 模板里的旧路径可能是相对/绝对路径，也可能带英文双引号，
            # 统一改写成指向工作目录内软链接的 ./<filename>（不带引号）。
            if re.match(r'^\s*pair_style\s+deepmd\s+"?\S+\.pth"?\s*$', line):
                line = re.sub(
                    r'(^\s*pair_style\s+deepmd\s+)"?\S+\.pth"?(\s*$)',
                    rf"\g<1>./{model_filename}\g<2>",
                    line,
                )

            # 含 H 结构：势函数元素列表由 C 改为 C H
            # DeepMD 的 pair_coeff 形如 "pair_coeff * * C"，模型路径在
            # pair_style 行而非 pair_coeff 行，故只匹配元素列表。
            if has_hydrogen and re.match(
                r"^\s*pair_coeff\s+\*\s+\*\s+C\s*$",
                line,
            ):
                line = "pair_coeff * * C H\n"

            new_lines.append(line)

        # 这些字段缺失时直接报错，避免模板没有被真正改到。
        # opt 模板没有 velocity/nvt/run 行，只要求 nfix。
        missing_edits = []
        if not found_nfix:
            missing_edits.append("variable nfix equal")
        if lammps_mode == "md":
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

        if lammps_mode == "md":
            L.info(
                f"已生成 {dst_dir}，"
                f"mode=md，"
                f"nfix={n_fix}，"
                f"T={temperature} K，"
                f"md_steps={md_steps}，"
                f"lammps_seed={lammps_seed}，"
                f"含H={has_hydrogen}"
            )
        else:
            L.info(
                f"已生成 {dst_dir}，"
                f"mode=opt（几何优化，模板 opt.lammps），"
                f"nfix={n_fix}，"
                f"含H={has_hydrogen}"
            )
