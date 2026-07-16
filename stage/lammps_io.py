import os
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
    input_dir=None,
    md_steps=40000,
    files=("in.lammps", "CH.airebo-m", "run.sh")
):
    """
    为不同结构生成 LAMMPS 计算目录，并修改对应的 in.lammps 文件。

    Parameters
    ----------
    temperature : int
        模拟温度，单位 K，用于生成如 300K/5_5/DV_DV/lammps 的目录。
    structures : dict
        结构字典，函数只使用其中的键作为文件夹名称，同时用 atoms 判断是否含 H。
    chirality : tuple
        CNT 手性，例如 (5, 5)。
    l_PL : int
        一个 PL 包含的 unit cell 数。
    N_uc : int
        一个 unit cell 的原子数。
    data_root : str or Path
        数据根目录，例如 /personal/CNT_defects_package/work_flow_test/data。
    input_dir : str or Path
        LAMMPS 模板输入文件所在目录。
        如果不传，默认使用 lammps_io.py 所在目录的 ../input_files/lammps。
    md_steps : int
        热弛豫 MD 的步数，用于替换 in.lammps 末尾的 `run <N>`。
        必须与 dump2fdf 的 --every 保持一致，否则会破坏"只取最后一帧"。
    files : tuple
        需要复制到每个结构目录中的模板文件。
    """

    data_root = Path(data_root).resolve()

    if input_dir is None:
        this_file_dir = Path(__file__).resolve().parent
        input_dir = this_file_dir / "../input_files/lammps"

    input_dir = Path(input_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"源文件夹不存在: {input_dir}")

    #n_fix = 4 * l_PL * N_uc
    n_fix = 0  # 如果想全部放开，用这个

    m, n = chirality

    for folder, atoms in structures.items():
        dst_dir = data_root / f"{temperature}K" / f"{m}_{n}" / folder / "lammps"
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

        with open(lammps_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        has_hydrogen = "H" in atoms.get_chemical_symbols()

        has_mass_2 = any(
            re.match(r"^\s*mass\s+2\s+", line)
            for line in lines
        )

        new_lines = []

        for line in lines:
            # 修改固定原子数
            if re.match(r"^\s*variable\s+nfix\s+equal\s+", line):
                line = f"variable nfix equal {n_fix}\n"

            # 修改初始速度温度
            # 例如：velocity mobile create 300 23456789 mom yes rot yes dist gaussian
            if re.match(r"^\s*velocity\s+mobile\s+create\s+", line):
                parts = line.split()
                parts[3] = str(temperature)
                line = " ".join(parts) + "\n"

            # 修改 NVT 热浴温度
            # 例如：fix 1 mobile nvt temp 300.0 300.0 0.1
            if re.match(r"^\s*fix\s+1\s+mobile\s+nvt\s+temp\s+", line):
                parts = line.split()
                parts[5] = str(float(temperature))
                parts[6] = str(float(temperature))
                line = " ".join(parts) + "\n"

            # 修改热弛豫 MD 步数
            # 例如：run             40000
            # 必须与 dump2fdf 的 --every 一致，否则会多抽/漏抽帧。
            if re.match(r"^\s*run\s+\d+\s*$", line):
                line = f"run             {int(md_steps)}\n"

            # 含 H 结构：在 mass 1 后面加入 mass 2
            if has_hydrogen and re.match(r"^\s*mass\s+1\s+12", line):
                new_lines.append(line)

                if not has_mass_2:
                    new_lines.append("mass        2 1\n")

                continue

            # 含 H 结构：势函数元素列表由 C 改为 C H
            if has_hydrogen and re.match(
                r"^\s*pair_coeff\s+\*\s+\*\s+\.\/CH\.airebo-m\s+C\s*$",
                line
            ):
                line = "pair_coeff * * ./CH.airebo-m  C H\n"

            new_lines.append(line)

        with open(lammps_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        L.info(
            f"已生成 {dst_dir}，"
            f"nfix = {n_fix}，"
            f"T = {temperature} K，"
            f"md_steps = {md_steps}，"
            f"含H = {has_hydrogen}"
        )