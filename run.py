#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import shutil
import json
from pathlib import Path

# 让编排层与各 stage 脚本共用同一份 logkit
_STAGE_DIR = Path(__file__).resolve().parent / "stage"
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))
import logkit as L

def run_cmd(cmd, dry_run=False, env=None):
    L.run(" ".join(map(str, cmd)))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)

def has_dump_files(p: Path) -> bool:
    """
    判断目录下是否存在 LAMMPS dump 文件。
    避免把 dump2fdf_batch.py 这种脚本误判为 dump。
    """
    if not p.is_dir():
        return False

    for f in p.iterdir():
        if not f.is_file():
            continue

        name = f.name

        if f.suffix in {".dump", ".lammpstrj"}:
            return True

        if name == "dump":
            return True

        if name.startswith("dump.") or name.startswith("dump_") or name.startswith("dump-"):
            return True

    return False


def is_leaf_structure_dir(p: Path) -> bool:
    """
    判断是否是结构目录，例如:
        data/300K/5_5/DV_DV

    支持两种情况：
    1. dump 直接在结构目录下
    2. dump 在结构目录/lammps 下
    """
    if not p.is_dir():
        return False

    if p.name in {"dpnegf", "lammps"}:
        return False

    if "dpnegf" in p.parts:
        return False

    # 情况 1：dump 直接在结构目录下
    if has_dump_files(p):
        return True

    # 情况 2：dump 在结构目录/lammps 下
    lammps_dir = p / "lammps"
    if has_dump_files(lammps_dir):
        return True

    return False


def find_structure_dirs(root: Path):
    """
    返回真正的结构目录，而不是 lammps 目录。

    例如返回:
        data/300K/5_5/DV_DV

    不返回:
        data/300K/5_5/DV_DV/lammps
    """
    root = root.resolve()

    # 如果 root 本身就是 lammps 目录，则返回它的父目录
    if root.name == "lammps" and has_dump_files(root):
        return [root.parent]

    # 如果 root 本身就是结构目录
    if is_leaf_structure_dir(root):
        return [root]

    dirs = []

    # 优先从所有 lammps 目录反推结构目录
    for lammps_dir in root.rglob("lammps"):
        if "dpnegf" in lammps_dir.parts:
            continue

        if has_dump_files(lammps_dir):
            dirs.append(lammps_dir.parent)

    # 兼容 dump 直接放在结构目录下的情况
    for p in root.rglob("*"):
        if is_leaf_structure_dir(p):
            dirs.append(p)

    return sorted(set(dirs))

def is_lammps_workdir(p: Path) -> bool:
    """
    判断是否是 LAMMPS 工作目录。
    例如:
        data/300K/5_5/DV_DV/lammps
    """
    if not p.is_dir():
        return False

    if p.name != "lammps":
        return False

    required = ["data.lmp", "in.lammps", "CH.airebo-m", "run.sh"]

    return all((p / name).exists() for name in required)


def find_lammps_workdirs(root: Path):
    """
    在 root 下递归寻找所有 lammps 工作目录。
    """
    if is_lammps_workdir(root):
        return [root]

    workdirs = []
    for p in root.rglob("lammps"):
        if is_lammps_workdir(p):
            workdirs.append(p)

    return sorted(set(workdirs))

def is_dpnegf_workdir(p: Path) -> bool:
    """
    判断是否是 DPNEGF 工作目录。
    例如:
        data/300K/5_5/DV_DV/dpnegf/40000
    """
    if not p.is_dir():
        return False

    required = [
        "input.json",
        "run.py",
        "run.sh",
    ]

    for name in required:
        if not (p / name).exists():
            return False

    if not list(p.glob("*.xyz")):
        return False

    if not list(p.glob("nnenv*.pth")):
        return False

    return True


def find_dpnegf_workdirs_from_structures(structure_dirs):
    """
    只从本轮识别到的结构目录中寻找 DPNEGF 工作目录，
    避免扫到旧的漂移目录或者其他历史残留。
    """
    workdirs = []

    for struct_dir in structure_dirs:
        dpnegf_root = struct_dir / "dpnegf"

        if not dpnegf_root.exists():
            continue

        # 兼容 dpnegf 本身就是工作目录的情况
        if is_dpnegf_workdir(dpnegf_root):
            workdirs.append(dpnegf_root)

        # 常规情况：dpnegf/40000, dpnegf/80000, ...
        for p in dpnegf_root.rglob("*"):
            if is_dpnegf_workdir(p):
                workdirs.append(p)

    return sorted(set(workdirs))

def clean_dpnegf_output_cache(root):
    """
    在 root 目录下递归寻找所有 */output/ 目录，并清理：

    1. output/self_energy 目录
    2. output/HS_device.h5
    3. output/HS_lead_L.h5
    4. output/HS_lead_R.h5

    只清理形如 */output/... 的内容，避免误删其他同名文件。
    """
    root = Path(root).resolve()

    if not root.exists():
        L.error(f"清理缓存时 root 不存在: {root}")
        return

    removed_dirs = 0
    removed_files = 0

    hs_files = [
        "HS_device.h5",
        "HS_lead_L.h5",
        "HS_lead_R.h5",
    ]

    for output_dir in sorted(root.rglob("output")):
        if not output_dir.is_dir():
            continue

        # 1. 删除 self_energy 目录
        self_energy_dir = output_dir / "self_energy"
        if self_energy_dir.is_dir():
            L.clean(f"rm -r {self_energy_dir}")
            shutil.rmtree(self_energy_dir)
            removed_dirs += 1

        # 2. 删除 HS 矩阵文件
        for fname in hs_files:
            fpath = output_dir / fname
            if fpath.is_file():
                L.clean(f"rm {fpath}")
                fpath.unlink()
                removed_files += 1

    L.clean(f"已清理 self_energy 目录: {removed_dirs} 个，HS 文件: {removed_files} 个")

def main():
    parser = argparse.ArgumentParser(
        description="CNT workflow: eledefects -> dump -> fdf -> xyz -> copy dpnegf input"
    )

    parser.add_argument(
        "--root",
        default=None,
        help="数据根目录。可以是总目录 data，也可以是单个结构目录，例如 data/300K/5_5/5775",
    )

    parser.add_argument(
        "--stage",
        default="/personal/CNT_defects_package/work_flow_test/stage",
        help="脚本所在目录，默认是 work_flow_test/stage",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="指定 Python 解释器，默认使用当前环境的 python",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="config.json 路径。如果提供，会通过环境变量 CNT_CONFIG 传给 eledefects.py",
    )


    parser.add_argument(
        "--skip-ele",
        action="store_true",
        help="跳过 eledefects.py，只执行后续 dump2fdf/fdf2xyz/copy_input",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不实际执行",
    )

    parser.add_argument(
    "--skip-dpnegf",
    action="store_true",
    help="跳过 DPNEGF 运行，只生成 DPNEGF 输入文件",
    )
    args = parser.parse_args()

    stage = Path(args.stage).resolve()
    py = args.python

    if args.config is not None:
        config_path = Path(args.config).resolve()
    else:
        config_path = Path(__file__).resolve().parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config 不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # MD 步数：dump2fdf 用它作为 --every，保证只抽出最后一帧。
    # 必须与 in.lammps 里 `run <N>` 一致（同由 config.md_steps 驱动）。
    md_steps = int(config.get("md_steps", 40000))

    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        if "data_root" not in config:
            raise KeyError("config 中没有定义 data_root")
        root = Path(config["data_root"]).resolve()

    ele_defects_script = stage / "ele_defects_ele.py"
    sub_lammps_script = stage / "sub_lmps.py"
    sub_dpnegf_script = stage / "sub_dpnegf.py"

    dump2fdf = stage / "dump2fdf_batch.py"
    fdf2xyz = stage / "fdf2xyz.py"
    copy_input = stage / "copy_input_dpnegf.py"

    

    required_scripts = [dump2fdf, fdf2xyz, copy_input]

    if not args.skip_ele:
        required_scripts = [ele_defects_script, sub_lammps_script] + required_scripts

    if not args.skip_dpnegf:
        required_scripts.append(sub_dpnegf_script)

    if not root.exists():
        raise FileNotFoundError(f"root 不存在: {root}")

    env = os.environ.copy()



    env["CNT_CONFIG"] = str(config_path)

    L.info(f"CNT_CONFIG = {config_path}")
    L.info(f"root       = {root}")
    L.info(f"stage      = {stage}")

    # ============================================================
    # 0. 先运行 eledefects.py
    # ============================================================
    if not args.skip_ele:
        L.phase(1, 6, "生成缺陷几何 (eledefects)")
        run_cmd(
            [
                py,
                str(ele_defects_script),
            ],
            dry_run=args.dry_run,
            env=env,
        )
        # ============================================================
        # 0.5 运行lammps任务
        # ============================================================
        L.phase(2, 6, "LAMMPS 退火")
        lammps_workdirs = find_lammps_workdirs(root)

        if not lammps_workdirs:
            raise RuntimeError(
                f"没有找到 LAMMPS 工作目录。请确认 {root} 下存在类似:\n"
                f"  data/300K/5_5/DV_DV/lammps\n"
                f"并且其中包含 data.lmp / in.lammps / CH.airebo-m / run.sh"
            )

        L.info(f"找到 LAMMPS 工作目录: {len(lammps_workdirs)} 个")
        for i, w in enumerate(lammps_workdirs, 1):
            L.item(i, len(lammps_workdirs), w)

        for workdir in lammps_workdirs:
            run_cmd(
                [
                    py,
                    str(sub_lammps_script),
                    str(workdir),
                ],
                dry_run=args.dry_run,
                env=env,
            )
    # ============================================================
    # 1. 查找含 dump 的结构目录
    # ============================================================
    L.phase(3, 6, "dump -> fdf")
    structure_dirs = find_structure_dirs(root)

    if not structure_dirs:
        raise RuntimeError(
            f"没有找到结构目录。请确认 {root} 下是否存在 dump / *.dump / *.lammpstrj 文件。\n"
            f"注意：eledefects.py 通常只生成 lammps 输入文件；如果还没运行 LAMMPS，就不会有 dump。"
        )

    L.info(f"找到结构目录: {len(structure_dirs)} 个")
    for i, d in enumerate(structure_dirs, 1):
        L.item(i, len(structure_dirs), d)

    # ============================================================
    # 2. 每个结构目录单独 dump -> fdf
    # ============================================================
    for struct_dir in structure_dirs:
        outroot = struct_dir / "dpnegf"

        lammps_dir = struct_dir / "lammps"

        if lammps_dir.exists():
            dump_root = lammps_dir
        else:
            dump_root = struct_dir

        run_cmd(
            [
                py,
                str(dump2fdf),
                "--root",
                str(dump_root),
                "--outroot",
                str(outroot),
                "--every",
                str(md_steps),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    # ============================================================
    # 3. 对每个结构的 dpnegf 目录做 fdf -> xyz
    # ============================================================
    L.phase(4, 6, "fdf -> xyz")
    for struct_dir in structure_dirs:
        dpnegf_dir = struct_dir / "dpnegf"

        if not dpnegf_dir.exists():
            L.warn(f"dpnegf 目录不存在，跳过 fdf2xyz: {dpnegf_dir}")
            continue

        run_cmd(
            [
                py,
                str(fdf2xyz),
                "--root",
                str(dpnegf_dir),
                "--outroot",
                str(dpnegf_dir),
                "--config",
                str(config_path),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    # ============================================================
    # 4. 复制 dpnegf 输入文件
    # ============================================================
    L.phase(5, 6, "复制 DPNEGF 输入")
    project_root = Path(__file__).resolve().parent
    dpnegf_input_dir = project_root / "input_files" / "dpnegf"

    if not dpnegf_input_dir.exists():
        raise FileNotFoundError(f"dpnegf input_dir 不存在: {dpnegf_input_dir}")

    # 找 dpnegf 模型文件
    model_files = sorted(dpnegf_input_dir.glob("*.pth"))

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"dpnegf input_dir 中未找到 .pth 模型文件: {dpnegf_input_dir}"
        )

    if len(model_files) > 1:
        names = "\n".join(f"  {p.name}" for p in model_files)
        raise ValueError(
            f"dpnegf input_dir 中找到多个 .pth 模型文件，请只保留一个或手动指定：\n{names}"
        )

    model_file = model_files[0].name

    L.info(f"DPNEGF 模型文件 = {model_file}")

    for struct_dir in structure_dirs:
        dpnegf_dir = struct_dir / "dpnegf"

        if not dpnegf_dir.exists():
            L.warn(f"dpnegf 目录不存在，跳过 copy_input: {dpnegf_dir}")
            continue

        run_cmd(
            [
                py,
                str(copy_input),
                "--root",
                str(dpnegf_dir),
                "--input-dir",
                str(dpnegf_input_dir),
                "--model-file",
                model_file,
                "--config",
                str(config_path),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    # ============================================================
    # 5. 运行所有刚刚建立好的 DPNEGF 工作目录
    # ============================================================
    if not args.skip_dpnegf:
        L.phase(6, 6, "运行 DPNEGF")
        dpnegf_workdirs = find_dpnegf_workdirs_from_structures(structure_dirs)

        if not dpnegf_workdirs:
            raise RuntimeError(
                "没有找到 DPNEGF 工作目录。请确认每个 leaf 目录下存在：\n"
                "  input.json / run.py / run.sh / *.xyz / nnenv*.pth\n"
                "常见目录应该类似：\n"
                "  data/300K/5_5/DV_DV/dpnegf/40000"
            )

        L.info(f"找到 DPNEGF 工作目录: {len(dpnegf_workdirs)} 个")
        for i, w in enumerate(dpnegf_workdirs, 1):
            L.item(i, len(dpnegf_workdirs), w)

        for workdir in dpnegf_workdirs:
            run_cmd(
                [
                    py,
                    str(sub_dpnegf_script),
                    str(workdir),
                ],
                dry_run=args.dry_run,
                env=env,
            )
            clean_dpnegf_output_cache(workdir)
            

        
if __name__ == "__main__":
    main()