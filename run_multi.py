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
from slurm_utils import wait_for_jobs, check_flags
from negf_provenance import result_is_readable

def run_cmd(cmd, dry_run=False, env=None, cwd=None):
    L.run(" ".join(map(str, cmd)))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def collect_job_ids(workdirs):
    """
    从每个 workdir 读取 sub_*.py 写下的 job_id.txt，收集 SLURM job_id。
    没有 job_id.txt 的（如已 done 而跳过、或 dry-run）自动忽略。
    """
    ids = []
    for w in workdirs:
        f = Path(w) / "job_id.txt"
        if f.exists():
            jid = f.read_text().strip()
            if jid and jid != "local":
                ids.append(jid)
    return ids


def barrier_and_check(workdirs, kind, poll_interval, dry_run):
    """
    SLURM 屏障：等这批 workdir 的作业全部离开队列，再按 flag 检查成败。
    kind: "lammps" 或 "dpnegf"。任一失败/缺失 flag 时抛错，阻止进入下一阶段。
    """
    if dry_run:
        L.info(f"[dry-run] 跳过 {kind} SLURM 屏障等待")
        return

    job_ids = collect_job_ids(workdirs)
    wait_for_jobs(job_ids, poll_interval=poll_interval, label=kind.upper())

    done_dirs, failed_dirs, missing_dirs = check_flags(workdirs, kind)
    L.info(f"{kind} 结果: 完成={len(done_dirs)} 失败={len(failed_dirs)} 缺flag={len(missing_dirs)}")

    # P1-10：done flag 不等于成功。对 DPNEGF，进一步校验结果文件
    # output/negf.out.pth 存在、非空且可读取；损坏/缺失的 done 目录视作失败，
    # 让 batch_generate 的 per-task 重试能捕获，而不是拖到批次末尾电导汇总。
    if kind == "dpnegf":
        bad_result_dirs = []
        for d in done_dirs:
            if not result_is_readable(d):
                bad_result_dirs.append(d)
        if bad_result_dirs:
            for d in bad_result_dirs:
                L.error(
                    f"dpnegf done flag 存在但结果文件缺失/空/不可读取: {d}"
                )
            # 把这些目录从 done 移到 failed，让下面的统一报错生效。
            done_dirs = [d for d in done_dirs if d not in set(bad_result_dirs)]
            failed_dirs.extend(bad_result_dirs)

    if failed_dirs or missing_dirs:
        for d in failed_dirs:
            L.error(f"{kind} 失败: {d}（见 {kind}_failed.flag / slurm-*.err）")
        for d in missing_dirs:
            L.error(f"{kind} 无 done/failed flag（作业可能被杀或超时）: {d}")
        raise RuntimeError(
            f"{kind} 阶段有作业未成功，已中止后续阶段。"
            f"修复后删除对应 {kind}_failed.flag / *_submitted.flag 重跑。"
        )

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

    # 排除 dpnegf 工作目录（只检查直接目录名，避免误伤包含 dpnegf 的父路径）
    if p.name == "dpnegf":
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
        # 跳过 dpnegf/<timestep>/lammps 这样的情况，只检查父目录名
        if lammps_dir.parent.name == "dpnegf":
            continue

        if has_dump_files(lammps_dir):
            dirs.append(lammps_dir.parent)

    # 兼容 dump 直接放在结构目录下的情况
    for p in root.rglob("*"):
        if is_leaf_structure_dir(p):
            dirs.append(p)

    return sorted(set(dirs))


def find_poscar_structure_dirs(root: Path):
    """Find structure directories containing POSCAR for md_steps=0."""
    root = root.resolve()
    if (root / "POSCAR").is_file():
        return [root]
    dirs = set()
    for path in root.rglob("POSCAR"):
        if path.is_file():
            # ele_multi_defects_ele.py writes POSCAR under <structure>/lammps.
            dirs.add(path.parent.parent.resolve() if path.parent.name == "lammps" else path.parent.resolve())
    return sorted(dirs)

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

    # P2-12：接受任意合法名称的 .pth 模型文件，不硬编码 nnenv*.pth。
    # staging 阶段接受任意 *.pth，工作目录检查应与之一致。
    if not list(p.glob("*.pth")):
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

def clean_dpnegf_output_cache(root, save_self_energy=False, self_energy_save_path=None):
    """
    在 root 目录下递归寻找所有 */output/ 目录，并清理：

    1. output/self_energy 目录（save_self_energy=True 时保留）
    2. output/HS_device.h5
    3. output/HS_lead_L.h5
    4. output/HS_lead_R.h5

    P2-6：若配置中 self_energy_cache.save_path 指向 leaf 下的独立目录
    （如 ./self_energy/，而非 output/self_energy），也一并清理，否则
    即使 save_self_energy=False，实际 ./self_energy/ 仍会保留。

    只清理形如 */output/... 或配置指定的路径，避免误删其他同名文件。
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

    # 解析配置中的自能缓存路径名（可能是 "self_energy"、"./self_energy/" 等）。
    se_dirname = None
    if self_energy_save_path:
        se_path = Path(self_energy_save_path)
        # 取相对路径的最后一段作为目录名（如 "./self_energy/" -> "self_energy"）。
        se_dirname = se_path.name or se_path.parent.name
        # 若 save_path 是 output/self_energy，则已被下面的 output 循环覆盖。
        if se_path.parts and se_path.parts[0] == "output":
            se_dirname = None  # 已在 output 循环中处理

    for output_dir in sorted(root.rglob("output")):
        if not output_dir.is_dir():
            continue

        leaf_dir = output_dir.parent

        # 1. 删除 output/self_energy 目录（如配置要求保存则跳过）
        self_energy_dir = output_dir / "self_energy"
        if self_energy_dir.is_dir() and not save_self_energy:
            L.clean(f"rm -r {self_energy_dir}")
            shutil.rmtree(self_energy_dir)
            removed_dirs += 1

        # P2-6：删除配置指定的独立自能缓存目录（如 leaf/self_energy）。
        if se_dirname:
            alt_se_dir = leaf_dir / se_dirname
            if alt_se_dir.is_dir() and not save_self_energy:
                L.clean(f"rm -r {alt_se_dir}")
                shutil.rmtree(alt_se_dir)
                removed_dirs += 1

        # 2. 删除 HS 矩阵文件
        for fname in hs_files:
            fpath = output_dir / fname
            if fpath.is_file():
                L.clean(f"rm {fpath}")
                fpath.unlink()
                removed_files += 1

    if save_self_energy:
        L.clean(f"保留 self_energy 目录，已清理 HS 文件: {removed_files} 个")
    else:
        L.clean(f"已清理 self_energy 目录: {removed_dirs} 个，HS 文件: {removed_files} 个")


def remove_lammps_dump_files(lammps_dir: Path) -> int:
    """Remove .dump trajectories only after dump->FDF conversion succeeded."""
    removed = 0
    if not lammps_dir.is_dir():
        return removed
    for dump_file in lammps_dir.rglob("*.dump"):
        if dump_file.is_file():
            dump_file.unlink()
            removed += 1
            L.clean(f"rm {dump_file}")
    return removed


def build_collect_conductance_cmd(
    python: str,
    script: Path,
    root: Path,
    configuration_dirs: list[Path],
) -> list[str]:
    """Build the collect_md_conductance.py invocation.

    collect_md_conductance.py only accepts --root / --configuration-dir /
    --configuration-settings -- it has no --config / --structure-dir options,
    so passing those makes argparse raise "ambiguous option" *after* all the
    expensive DPNEGF work has finished. Conductance values (Ef/Ec/Ev) and the
    transport temperature are read by the collector directly from each
    replica's negf.out.pth (precomputed by DPNEGF), so no external energy
    reference or temperature needs to be passed here.
    """
    cmd = [
        python,
        str(script),
        "--root",
        str(root),
    ]
    for struct_dir in configuration_dirs:
        cmd.extend(["--configuration-dir", str(struct_dir)])
    return cmd


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
        default=str(Path(__file__).resolve().parent / "stage"),
        help="脚本所在目录，默认是本工作流的 stage",
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

    parser.add_argument(
        "--skip-conductance",
        action="store_true",
        help="跳过最终的电导日志收集",
    )

    parser.add_argument(
        "--conductance-python",
        default=None,
        help="运行电导收集脚本的 Python；默认复用 --python（需要 torch）",
    )

    parser.add_argument(
        "--scheduler",
        choices=["local", "slurm"],
        default="slurm",
        help="local: 逐个 bash run.sh 阻塞跑; slurm: 批量 sbatch 提交后按阶段屏障等待",
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="slurm 模式下 squeue 轮询间隔（秒）",
    )
    args = parser.parse_args()

    stage = Path(args.stage).resolve()
    py = args.python

    if args.config is not None:
        config_path = Path(args.config).resolve()
    else:
        config_path = Path(__file__).resolve().parent / "config_multi.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config 不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # MD 步数：dump2fdf 用它作为 --every，保证只抽出最后一帧。
    # 必须与 in.lammps 里 `run <N>` 一致（同由 config.md_steps 驱动）。
    md_steps = int(config.get("md_steps", 40000))
    if md_steps < 0:
        raise ValueError("md_steps must be greater than or equal to zero")

    # 独立轨迹模式：一次 run_multi.py 对应一条 LAMMPS 轨迹，
    # 后续固定只提取 md_steps 对应的最后一帧。

    if args.root is not None:
        root = Path(args.root).resolve()
    else:
        if "data_root" not in config:
            raise KeyError("config 中没有定义 data_root")
        root = Path(config["data_root"]).resolve()

    ele_defects_script = stage / "ele_multi_defects_ele.py"
    sub_lammps_script = stage / "sub_lmps.py"
    sub_dpnegf_script = stage / "sub_dpnegf.py"
    collect_conductance_script = stage / "collect_md_conductance.py"

    dump2fdf = stage / "dump2fdf_batch.py"
    fdf2xyz = stage / "fdf2xyz.py"
    copy_input = stage / "copy_input_dpnegf.py"

    

    required_scripts = [dump2fdf, fdf2xyz, copy_input]

    if not args.skip_ele:
        required_scripts = [ele_defects_script, sub_lammps_script] + required_scripts

    if not args.skip_dpnegf:
        required_scripts.append(sub_dpnegf_script)
        if not args.skip_conductance:
            required_scripts.append(collect_conductance_script)

    missing_scripts = [p for p in required_scripts if not p.is_file()]
    if missing_scripts:
        raise FileNotFoundError(
            "缺少工作流脚本:\n" + "\n".join(f"  {p}" for p in missing_scripts)
        )


    if not root.exists():
        L.warn(f"root 目录不存在，先创建: {root}"); root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()



    env["CNT_CONFIG"] = str(config_path)

    # 当前 run_multi.py 只负责一个明确的工作根目录。
    # 将它显式传给结构生成脚本，避免 data_root 的旧目录规则
    # 把多个 replica 写到共同的配置目录。
    env["CNT_STRUCTURE_ROOT"] = str(root)

    L.info(f"CNT_CONFIG         = {config_path}")
    L.info(f"CNT_STRUCTURE_ROOT = {root}")
    L.info(f"root               = {root}")
    L.info(f"stage              = {stage}")

    # ============================================================
    # 0. 先运行 eledefects.py
    # ============================================================
    if not args.skip_ele:
        L.phase(1, 7, "生成缺陷几何 (eledefects)")
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
        L.phase(2, 7, "LAMMPS 退火")
        lammps_workdirs = find_lammps_workdirs(root)

        if md_steps == 0:
            L.info("md_steps=0，跳过 LAMMPS 运行，直接使用 POSCAR")
        elif not lammps_workdirs:
            raise RuntimeError(
                f"没有找到 LAMMPS 工作目录。请确认 {root} 下存在类似:\n"
                f"  data/300K/5_5/DV_DV/lammps\n"
                f"并且其中包含 data.lmp / in.lammps / CH.airebo-m / run.sh"
            )

        if md_steps > 0:
            L.info(f"找到 LAMMPS 工作目录: {len(lammps_workdirs)} 个")
            for i, w in enumerate(lammps_workdirs, 1):
                L.item(i, len(lammps_workdirs), w)
            for workdir in lammps_workdirs:
                run_cmd([py, str(sub_lammps_script), str(workdir), "--scheduler", args.scheduler], dry_run=args.dry_run, env=env)

        # SLURM 屏障：等所有 LAMMPS 作业跑完再进入 dump->fdf（后者依赖 dump 产物）。
        # local 模式下每个 sub_lmps 已阻塞跑完，这里等价于一次性 flag 复核。
        if md_steps > 0 and args.scheduler == "slurm":
            barrier_and_check(
                lammps_workdirs, "lammps",
                poll_interval=args.poll_interval,
                dry_run=args.dry_run,
            )
    # ============================================================
    # 1. 查找含 dump 的结构目录
    # ============================================================
    L.phase(3, 7, "dump -> fdf")
    structure_dirs = find_poscar_structure_dirs(root) if md_steps == 0 else find_structure_dirs(root)

    if not structure_dirs:
        raise RuntimeError(
            (
                f"没有找到结构目录。请确认 {root} 下存在 POSCAR 文件。"
                if md_steps == 0 else
                f"没有找到结构目录。请确认 {root} 下是否存在 dump / *.dump / *.lammpstrj 文件。\n"
                f"注意：eledefects.py 通常只生成 lammps 输入文件；如果还没运行 LAMMPS，就不会有 dump。"
            )
        )

    L.info(f"找到结构目录: {len(structure_dirs)} 个")
    for i, d in enumerate(structure_dirs, 1):
        L.item(i, len(structure_dirs), d)

    # ============================================================
    # 2. 每个结构目录单独 dump -> fdf
    # ============================================================
    for struct_dir in structure_dirs:
        outroot = struct_dir / "dpnegf"

        if md_steps == 0:
            poscar_dir = struct_dir / "lammps" if (struct_dir / "lammps" / "POSCAR").is_file() else struct_dir
            if not (poscar_dir / "POSCAR").is_file():
                raise FileNotFoundError(f"md_steps=0 时找不到 POSCAR: {poscar_dir / 'POSCAR'}")
            if not args.dry_run and shutil.which("sgeom") is None:
                raise FileNotFoundError("md_steps=0 需要 sgeom 命令，但当前环境中未找到")
            run_cmd(["sgeom", "POSCAR", "STRUCT.fdf"], dry_run=args.dry_run, cwd=poscar_dir)
            if not args.dry_run:
                outroot.mkdir(parents=True, exist_ok=True)
                shutil.copy2(poscar_dir / "STRUCT.fdf", outroot / "STRUCT.fdf")
            continue

        lammps_dir = struct_dir / "lammps"

        if lammps_dir.exists():
            dump_root = lammps_dir
        else:
            dump_root = struct_dir

        # 每条独立 LAMMPS 轨迹只取最后一帧。
        # md_steps 必须与 in.lammps 中的 run <N> 一致。
        L.info(
            f"单轨迹末帧采样: timestep={md_steps}, samples=1"
        )

        cmd = [
            py,
            str(dump2fdf),
            "--root",
            str(dump_root),
            "--outroot",
            str(outroot),
            "--every",
            str(md_steps),
            "--samples",
            "1",
        ]
        run_cmd(cmd, dry_run=args.dry_run, env=env)

    # ============================================================
    # 3. 对每个结构的 dpnegf 目录做 fdf -> xyz
    # ============================================================
    L.phase(4, 7, "fdf -> xyz")
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
    L.phase(5, 7, "复制 DPNEGF 输入")
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
        L.phase(6, 7, "运行 DPNEGF")
        dpnegf_workdirs = find_dpnegf_workdirs_from_structures(structure_dirs)

        if not dpnegf_workdirs:
            raise RuntimeError(
                "没有找到 DPNEGF 工作目录。请确认每个 leaf 目录下存在：\n"
                "  input.json / run.py / run.sh / *.xyz / *.pth\n"
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
                    "--scheduler",
                    args.scheduler,
                ],
                dry_run=args.dry_run,
                env=env,
            )

        # SLURM 屏障：等所有 DPNEGF 作业跑完并检查 flag，再统一清理缓存。
        # local 模式下每个 sub_dpnegf 已阻塞跑完，可直接进入清理。
        if args.scheduler == "slurm":
            barrier_and_check(
                dpnegf_workdirs, "dpnegf",
                poll_interval=args.poll_interval,
                dry_run=args.dry_run,
            )

        # 全部 DPNEGF 跑完后，对整棵 root 树统一清理一次缓存。
        # 注意：不要放在 for 循环内，否则 O(N^2) 全树遍历，
        # 且并行化后会误删其它 workdir 尚在使用的 self_energy / HS_*.h5。
        # slurm 屏障保证此刻所有作业都已结束，清理是安全的。
        if not args.dry_run:
            save_self_energy = config.get("save_self_energy", False)
            if not isinstance(save_self_energy, bool):
                raise ValueError("save_self_energy must be a boolean")
            # P2-6：读取配置中的自能缓存路径，确保清理路径与实际写入路径一致。
            se_cache = config.get("self_energy_cache", {})
            se_save_path = None
            if isinstance(se_cache, dict):
                se_save_path = se_cache.get("save_path")
            clean_dpnegf_output_cache(
                root,
                save_self_energy=save_self_energy,
                self_energy_save_path=se_save_path,
            )

            # 仅在本次发现的全部 DPNEGF 工作目录均成功后才删除对应 MD
            # 轨迹；任一 DPNEGF 失败会在屏障/本地运行处抛错并跳过这里。
            for struct_dir in structure_dirs:
                lammps_dir = struct_dir / "lammps"
                removed = remove_lammps_dump_files(lammps_dir)
                L.clean(f"DPNEGF 成功后清理 LAMMPS dump: {removed} 个 ({lammps_dir})")

        # ============================================================
        # 6. 收集本次 MD 对应的 DPNEGF 电导日志
        # ============================================================
        if not args.skip_conductance:
            L.phase(7, 7, "收集电导日志")

            conductance_python = shutil.which(args.conductance_python or py)
            if not args.dry_run and conductance_python is None:
                raise FileNotFoundError(
                    f"找不到电导收集 Python: {args.conductance_python or py}"
                )

            # 注意：collect_md_conductance.py 只接受 --root /
            # --configuration-dir / --configuration-settings，不接受
            # --config / --structure-dir（后者会让 argparse 报
            # ambiguous option 而在昂贵计算结束后失败）。
            # 电导值（Ef/Ec/Ev）与输运温度由收集器直接从每个 replica 的
            # negf.out.pth 读取（由 DPNEGF 预计算），无需在此传入外部能量
            # 参考或温度。
            cmd = build_collect_conductance_cmd(
                python=conductance_python or (args.conductance_python or py),
                script=collect_conductance_script,
                root=root,
                configuration_dirs=structure_dirs,
            )

            run_cmd(cmd, dry_run=args.dry_run, env=env)


if __name__ == "__main__":
    main()
