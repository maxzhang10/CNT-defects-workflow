#!/usr/bin/env python3

import argparse
import copy
import importlib
import json
import math
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHIRAL_CONFIGS = [
    {
        "m": 5,
        "n": 5,
        "l_def": 8,
        "N_defects": 2,
        "structures": ["5775"],
        # 半导体带边电导：省略时保持原有 E_F=0 的电导计算。
        "semi": True,
        "energy_window": [-0.25, 0.25],
        # 有限偏压带边电导示例：启用时由外部 Ec/Ev 定义两个计算中心。
        # "conductance_mode": "band_edge_bias",
        # "Ec_eV": 0.32,
        # "Ev_eV": -0.28,
        # "bias_eV": 0.01,
        # "fermi_difference_threshold": 1e-6,
        # DPNEGF 透射谱能量网格步长（eV）。
        "espacing": 0.1,
    }
]


TEMPERATURES = [500]


# 每个物理配置独立运行的 LAMMPS 轨迹数。
# 可由命令行 --lammps-repeats 覆盖。
DEFAULT_LAMMPS_REPEATS = 2

# 两类随机种子分开管理：
# seed 控制缺陷结构；同一物理配置的多个 replica 共用同一个 seed。
# lammps_seed 控制热运动轨迹；每个 replica 使用不同的 seed。
BASE_STRUCTURE_SEED = 20260705
BASE_LAMMPS_SEED = 23456789


TEMPLATE = {
    "temperature": 300,
    "chirality": [5, 5],
    "r_max": 6.50,
    "data_root": (
        "/public5/home/t6s008517/dpnegf/workflow_2/5_5/"
        
    ),
    "structures": ["5775"],
    "N_defects": 3,
    "l_def": 5,
    "md_steps": 50000,
    # 每条独立 LAMMPS 轨迹只取最后一帧做 DPNEGF。
    "md_sampling": {
        "n_samples": 1,
    },
}


def load_cnt_geometry(workflow_dir):
    """
    从 workflow/stage 目录导入 cnt_geometry。

    batch 脚本可以放在 workflow 根目录，
    不要求当前工作目录正好是 stage。
    """
    stage_dir = workflow_dir / "stage"
    module_path = stage_dir / "cnt_geometry.py"

    if not module_path.is_file():
        raise FileNotFoundError(
            f"找不到 cnt_geometry.py: {module_path}"
        )

    stage_dir_text = str(stage_dir)

    if stage_dir_text not in sys.path:
        sys.path.insert(0, stage_dir_text)

    return importlib.import_module("cnt_geometry")


def make_folder_name(
    structures,
    l_def,
    density,
):
    """
    生成与 ele_multi_defects_ele.py 完全一致的目录名。

    示例：
        5775_L008_0.1016A-1
    """
    if not structures:
        raise ValueError("structures 不能为空")

    type_name = "_".join(
        str(item)
        for item in structures
    )

    return (
        f"{type_name}"
        f"_L{int(l_def):03d}"
        f"_{float(density):.4f}A-1"
    )


def build_tasks(
    base_root,
    cnt_geometry,
    lammps_repeats,
):
    """
    将每个物理配置展开为 lammps_repeats 条独立 LAMMPS 轨迹。

    同一物理配置的 replica：
    - 使用独立的缺陷结构 seed（每个 replica 缺陷位置不同）；
    - 使用独立的 lammps_seed（每个 replica 热运动轨迹不同）；
    - 写入同一配置目录下相互独立的 replica_XXX 子目录。
    """
    tasks = []
    used_roots = {}
    physical_index = 0
    task_index = 0

    for cfg in CHIRAL_CONFIGS:
        conductance_mode = cfg.get("conductance_mode", "fermi")
        if conductance_mode not in {"fermi", "band_edge_bias"}:
            raise ValueError("conductance_mode must be 'fermi' or 'band_edge_bias'")
        band_edge_settings = {}
        if conductance_mode == "band_edge_bias":
            try:
                ec = float(cfg["Ec_eV"])
                ev = float(cfg["Ev_eV"])
                bias = float(cfg["bias_eV"])
                threshold = float(cfg["fermi_difference_threshold"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "band_edge_bias requires numeric Ec_eV, Ev_eV, bias_eV, "
                    "and fermi_difference_threshold"
                ) from exc
            if not all(math.isfinite(value) for value in (ec, ev, bias, threshold)):
                raise ValueError("band_edge_bias settings must be finite")
            if ev >= ec or bias <= 0.0 or not 0.0 < threshold < 1.0:
                raise ValueError("band_edge_bias requires Ev_eV < Ec_eV, bias_eV > 0, and 0 < threshold < 1")
            band_edge_settings = {
                "Ec_eV": ec, "Ev_eV": ev, "bias_eV": bias,
                "fermi_difference_threshold": threshold,
            }
        semi = bool(cfg.get("semi", False))
        try:
            espacing = float(cfg.get("espacing", 0.1))
        except (TypeError, ValueError) as exc:
            raise ValueError("espacing must be a positive numeric value in eV") from exc
        if not math.isfinite(espacing) or espacing <= 0.0:
            raise ValueError("espacing must be a finite positive value in eV")
        energy_window = None
        if semi:
            raw_window = cfg.get("energy_window")
            if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:
                raise ValueError(
                    "semi=True requires energy_window=[relative_emin, relative_emax]"
                )
            try:
                energy_window = [float(raw_window[0]), float(raw_window[1])]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "semi=True energy_window must contain two numeric values"
                ) from exc
            if not all(math.isfinite(value) for value in energy_window):
                raise ValueError("semi=True energy_window values must be finite")
            if energy_window[0] >= energy_window[1]:
                raise ValueError(
                    "semi=True requires energy_window[0] < energy_window[1]"
                )
        for temperature in TEMPERATURES:
            physical_index += 1

            m = int(cfg["m"])
            n = int(cfg["n"])
            l_def = int(cfg["l_def"])
            n_defects = int(cfg["N_defects"])
            structures = list(cfg["structures"])

            # 必须使用与结构生成脚本相同的 geo_info，
            # 确保这里算出的密度和目录名完全一致。
            T, N_uc, l_PL, length = cnt_geometry.geo_info(
                m,
                n,
                float(TEMPLATE["r_max"]),
                l_def,
            )

            density = n_defects / (l_def * T)

            folder_name = make_folder_name(
                structures=structures,
                l_def=l_def,
                density=density,
            )

            # 每个 replica 使用独立的缺陷结构 seed，
            # 实现缺陷位置的随机采样。
            for replica in range(1, lammps_repeats + 1):
                task_index += 1
                # structure_seed: 控制缺陷位置和类型，每个 replica 独立
                structure_seed = BASE_STRUCTURE_SEED + task_index
                # lammps_seed: 控制 LAMMPS 热运动轨迹，每个 replica 独立
                lammps_seed = BASE_LAMMPS_SEED + task_index

                # 目录层级：
                # data_root/温度/手性/缺陷配置/replica_XXX
                # 每个 replica 内部拥有独立的 lammps、dpnegf 和 flag。
                configuration_root = (
                    base_root
                    / f"{int(temperature)}K"
                    / f"{m}_{n}"
                    / folder_name
                )

                run_root = (
                    configuration_root
                    / f"replica_{replica:03d}"
                )

                task_id = (
                    f"{int(temperature)}K_"
                    f"{m}_{n}_"
                    f"{folder_name}_"
                    f"rep{replica:03d}"
                )

                root_key = str(run_root.resolve())

                if root_key in used_roots:
                    previous_task = used_roots[root_key]

                    raise ValueError(
                        "检测到两个任务对应同一个目录：\n"
                        f"  任务1: {previous_task}\n"
                        f"  任务2: {task_id}\n"
                        f"  目录:  {run_root}\n"
                        "请检查 l_def、N_defects、structures 和 replica。"
                    )

                used_roots[root_key] = task_id

                tasks.append({
                    "task_id": task_id,
                    "temperature": int(temperature),
                    "m": m,
                    "n": n,
                    "l_def": l_def,
                    "N_defects": n_defects,
                    "structures": structures,
                    "density": density,
                    "folder_name": folder_name,
                    "replica": replica,
                    "lammps_repeats": lammps_repeats,
                    "data_root": base_root,
                    "configuration_root": configuration_root,
                    "run_root": run_root,
                    "T": float(T),
                    "N_uc": int(N_uc),
                    "l_PL": int(l_PL),
                    "length": int(length),
                    "seed": structure_seed,
                    "lammps_seed": lammps_seed,
                    "semi": semi,
                    "energy_window": energy_window,
                    "espacing": espacing,
                    "conductance_mode": conductance_mode,
                    **band_edge_settings,
                })

    return tasks


def make_task_config(task):
    """
    创建当前 replica 的独立配置。

    data_root 保留整个批次的数据根目录；
    structure_root 和 run_multi.py 的 --root 都指向具体 replica 目录。
    """
    config = copy.deepcopy(TEMPLATE)

    config.update({
        "temperature": task["temperature"],
        "chirality": [
            task["m"],
            task["n"],
        ],
        "l_def": task["l_def"],
        "N_defects": task["N_defects"],
        "structures": task["structures"],
        "data_root": str(task["data_root"]),
        "structure_root": str(task["run_root"]),
        "seed": task["seed"],
        "lammps_seed": task["lammps_seed"],
        "lammps_replica": task["replica"],
        "lammps_repeats": task["lammps_repeats"],
        "density_A_inv": task["density"],
        "configuration_root": str(task["configuration_root"]),
        "md_sampling": {"n_samples": 1},
        "semi": task["semi"],
        "energy_window": task["energy_window"],
        "espacing": task["espacing"],
        "conductance_mode": task["conductance_mode"],
        "Ec_eV": task.get("Ec_eV"),
        "Ev_eV": task.get("Ev_eV"),
        "bias_eV": task.get("bias_eV"),
        "fermi_difference_threshold": task.get("fermi_difference_threshold"),
    })

    return config


# 必须保留的输入文件模式（相对于 run_root）
REQUIRED_INPUT_FILES = [
    "workflow_config.json",
    "*.in",           # LAMMPS 输入文件
    "*.data",         # LAMMPS 数据文件
    "*.xyz",          # 结构文件
    "POSCAR",         # VASP 输入
    "input.json",     # DPNEGF 输入配置
]


def cleanup_failed_directory(run_root):
    """
    清理失败目录中的输出文件和哨兵文件，只保留必须的输入文件。

    删除的内容包括：
    - 日志文件 (*.log, *.out, *.err)
    - 结果文件 (*.csv, *.txt, conductance_*)
    - SLURM 输出 (*.slurm-out, slurm-*.out)
    - 临时目录 (lammps/, dpnegf/, tmp/)
    - 完成标记文件 (*.done, *.flag, SUCCESS, FAILED)
    """
    if not run_root.exists():
        return

    # 要删除的文件模式
    output_patterns = [
        "*.log",
        "*.out",
        "*.err",
        "*.csv",
        "*.txt",
        "*.done",
        "*.flag",
        "*.slurm-out",
        "slurm-*.out",
        "conductance_*",
        "SUCCESS",
        "FAILED",
        "*.traj",
        "*.dump",
        "*.restart",
    ]

    # 要删除的目录
    dir_names = [
        "lammps",
        "dpnegf",
        "tmp",
        "__pycache__",
    ]

    deleted_count = 0

    # 删除匹配模式的文件
    for pattern in output_patterns:
        for file_path in run_root.glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    pass

    # 删除特定目录
    for dir_name in dir_names:
        dir_path = run_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                import shutil
                shutil.rmtree(dir_path)
                deleted_count += 1
            except OSError:
                pass

    print(f"        [CLEANUP] 清理完成，删除 {deleted_count} 项")


def run_single_task(
    task,
    scheduler,
    workflow_dir,
):
    task_id = task["task_id"]
    run_root = task["run_root"]

    config = make_task_config(task)

    config_path = None
    start = time.time()

    try:
        # 提前创建独立根目录。
        # 之后 ele_multi_defects_ele.py 会在其中生成 lammps/。
        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 永久保存实际运行配置。
        permanent_config = (
            run_root
            / "workflow_config.json"
        )

        with permanent_config.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False,
            )

        # 每个任务使用唯一临时配置文件，
        # 避免多个 run_multi.py 并发读取同一个 config。
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"cnt_{task_id}_",
            delete=False,
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False,
            )

            config_path = Path(file.name)

        cmd = [
            sys.executable,
            str(workflow_dir / "run_multi.py"),
            "--config",
            str(config_path),
            "--root",
            str(run_root),
            "--scheduler",
            scheduler,
            # 批量模式下不在每个 replica 内单独收集；
            # 等全部 workflow 成功后统一跨 replica 汇总。
            "--skip-conductance",
        ]

        print(
            f"[START] {task_id}",
            flush=True,
        )

        print(
            f"        root    = {run_root}",
            flush=True,
        )

        print(
            f"        l_def   = {task['l_def']}",
            flush=True,
        )

        print(
            f"        defects = {task['N_defects']}",
            flush=True,
        )

        print(
            f"        density = "
            f"{task['density']:.6f} A^-1",
            flush=True,
        )

        print(
            f"        replica = "
            f"{task['replica']}/{task['lammps_repeats']}",
            flush=True,
        )

        print(
            f"        seeds   = "
            f"structure:{task['seed']} "
            f"lammps:{task['lammps_seed']}",
            flush=True,
        )

        result = subprocess.run(
            cmd,
            cwd=workflow_dir,
            text=True,
            check=False,
        )

        elapsed = time.time() - start

        status = (
            "OK"
            if result.returncode == 0
            else "FAIL"
        )

        print(
            f"[{status}] {task_id} "
            f"exit={result.returncode}, "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

        return (
            task_id,
            result.returncode,
        )

    except Exception as exc:
        elapsed = time.time() - start

        print(
            f"[ERROR] {task_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        print(
            f"[FAIL] {task_id} "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

        return (
            task_id,
            1,
        )

    finally:
        if config_path is not None:
            config_path.unlink(
                missing_ok=True
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "并行运行不同长度的 "
            "LAMMPS + DPNEGF 工作流"
        )
    )

    parser.add_argument(
        "--scheduler",
        choices=[
            "local",
            "slurm",
        ],
        default="slurm",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=50,
        help=(
            "同时启动的独立 workflow 数。默认不限制，"
            "即一次启动全部展开任务；例如 3 个配置 × 5 次重复 = 15。"
            "需要限制提交并发时再显式指定该参数。"
        ),
    )

    parser.add_argument(
        "--lammps-repeats",
        type=int,
        default=DEFAULT_LAMMPS_REPEATS,
        help=(
            "每个温度/手性/缺陷配置独立运行的 LAMMPS 次数。"
            "每次使用不同 lammps_seed，并且只取最后一帧做 DPNEGF。"
        ),
    )

    parser.add_argument(
        "--workflow-dir",
        default=str(Path(__file__).resolve().parent),
        help=(
            "工作流代码目录，默认使用 batch_generate.py 所在目录。"
            "这样从 workflow_2 运行时会自动使用 workflow_2/run_multi.py "
            "及 workflow_2/stage，而不会误调用其他工作流副本。"
        ),
    )

    parser.add_argument(
        "--data-root",
        default=TEMPLATE["data_root"],
    )

    parser.add_argument(
        "--e-fermi",
        type=float,
        default=0.0,
        help="汇总电导时使用的费米能，默认 0 eV。",
    )

    parser.add_argument(
        "--transport-temperature-k",
        type=float,
        default=TEMPERATURES[0],
        help=(
            "汇总电导时使用的输运温度(K)。"
            "默认使用 TEMPERATURES[0]；可通过该参数显式覆盖。"
        ),
    )

    parser.add_argument(
        "--skip-conductance",
        action="store_true",
        help="跳过批次结束后的跨 replica 电导汇总。",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="任务失败后的最大重试次数，默认 2 次。",
    )

    args = parser.parse_args()

    if args.max_parallel is not None and args.max_parallel < 1:
        raise ValueError(
            "--max-parallel 必须大于 0"
        )

    if args.lammps_repeats < 1:
        raise ValueError(
            "--lammps-repeats 必须大于 0"
        )

    if (
        args.transport_temperature_k is not None
        and args.transport_temperature_k < 0.0
    ):
        raise ValueError(
            "--transport-temperature-k 必须大于等于 0"
        )

    workflow_dir = Path(
        args.workflow_dir
    ).resolve()

    base_root = Path(
        args.data_root
    ).resolve()

    run_multi_path = (
        workflow_dir
        / "run_multi.py"
    )

    if not run_multi_path.is_file():
        raise FileNotFoundError(
            f"找不到 {run_multi_path}"
        )

    collect_script = (
        workflow_dir
        / "stage"
        / "collect_md_conductance.py"
    )

    if not args.skip_conductance and not collect_script.is_file():
        raise FileNotFoundError(
            f"找不到跨 replica 电导收集脚本: {collect_script}"
        )

    cnt_geometry = load_cnt_geometry(
        workflow_dir
    )

    tasks = build_tasks(
        base_root=base_root,
        cnt_geometry=cnt_geometry,
        lammps_repeats=args.lammps_repeats,
    )

    if not tasks:
        print("[INFO] 没有需要运行的任务")
        return 0

    if args.max_parallel is None:
        max_workers = len(tasks)
    else:
        max_workers = min(
            args.max_parallel,
            len(tasks),
        )

    print("=" * 70)
    print(f"[INFO] 总任务数:       {len(tasks)}")
    print(f"[INFO] 每配置轨迹数:   {args.lammps_repeats}")
    print(f"[INFO] 独立任务目录数: {len(tasks)}")
    print(f"[INFO] 并行 workflow:  {max_workers}")
    print(f"[INFO] scheduler:      {args.scheduler}")
    print(f"[INFO] workflow_dir:   {workflow_dir}")
    print(f"[INFO] data_root:      {base_root}")

    print("[INFO] 任务目录：")

    for task in tasks:
        print(
            f"  {task['task_id']}\n"
            f"    -> {task['run_root']}\n"
            f"       lammps_seed={task['lammps_seed']}"
        )

    print("=" * 70)

    all_results = {}
    retry_counts = {}
    start = time.time()

    # 准备任务队列，支持重试
    pending_tasks = list(tasks)
    completed_tasks = set()

    while pending_tasks:
        # 每个 task 都有独立 run_root，
        # 因此不再按温度/手性分组串行。
        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_task = {
                executor.submit(
                    run_single_task,
                    task,
                    args.scheduler,
                    workflow_dir,
                ): task
                for task in pending_tasks
            }

            # 清空待处理列表，准备收集下一轮（重试）任务
            pending_tasks = []

            for future in as_completed(
                future_to_task
            ):
                task = future_to_task[future]
                task_id = task["task_id"]

                try:
                    result = future.result()
                    returncode = result[1]
                except Exception as exc:
                    print(
                        f"[ERROR] 未捕获异常 "
                        f"{task_id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    returncode = 1

                # 检查任务是否成功
                if returncode == 0:
                    all_results[task_id] = (task_id, 0)
                    completed_tasks.add(task_id)
                else:
                    # 任务失败，检查是否需要重试
                    current_retries = retry_counts.get(task_id, 0)

                    if current_retries < args.max_retries:
                        retry_counts[task_id] = current_retries + 1

                        print(
                            f"[RETRY] {task_id} "
                            f"第 {retry_counts[task_id]}/{args.max_retries} 次重试",
                            flush=True,
                        )

                        # 清理失败目录
                        cleanup_failed_directory(task["run_root"])

                        # 将任务加入下一轮重试队列
                        pending_tasks.append(task)
                    else:
                        # 重试次数已用完
                        print(
                            f"[MAX_RETRY] {task_id} "
                            f"已达到最大重试次数 {args.max_retries}，放弃",
                            flush=True,
                        )
                        all_results[task_id] = (task_id, returncode)
                        completed_tasks.add(task_id)

    # 转换结果为列表格式
    all_results_list = list(all_results.values())

    failed = [
        task_id
        for task_id, returncode in all_results_list
        if returncode != 0
    ]

    elapsed = time.time() - start

    print("\n" + "=" * 70)

    print(
        f"成功: "
        f"{len(all_results_list) - len(failed)}"
        f"/{len(all_results_list)}"
    )

    print(
        f"失败: {len(failed)}"
    )

    if retry_counts:
        total_retries = sum(retry_counts.values())
        print(f"总重试次数: {total_retries}")

    print(
        f"总耗时: "
        f"{elapsed / 60:.1f} min"
    )

    if failed:
        print("失败任务：")

        for task_id in failed:
            print(f"  {task_id}")

        return 1

    if not args.skip_conductance:
        print("\n" + "=" * 70)
        print("[INFO] 全部 workflow 成功，开始跨 replica 汇总电导")

        collect_cmd = [
            sys.executable,
            str(collect_script),
            "--root",
            str(base_root),
            "--e-fermi",
            str(args.e_fermi),
            "--expected-replicas",
            str(args.lammps_repeats),
        ]

        if args.transport_temperature_k is not None:
            collect_cmd.extend(
                [
                    "--transport-temperature-k",
                    str(args.transport_temperature_k),
                ]
            )

        # 只汇总本次 batch 展开的物理配置，避免把 data_root 下
        # 其他历史配置或测试目录混入本次总表。
        configuration_roots = sorted(
            {
                Path(task["configuration_root"]).resolve()
                for task in tasks
            }
        )
        for configuration_root in configuration_roots:
            collect_cmd.extend(
                ["--configuration-dir", str(configuration_root)]
            )
            matching_tasks = [
                task for task in tasks
                if Path(task["configuration_root"]).resolve() == configuration_root
            ]
            first_task = matching_tasks[0]
            if any(
                task["semi"] != first_task["semi"]
                or task["energy_window"] != first_task["energy_window"]
                or task["conductance_mode"] != first_task["conductance_mode"]
                or task["temperature"] != first_task["temperature"]
                for task in matching_tasks
            ):
                raise ValueError(
                    f"configuration {configuration_root} has inconsistent "
                    "semi/energy_window/temperature task settings"
                )
            collect_cmd.extend(
                [
                    "--configuration-settings",
                    json.dumps(
                        {
                            "configuration_dir": str(configuration_root),
                            "semi": first_task["semi"],
                            "energy_window": first_task["energy_window"],
                            "temperature": first_task["temperature"],
                            "conductance_mode": first_task["conductance_mode"],
                            "Ec_eV": first_task.get("Ec_eV"),
                            "Ev_eV": first_task.get("Ev_eV"),
                            "bias_eV": first_task.get("bias_eV"),
                            "fermi_difference_threshold": first_task.get("fermi_difference_threshold"),
                        }
                    ),
                ]
            )

        collect_result = subprocess.run(
            collect_cmd,
            cwd=workflow_dir,
            text=True,
            check=False,
        )

        if collect_result.returncode != 0:
            print(
                f"[FAIL] 电导汇总失败，exit={collect_result.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return collect_result.returncode

        print("[OK] 电导汇总完成", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
