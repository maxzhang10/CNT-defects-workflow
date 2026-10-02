#!/usr/bin/env python3

import argparse
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import threading
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
        # 电导模式决定写入 input.json 的 conductance_options.mu 标签：
        #   fermi           -> ["Ef"]      （以费米能级为化学势）
        #   band_edge_bias  -> ["Ev", "Ec"]（以带边为化学势）
        # Ec/Ev/Ef 的实际能量值不再由外部提供，而是由 DPNEGF 内部计算
        # 并随 negf.out.pth 的 conductance 列表输出，由收集器按标签读取。
        "conductance_mode": "band_edge_bias",
        # 能量网格（energy_grid: clenshaw_curtis/num_points/half_width）完全
        # 沿用 input.json 模板默认值，不再由工作流外置。
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

# LAMMPS 模式，全局作用于整个批次：
#   "md"  —— NVT 热退火，从 config 读取温度/md_steps/lammps_seed；
#   "opt" —— 只做几何优化，LAMMPS 输入改用 input_files/lammps/opt.lammps
#            （拷贝到工作目录时命名为 in.lammps），只改写固定原子数 nfix，
#            不消费温度/步数/种子。
LAMMPS_MODE = "opt"


class ProgressRecorder:
    """
    记录每个物理配置的 replica 任务进度，追加写入 record.log。

    每行格式：
        [时间] 配置相对路径: 完成数/总数 (当前 task_id OK/FAIL)

    计数按 task_id 去重，任务无论 OK/FAIL 都计入"已完成"，
    表示该 replica 的 workflow 已经跑完一轮、不会再被本批次重试。
    """

    def __init__(self, log_path, total_by_config):
        self._log_path = log_path
        self._total_by_config = total_by_config
        self._done_by_config = {
            config: set() for config in total_by_config
        }
        self._lock = threading.Lock()

    def record(self, config_name, task_id, status):
        with self._lock:
            done = self._done_by_config[config_name]
            done.add(task_id)
            total = self._total_by_config[config_name]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"[{timestamp}] {config_name}: "
                f"{len(done)}/{total} done "
                f"({task_id} {status})\n"
            )
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line)

TEMPLATE = {
    "temperature": 300,
    "chirality": [5, 5],
    "r_max": 6.50,
    "data_root": (
        "./"
        
    ),
    "structures": ["5775"],
    "N_defects": 3,
    "l_def": 5,
    "md_steps": 50000,
    # 每条独立 LAMMPS 轨迹只取最后一帧做 DPNEGF。
    "md_sampling": {
        "n_samples": 1,
    },
    # 能量网格（energy_grid: clenshaw_curtis/num_points/half_width）完全沿用
    # input.json 模板默认值，不再由工作流外置 espacing/negf_energy_window。
    # False 保持既有行为：DPNEGF 成功后清理 output/self_energy。
    "save_self_energy": False,
    # 覆盖 DPNEGF input.json 的 self_energy_options.cache 配置。
    "self_energy_cache": {
        "use_saved": True,
        "save_path": "./self_energy/",
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
        save_self_energy = cfg.get(
            "save_self_energy", TEMPLATE["save_self_energy"]
        )
        if not isinstance(save_self_energy, bool):
            raise ValueError("save_self_energy must be a boolean")
        # 能量网格（energy_grid）完全沿用 input.json 模板默认值，
        # 不再从 cfg/TEMPLATE 读取 espacing/negf_energy_window。
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

                # record.log 里按此名称聚合 replica 进度
                config_name = (
                    f"{int(temperature)}K/"
                    f"{m}_{n}/"
                    f"{folder_name}"
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
                    "config_name": config_name,
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
                    "save_self_energy": save_self_energy,
                    "conductance_mode": conductance_mode,
                })

    return tasks


def make_task_config(task):
    """
    创建当前 replica 的独立配置。

    data_root 保留整个批次的数据根目录；
    structure_root 和 run_multi.py 的 --root 都指向具体 replica 目录。
    """
    config = copy.deepcopy(TEMPLATE)

    # LAMMPS 模式全局唯一：模块级 LAMMPS_MODE（缺省 "md"）对整个批次生效，
    # 写入每个 replica 的 config，不在 CHIRAL_CONFIGS 单独覆盖。
    lammps_mode = LAMMPS_MODE
    if lammps_mode not in ("md", "opt"):
        raise ValueError(
            f"lammps_mode 必须为 'md' 或 'opt'，当前值: {lammps_mode}"
        )

    # conductance_options.mu 只写入字符串标签（非数值）：
    #   fermi          -> ["Ef"]
    #   band_edge_bias -> ["Ev", "Ec"]
    # Ec/Ev/Ef 的实际能量值由 DPNEGF 计算后随 negf.out.pth 输出，
    # 由收集器按标签从结果文件读取，不再由外部提供。
    mu_labels = (
        ["Ev", "Ec"]
        if task["conductance_mode"] == "band_edge_bias"
        else ["Ef"]
    )

    config.update({
        "lammps_mode": lammps_mode,
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
        "save_self_energy": task["save_self_energy"],
        "conductance_mode": task["conductance_mode"],
        "conductance_options": {
            "mu": mu_labels,
        },
    })

    return config


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
        # P0-4：在覆盖 workflow_config.json 之前，先校验目录中已有完成结果
        # 的 provenance。只有旧结果的 provenance hash 与当前 config hash 完全
        # 一致才允许复用；不一致或无法确认时 fail-safe 报错，禁止覆盖旧配置
        # 并禁止复用旧结果。
        stage_dir = workflow_dir / "stage"
        if str(stage_dir) not in sys.path:
            sys.path.insert(0, str(stage_dir))
        negf_provenance = importlib.import_module("negf_provenance")
        new_hash = negf_provenance.compute_negf_config_hash(config)

        if run_root.is_dir():
            mismatches = []
            checked = 0
            for done_flag in sorted(run_root.rglob("dpnegf_done.flag")):
                leaf = done_flag.parent
                checked += 1
                result = negf_provenance.result_file_path(leaf)
                saved_hash = negf_provenance.read_hash_file(
                    negf_provenance.provenance_hash_path(leaf)
                )
                if not result.is_file() or result.stat().st_size == 0:
                    mismatches.append(
                        f"{leaf}: done flag 存在但结果文件缺失或为空 ({result})"
                    )
                elif saved_hash is None:
                    mismatches.append(
                        f"{leaf}: 旧结果缺少 provenance hash，无法确认配置一致性"
                    )
                elif saved_hash != new_hash:
                    mismatches.append(
                        f"{leaf}: 配置已变更 (provenance={saved_hash[:12]}... "
                        f"!= 当前={new_hash[:12]}...)"
                    )
            if mismatches:
                details = "\n".join(f"  - {m}" for m in mismatches)
                raise RuntimeError(
                    f"检测到 {checked} 个已完成 DPNEGF 结果与新配置不一致，"
                    f"禁止覆盖旧配置或复用旧结果 (P0-4):\n{details}\n"
                    "请使用新目录运行，或删除对应 dpnegf_done.flag 与 output/ "
                    "后显式重新计算。"
                )

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
            f"[START] {task_id}  "
            f"density={task['density']:.4f}A^-1  "
            f"replica={task['replica']}/{task['lammps_repeats']}",
            flush=True,
        )

        # 隔离 run_multi.py 的详细输出：每个 replica 的完整内部日志
        # （含全部 INFO 与 traceback）写入其自身目录的 workflow.log，
        # 不再混入 batch 主日志，便于失败时排查。
        # 以写入模式打开，覆盖已有同名 workflow.log（无需显式删除）。
        workflow_log = run_root / "workflow.log"
        with workflow_log.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            result = subprocess.run(
                cmd,
                cwd=workflow_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
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

        if result.returncode != 0:
            print(
                f"[FAIL] {task_id}\n"
                f"        dir        = {run_root.resolve()}\n"
                f"        exit       = {result.returncode}\n"
                f"        detail_log = {workflow_log.resolve()}\n"
                f"        (目录已保留，未自动清理或重试)",
                flush=True,
            )

        return (
            task_id,
            result.returncode,
            status,
        )

    except Exception as exc:
        elapsed = time.time() - start

        print(
            f"[FAIL] {task_id}\n"
            f"        dir   = {run_root.resolve()}\n"
            f"        error = {type(exc).__name__}: {exc}\n"
            f"        (目录已保留，未自动清理或重试)",
            file=sys.stderr,
            flush=True,
        )
        # 注意：此处异常发生在 subprocess.run 之前（目录已建/未建），
        # run_multi.py 未启动，没有 workflow.log 可指。

        return (
            task_id,
            1,
            "FAIL",
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
        "--skip-conductance",
        action="store_true",
        help="跳过批次结束后的跨 replica 电导汇总。",
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

    # 导入 NEGF provenance 工具（与 stage 共享同一 config hash 定义）。
    negf_provenance = importlib.import_module("negf_provenance")

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
    print(f"[INFO] 总任务数: {len(tasks)}  "
          f"每配置 replica 数: {args.lammps_repeats}  "
          f"并发: {max_workers}  scheduler: {args.scheduler}")
    print(f"[INFO] data_root: {base_root}")
    print("=" * 70)

    all_results = {}
    start = time.time()

    # 每个物理配置的 replica 总数，用于 record.log 进度计数
    total_by_config = {}
    for task in tasks:
        total_by_config[task["config_name"]] = (
            total_by_config.get(task["config_name"], 0) + 1
        )

    record_log = base_root / "record.log"
    recorder = ProgressRecorder(record_log, total_by_config)
    with record_log.open("a", encoding="utf-8") as f:
        f.write(
            f"\n===== batch start {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({len(tasks)} tasks, "
            f"{len(total_by_config)} configurations) =====\n"
        )

    print(f"[INFO] record.log: {record_log}")

    # 每个 task 都有独立 run_root，并发执行所有任务。
    # 单个任务失败不中断整个批次：保留其目录文件，记录失败信息，
    # 由调用方事后检查并决定是否手动重跑。
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
            for task in tasks
        }

        for future in as_completed(
            future_to_task
        ):
            task = future_to_task[future]
            task_id = task["task_id"]

            try:
                result = future.result()
                returncode = result[1]
                status = result[2]
            except Exception as exc:
                print(
                    f"[ERROR] 未捕获异常 "
                    f"{task_id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                returncode = 1
                status = "FAIL"

            recorder.record(
                task["config_name"],
                task_id,
                status,
            )

            all_results[task_id] = (task_id, returncode)

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

    print(
        f"总耗时: "
        f"{elapsed / 60:.1f} min"
    )

    # 批次结束时在 record.log 写一行每个配置的最终进度汇总
    with record_log.open("a", encoding="utf-8") as f:
        f.write(
            f"----- batch end {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(success {len(all_results_list) - len(failed)}"
            f"/{len(all_results_list)}) -----\n"
        )
        for config_name in sorted(total_by_config):
            done = len(recorder._done_by_config[config_name])
            f.write(
                f"  {config_name}: "
                f"{done}/{total_by_config[config_name]} done\n"
            )

    if failed:
        print("失败任务：")

        for task_id in failed:
            task = next(
                t for t in tasks
                if t["task_id"] == task_id
            )
            print(
                f"  {task_id}\n"
                f"    -> {task['run_root'].resolve()}"
            )

        return 1

    if not args.skip_conductance:
        print("\n" + "=" * 70)
        print("[INFO] 全部 workflow 成功，开始跨 replica 汇总电导")

        collect_cmd = [
            sys.executable,
            str(collect_script),
            "--root",
            str(base_root),
            "--expected-replicas",
            str(args.lammps_repeats),
        ]

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
                task["conductance_mode"] != first_task["conductance_mode"]
                or task["temperature"] != first_task["temperature"]
                for task in matching_tasks
            ):
                raise ValueError(
                    f"configuration {configuration_root} has inconsistent "
                    "conductance_mode/temperature task settings"
                )
            collect_cmd.extend(
                [
                    "--configuration-settings",
                    json.dumps(
                        {
                            "configuration_dir": str(configuration_root),
                            "temperature": first_task["temperature"],
                            "conductance_mode": first_task["conductance_mode"],
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
