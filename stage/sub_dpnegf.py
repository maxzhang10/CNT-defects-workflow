#!/usr/bin/env python3
"""
提交单个 DPNEGF 工作目录。

SLURM 状态语义
--------------
尚未提交：
    无 job_id.txt
    无 dpnegf_submitted.flag
    无 dpnegf_started.flag
    无 dpnegf_failed.flag

sbatch 成功：
    job_id.txt
    dpnegf_submitted.flag

作业真正开始：
    由 run.sh 写 dpnegf_started.flag

真正计算成功/失败：
    由 run.sh 写 dpnegf_done.flag / dpnegf_failed.flag

达到 AssocMaxSubmitJobLimit 时自动等待重试；
sbatch 成功前不会写 started/failed。
"""

from pathlib import Path
from typing import Optional
import argparse
import os
import shutil
import subprocess
import sys
import time

import logkit as L
from slurm_utils import (
    SlurmSubmitError,
    SlurmSubmitLimitError,
    sbatch_submit_with_retry,
)
from negf_provenance import (
    expected_hash_path,
    provenance_hash_path,
    provenance_is_valid,
    read_hash_file,
    result_file_path,
    result_is_readable,
    write_hash_file,
)


def local_dpnegf_python() -> str:
    """返回本地运行 DPNEGF run.py 的 Python 解释器。

    local 模式不应复用 SLURM 模板 run.sh 中硬编码的 venv 路径
    (~/software-t6s008517/DeePTB/.venv/bin/activate)，而应使用当前
    解释器（即 run_multi.py 通过 --python 选定的解释器）。
    可通过环境变量 CNT_DPTB_PYTHON 覆盖。
    """
    return os.environ.get("CNT_DPTB_PYTHON") or sys.executable


def copy_provenance_hash_on_success(workdir: Path) -> None:
    """成功后将 expected hash 复制为 provenance hash（与 run.sh 行为一致）。

    SLURM 模式下该复制由 run.sh 完成；local 模式绕过 run.sh，必须在此补做，
    否则下次 sub_dpnegf.py 的 done-flag provenance 校验会因缺少
    output/negf_config_hash.txt 而拒绝复用结果。
    """
    expected = expected_hash_path(workdir)
    if not expected.is_file():
        return
    target = provenance_hash_path(workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(expected, target)


def check_required_files(workdir: Path):
    required = (
        "input.json",
        "run.py",
        "run.sh",
    )

    missing = [
        name
        for name in required
        if not (workdir / name).exists()
    ]

    if not list(workdir.glob("*.xyz")):
        missing.append("*.xyz")

    # P2-12：接受任意合法名称的 .pth 模型文件，不硬编码 nnenv*.pth。
    if not list(workdir.glob("*.pth")):
        missing.append("*.pth")

    if missing:
        raise FileNotFoundError(
            f"Missing files in {workdir}:\n"
            + "\n".join(missing)
        )


def cleanup_legacy_false_flags(workdir: Path):
    """
    清理旧版脚本因 sbatch 未成功而误写的假 flag。
    """
    if (workdir / "dpnegf_done.flag").exists():
        return

    if (workdir / "dpnegf_submitted.flag").exists():
        return

    if (workdir / "job_id.txt").exists():
        return

    started_flag = workdir / "dpnegf_started.flag"
    failed_flag = workdir / "dpnegf_failed.flag"
    removed = []

    if started_flag.exists():
        started_flag.unlink()
        removed.append(started_flag.name)

    if failed_flag.exists():
        text = failed_flag.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if "sbatch submit failed" in text.lower():
            failed_flag.unlink()
            removed.append(failed_flag.name)

    if removed:
        L.warn(
            f"已清理旧版误写 flag: {workdir} -> "
            + ", ".join(removed)
        )


def write_submit_record(workdir: Path, job_id: str):
    """只有 sbatch 成功后才写提交记录。"""
    (workdir / "job_id.txt").write_text(
        job_id + "\n",
        encoding="utf-8",
    )
    (workdir / "dpnegf_submitted.flag").write_text(
        "slurm\n",
        encoding="utf-8",
    )

    (workdir / "dpnegf_submit_failed.flag").unlink(
        missing_ok=True
    )


def run_dpnegf_local(workdir: Path) -> str:
    """当前节点阻塞运行 DPNEGF。"""
    L.info(f"本地运行 DPNEGF: {workdir}")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "dpnegf_started.flag").write_text(
        start_time + "\n",
        encoding="utf-8",
    )

    stdout_path = workdir / "local_run.stdout"
    stderr_path = workdir / "local_run.stderr"

    # 用当前解释器直接运行 run.py，不复用 SLURM 模板 run.sh
    # （后者激活硬编码的 ~/software-t6s008517/DeePTB/.venv 虚拟环境）。
    python = local_dpnegf_python()
    cmd = [python, "run.py"]
    L.info(f"DPNEGF 本地命令: {' '.join(cmd)} (cwd={workdir})")

    try:
        with stdout_path.open("w", encoding="utf-8") as fout, \
             stderr_path.open("w", encoding="utf-8") as ferr:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                text=True,
                stdout=fout,
                stderr=ferr,
            )
    except FileNotFoundError as exc:
        fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
        (workdir / "dpnegf_failed.flag").write_text(
            f"Failed at {fail_time}\n"
            f"Python interpreter not found: {exc}\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"DPNEGF 本地 Python 解释器未找到，请用环境变量 CNT_DPTB_PYTHON 指定。\n"
            f"  workdir = {workdir}\n"
            f"  cmd = {' '.join(cmd)}\n"
            f"  {exc}"
        ) from exc

    if result.returncode != 0:
        fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
        (workdir / "dpnegf_failed.flag").write_text(
            f"Failed at {fail_time}\n"
            f"Return code: {result.returncode}\n",
            encoding="utf-8",
        )

        raise RuntimeError(
            f"DPNEGF local run failed in {workdir}\n"
            f"Return code: {result.returncode}\n"
            f"See:\n"
            f"  {stdout_path}\n"
            f"  {stderr_path}"
        )

    # run.py 成功后，把 expected hash 复制为 provenance hash
    # （SLURM 模式下由 run.sh 完成的步骤，local 模式必须在此补做）。
    copy_provenance_hash_on_success(workdir)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "dpnegf_done.flag").write_text(
        end_time + "\n",
        encoding="utf-8",
    )

    return "local"


def submit_dpnegf_slurm(
    workdir: Path,
    retry_interval: int,
    max_submit_retries: Optional[int],
) -> str:
    """
    提交 DPNEGF。

    达到提交数量上限时自动等待；
    sbatch 成功前不写任何计算状态 flag。
    """
    L.info(f"sbatch 提交 DPNEGF: {workdir}")

    job_id = sbatch_submit_with_retry(
        workdir=workdir,
        script="run.sh",
        retry_interval=retry_interval,
        max_retries=max_submit_retries,
        label="DPNEGF",
    )

    write_submit_record(workdir, job_id)

    L.ok(f"已提交 DPNEGF 作业 job_id = {job_id}")
    return job_id


# 遇到上次真实失败 flag 时的退出码。
# main() 返回它，__main__ 用 sys.exit() 传播；run_multi.py 的 run_cmd
# 用 check=True 调用，会因此抛 CalledProcessError 中断 local 流程，
# 避免把旧 failed flag 当作成功并继续消费陈旧 output。
EXIT_PRIOR_FAILURE = 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="提交单个 DPNEGF 工作目录（local 或 slurm）"
    )
    parser.add_argument(
        "workdir",
        help="DPNEGF 工作目录",
    )
    parser.add_argument(
        "--scheduler",
        choices=("local", "slurm"),
        default="local",
        help="local: 阻塞运行；slurm: 非阻塞 sbatch",
    )
    parser.add_argument(
        "--submit-retry-interval",
        type=int,
        default=60,
        help="达到 SLURM 提交上限后的基础重试间隔，默认 60 秒",
    )
    parser.add_argument(
        "--max-submit-retries",
        type=int,
        default=0,
        help="最大重试次数；0 表示无限重试",
    )
    args = parser.parse_args()

    if args.submit_retry_interval < 1:
        raise ValueError(
            "--submit-retry-interval 必须大于等于 1"
        )

    if args.max_submit_retries < 0:
        raise ValueError(
            "--max-submit-retries 不能小于 0"
        )

    max_submit_retries = (
        None
        if args.max_submit_retries == 0
        else args.max_submit_retries
    )

    workdir = Path(args.workdir).resolve()

    if not workdir.is_dir():
        raise FileNotFoundError(
            f"Workdir does not exist: {workdir}"
        )

    cleanup_legacy_false_flags(workdir)

    done_flag = workdir / "dpnegf_done.flag"
    failed_flag = workdir / "dpnegf_failed.flag"
    submitted_flag = workdir / "dpnegf_submitted.flag"
    job_id_file = workdir / "job_id.txt"

    if done_flag.exists():
        # Fail-safe：done flag 不等于成功。必须同时验证结果文件非空且
        # provenance hash（output/negf_config_hash.txt）与本次期望的 config hash
        # （expected_negf_config_hash.txt，由 copy_input_dpnegf.py 写入）完全一致，
        # 才允许复用旧结果；否则禁止复用，明确报错（P0-4）。
        # P1-10：进一步要求结果文件可读取（torch.load 能成功），
        # 损坏的结果同样视为失败，进入重试而非被当作成功。
        if not result_is_readable(workdir):
            result = result_file_path(workdir)
            raise RuntimeError(
                f"done flag 存在但结果文件不可读取（缺失/空/损坏）:\n"
                f"  workdir = {workdir}\n"
                f"  result  = {result}\n"
                f"请删除 {done_flag.name} 与 output/ 后显式重新计算。"
            )
        expected_hash = read_hash_file(expected_hash_path(workdir))
        if expected_hash is None:
            raise RuntimeError(
                f"无法复用旧结果: 缺少期望 config hash ({expected_hash_path(workdir).name})。\n"
                f"  workdir = {workdir}\n"
                f"请删除 {done_flag.name} 显式重新计算，或使用新目录。"
            )
        if not provenance_is_valid(workdir, expected_hash):
            saved_hash = read_hash_file(provenance_hash_path(workdir))
            result = result_file_path(workdir)
            if not result.is_file() or result.stat().st_size == 0:
                detail = f"结果文件缺失或为空: {result}"
            elif saved_hash is None:
                detail = f"缺少 provenance hash: {provenance_hash_path(workdir)}"
            else:
                detail = (
                    f"config 不一致: provenance hash={saved_hash} "
                    f"!= 期望 hash={expected_hash}"
                )
            raise RuntimeError(
                f"禁止复用旧 DPNEGF 结果，配置已变更或 provenance 无法确认:\n"
                f"  workdir = {workdir}\n"
                f"  {detail}\n"
                f"请使用新目录，或删除 {done_flag.name} 与 output/ 后显式重新计算。"
            )
        L.skip(f"DPNEGF 已完成且 provenance 一致，跳过: {workdir}")
        return 0

    if failed_flag.exists():
        L.warn(f"上次 DPNEGF 真正运行失败: {workdir}")
        L.warn(f"如需重跑请删除 {failed_flag}")
        # 返回非零，让 orchestrator（check=True）中止 local 流程，
        # 而不是把旧 failed flag 当成功并继续消费陈旧 output。
        return EXIT_PRIOR_FAILURE

    if submitted_flag.exists() and job_id_file.exists():
        job_id = job_id_file.read_text(
            encoding="utf-8",
        ).strip()
        L.warn(
            f"DPNEGF 之前已提交，跳过重复提交。job_id={job_id}"
        )
        return 0

    if submitted_flag.exists() != job_id_file.exists():
        L.warn(
            f"发现不完整提交记录，自动清理后重提: {workdir}"
        )
        submitted_flag.unlink(missing_ok=True)
        job_id_file.unlink(missing_ok=True)

    check_required_files(workdir)

    if args.scheduler == "local":
        job_id = run_dpnegf_local(workdir)
        L.ok("DPNEGF 本地运行完成。")
    else:
        try:
            job_id = submit_dpnegf_slurm(
                workdir=workdir,
                retry_interval=args.submit_retry_interval,
                max_submit_retries=max_submit_retries,
            )

        except SlurmSubmitLimitError:
            # 有限重试耗尽时仍不写 dpnegf_failed.flag。
            raise

        except SlurmSubmitError as exc:
            fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
            (workdir / "dpnegf_submit_failed.flag").write_text(
                f"Submit failed at {fail_time}\n{exc}\n",
                encoding="utf-8",
            )
            raise

    L.info(f"workdir = {workdir}")
    L.info(f"job_id  = {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
