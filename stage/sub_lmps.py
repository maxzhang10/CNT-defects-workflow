#!/usr/bin/env python3
"""
提交单个 LAMMPS 工作目录。

SLURM 状态语义
--------------
尚未提交：
    无 job_id.txt
    无 lammps_submitted.flag
    无 lammps_started.flag
    无 lammps_failed.flag

sbatch 成功：
    job_id.txt
    lammps_submitted.flag

作业真正开始：
    由 run.sh 写 lammps_started.flag

真正计算成功/失败：
    由 run.sh 写 lammps_done.flag / lammps_failed.flag

达到 AssocMaxSubmitJobLimit 时自动等待重试；
sbatch 成功前不会写 started/failed。
"""

from pathlib import Path
from typing import Optional
import argparse
import subprocess
import time

import logkit as L
from slurm_utils import (
    SlurmSubmitError,
    SlurmSubmitLimitError,
    sbatch_submit_with_retry,
)


def check_required_files(workdir: Path):
    required = (
        "data.lmp",
        "in.lammps",
        "CH.airebo-m",
        "run.sh",
    )

    missing = [
        name
        for name in required
        if not (workdir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing files in {workdir}:\n"
            + "\n".join(missing)
        )


def cleanup_legacy_false_flags(workdir: Path):
    """
    清理旧版脚本因 sbatch 未成功而误写的假 flag。

    只有以下记录都不存在时才处理：
        lammps_done.flag
        lammps_submitted.flag
        job_id.txt

    因此不会碰到真正已提交或已经完成的任务。
    """
    if (workdir / "lammps_done.flag").exists():
        return

    if (workdir / "lammps_submitted.flag").exists():
        return

    if (workdir / "job_id.txt").exists():
        return

    started_flag = workdir / "lammps_started.flag"
    failed_flag = workdir / "lammps_failed.flag"
    removed = []

    # 旧版在 sbatch 之前就写 started；没有提交记录时该 flag 不可信。
    if started_flag.exists():
        started_flag.unlink()
        removed.append(started_flag.name)

    # 只自动删除明确写着 sbatch submit failed 的假“计算失败”。
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
    """
    只有 sbatch 成功后才写提交记录。
    """
    (workdir / "job_id.txt").write_text(
        job_id + "\n",
        encoding="utf-8",
    )
    (workdir / "lammps_submitted.flag").write_text(
        "slurm\n",
        encoding="utf-8",
    )

    # 若上次是非临时提交错误，成功后清理诊断文件。
    (workdir / "lammps_submit_failed.flag").unlink(
        missing_ok=True
    )


def run_lammps_local(workdir: Path) -> str:
    """当前节点阻塞运行 LAMMPS。"""
    L.info(f"本地运行 LAMMPS: {workdir}")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "lammps_started.flag").write_text(
        start_time + "\n",
        encoding="utf-8",
    )

    stdout_path = workdir / "local_run.stdout"
    stderr_path = workdir / "local_run.stderr"

    with stdout_path.open("w", encoding="utf-8") as fout, \
         stderr_path.open("w", encoding="utf-8") as ferr:
        result = subprocess.run(
            ["bash", "run.sh"],
            cwd=workdir,
            text=True,
            stdout=fout,
            stderr=ferr,
        )

    if result.returncode != 0:
        fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
        (workdir / "lammps_failed.flag").write_text(
            f"Failed at {fail_time}\n"
            f"Return code: {result.returncode}\n",
            encoding="utf-8",
        )

        raise RuntimeError(
            f"LAMMPS local run failed in {workdir}\n"
            f"Return code: {result.returncode}\n"
            f"See:\n"
            f"  {stdout_path}\n"
            f"  {stderr_path}"
        )

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "lammps_done.flag").write_text(
        end_time + "\n",
        encoding="utf-8",
    )

    return "local"


def submit_lammps_slurm(
    workdir: Path,
    retry_interval: int,
    max_submit_retries: Optional[int],
) -> str:
    """
    提交 LAMMPS。

    达到提交数量上限时自动等待；
    sbatch 成功前不写任何计算状态 flag。
    """
    L.info(f"sbatch 提交 LAMMPS: {workdir}")

    job_id = sbatch_submit_with_retry(
        workdir=workdir,
        script="run.sh",
        retry_interval=retry_interval,
        max_retries=max_submit_retries,
        label="LAMMPS",
    )

    write_submit_record(workdir, job_id)

    L.ok(f"已提交 LAMMPS 作业 job_id = {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="提交单个 LAMMPS 工作目录（local 或 slurm）"
    )
    parser.add_argument(
        "workdir",
        help="LAMMPS 工作目录",
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

    done_flag = workdir / "lammps_done.flag"
    failed_flag = workdir / "lammps_failed.flag"
    submitted_flag = workdir / "lammps_submitted.flag"
    job_id_file = workdir / "job_id.txt"

    if done_flag.exists():
        L.skip(f"LAMMPS 已完成，跳过: {workdir}")
        return

    if failed_flag.exists():
        L.warn(f"上次 LAMMPS 真正运行失败: {workdir}")
        L.warn(f"如需重跑请删除 {failed_flag}")
        return

    if submitted_flag.exists() and job_id_file.exists():
        job_id = job_id_file.read_text(
            encoding="utf-8",
        ).strip()
        L.warn(
            f"LAMMPS 之前已提交，跳过重复提交。job_id={job_id}"
        )
        return

    # 处理孤立的 submitted.flag 或 job_id.txt。
    if submitted_flag.exists() != job_id_file.exists():
        L.warn(
            f"发现不完整提交记录，自动清理后重提: {workdir}"
        )
        submitted_flag.unlink(missing_ok=True)
        job_id_file.unlink(missing_ok=True)

    check_required_files(workdir)

    if args.scheduler == "local":
        job_id = run_lammps_local(workdir)
        L.ok("LAMMPS 本地运行完成。")
    else:
        try:
            job_id = submit_lammps_slurm(
                workdir=workdir,
                retry_interval=args.submit_retry_interval,
                max_submit_retries=max_submit_retries,
            )

        except SlurmSubmitLimitError:
            # 只有设置有限重试次数且耗尽后才会到这里。
            # 仍然不写 lammps_failed.flag。
            raise

        except SlurmSubmitError as exc:
            # 非临时 sbatch 错误写单独诊断文件，
            # 不伪装成 LAMMPS 计算失败。
            fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
            (workdir / "lammps_submit_failed.flag").write_text(
                f"Submit failed at {fail_time}\n{exc}\n",
                encoding="utf-8",
            )
            raise

    L.info(f"workdir = {workdir}")
    L.info(f"job_id  = {job_id}")


if __name__ == "__main__":
    main()
