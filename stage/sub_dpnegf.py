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
import subprocess
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
    write_hash_file,
)


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

    if not list(workdir.glob("nnenv*.pth")):
        missing.append("nnenv*.pth")

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


def main():
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
        return

    if failed_flag.exists():
        L.warn(f"上次 DPNEGF 真正运行失败: {workdir}")
        L.warn(f"如需重跑请删除 {failed_flag}")
        return

    if submitted_flag.exists() and job_id_file.exists():
        job_id = job_id_file.read_text(
            encoding="utf-8",
        ).strip()
        L.warn(
            f"DPNEGF 之前已提交，跳过重复提交。job_id={job_id}"
        )
        return

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


if __name__ == "__main__":
    main()
