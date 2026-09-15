#!/usr/bin/env python3
"""
SLURM 通用工具：非阻塞 sbatch 提交、提交限额自动重试与 squeue 屏障。

设计原则
--------
1. sbatch 成功以前，不应写 submitted/started/failed 等计算状态。
2. AssocMaxSubmitJobLimit 等提交数量上限属于临时状态，等待后重试。
3. 非临时 sbatch 错误属于“提交失败”，不是“计算失败”。
4. 作业真正开始、成功和失败，应由计算节点上的 run.sh 写：
       <kind>_started.flag
       <kind>_done.flag
       <kind>_failed.flag
"""

from pathlib import Path
from typing import Iterable, Optional
import random
import subprocess
import time

import logkit as L


class SlurmSubmitError(RuntimeError):
    """sbatch 提交失败，但作业并未进入计算队列。"""


class SlurmSubmitLimitError(SlurmSubmitError):
    """达到账号、关联或 QOS 的提交数量上限，可以等待后重试。"""


_SUBMIT_LIMIT_PATTERNS = (
    "AssocMaxSubmitJobLimit",
    "QOSMaxSubmitJobPerUserLimit",
    "QOSMaxSubmitJobLimit",
    "MaxSubmitJobs",
    "MaxSubmitJob",
    "job submit limit",
    "maximum number of submitted jobs",
    "maximum number of jobs",
)


def _is_submit_limit_error(stderr: str) -> bool:
    """判断 sbatch 错误是否属于可重试的提交数量上限。"""
    text = (stderr or "").lower()
    return any(pattern.lower() in text for pattern in _SUBMIT_LIMIT_PATTERNS)


def sbatch_submit(workdir: Path, script: str = "run.sh") -> str:
    """
    在 workdir 中执行一次 `sbatch --parsable <script>`。

    返回：
        SLURM job_id。

    异常：
        SlurmSubmitLimitError
            达到提交数量上限，可等待后重试。
        SlurmSubmitError
            其他 sbatch 错误，例如脚本、账户、分区或参数错误。
    """
    workdir = Path(workdir).resolve()

    result = subprocess.run(
        ["sbatch", "--parsable", script],
        cwd=workdir,
        text=True,
        capture_output=True,
    )

    stderr = result.stderr.strip()

    if result.returncode != 0:
        message = (
            f"sbatch 提交失败: {workdir}\n"
            f"Return code: {result.returncode}\n"
            f"stderr: {stderr}"
        )

        if _is_submit_limit_error(stderr):
            raise SlurmSubmitLimitError(message)

        raise SlurmSubmitError(message)

    # --parsable 输出通常是：
    #   123456
    # 或：
    #   123456;cluster
    job_id = result.stdout.strip().split(";", 1)[0].strip()

    if not job_id:
        raise SlurmSubmitError(
            f"sbatch 未返回 job_id: {workdir}\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )

    return job_id


def sbatch_submit_with_retry(
    workdir: Path,
    script: str = "run.sh",
    retry_interval: int = 60,
    max_retries: Optional[int] = None,
    label: str = "",
    jitter: int = 10,
) -> str:
    """
    提交作业；遇到提交数量上限时自动等待并重试。

    参数
    ----
    retry_interval:
        基础等待秒数，默认 60 秒。
    max_retries:
        最大重试次数。None 表示无限重试。
    label:
        日志标签，例如 LAMMPS 或 DPNEGF。
    jitter:
        在基础等待时间上增加 0~jitter 秒随机延迟，
        避免大量 workflow 同时醒来、再次一起撞提交上限。

    只有提交数量上限会重试。其他 sbatch 错误会立即抛出。
    """
    workdir = Path(workdir).resolve()

    if retry_interval < 1:
        raise ValueError("retry_interval 必须大于等于 1")

    if max_retries is not None and max_retries < 0:
        raise ValueError("max_retries 不能小于 0")

    attempt = 0
    prefix = f"[{label}] " if label else ""

    while True:
        try:
            return sbatch_submit(workdir, script)

        except SlurmSubmitLimitError as exc:
            attempt += 1

            if max_retries is not None and attempt > max_retries:
                raise SlurmSubmitLimitError(
                    f"{exc}\n"
                    f"已达到最大重试次数: {max_retries}"
                ) from exc

            extra = random.uniform(0, max(0, jitter))
            sleep_seconds = retry_interval + extra
            retry_total = "无限" if max_retries is None else str(max_retries)

            L.warn(
                f"{prefix}SLURM 提交数量达到上限；"
                f"本次没有创建作业，也不会写计算失败 flag。"
            )
            L.info(
                f"{prefix}{sleep_seconds:.1f} 秒后重试 "
                f"（第 {attempt}/{retry_total} 次）: {workdir}"
            )

            time.sleep(sleep_seconds)


class SlurmQueryError(RuntimeError):
    """squeue 查询本身失败（非"作业不存在"），不能当作完成。"""


def _running_job_ids(job_ids: Iterable[str]):
    """
    查询给定 job_id 中仍存在于 squeue 的作业。

    返回 alive 集合。当 squeue 查询本身失败（空 stdout + 非零返回码，
    如认证/网络故障）时抛 SlurmQueryError，而不是把空结果当成"全部离队"。
    """
    job_ids = [str(job_id) for job_id in job_ids if job_id]
    if not job_ids:
        return set()

    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "-o",
            "%A",
            "-j",
            ",".join(job_ids),
        ],
        text=True,
        capture_output=True,
    )

    # 部分作业已结束时，squeue 可能返回非零且仍输出存活的 job。
    # 仍从 stdout 中提取还能查到的 job。
    alive = {
        token
        for token in result.stdout.split()
        if token.isdigit()
    }

    # P2-2：非零返回码 + 空 stdout 意味着查询本身失败（认证/网络/调度器
    # 故障），不是所有作业都已离队。把它当作独立查询错误抛出，否则
    # wait_for_jobs 会把空 alive 当成完成，配合陈旧 done flag 造成假成功。
    if result.returncode != 0 and not alive:
        raise SlurmQueryError(
            f"squeue 查询失败 (rc={result.returncode}, stderr={result.stderr.strip()})，"
            f"不能确定作业状态: {job_ids}"
        )

    return alive & set(job_ids)


def wait_for_jobs(job_ids, poll_interval=30, label=""):
    """
    阻塞直到给定 job_id 全部离开 squeue。

    离开队列不代表计算成功，调用方仍应检查 done/failed flag。
    squeue 查询本身失败时（SlurmQueryError）重试若干次，仍失败则抛错，
    不会把查询故障当成"所有作业已离队"。
    """
    job_ids = [str(job_id) for job_id in job_ids if job_id]
    if not job_ids:
        return

    prefix = f"[{label}] " if label else ""
    L.info(f"{prefix}等待 {len(job_ids)} 个 SLURM 作业完成…")

    max_query_retries = 5
    query_retry_delay = min(poll_interval, 30)

    while True:
        try:
            alive = _running_job_ids(job_ids)
        except SlurmQueryError as exc:
            # 临时查询故障：重试，避免把故障当成完成。
            if max_query_retries > 0:
                max_query_retries -= 1
                L.warn(
                    f"{prefix}squeue 查询失败，{query_retry_delay:.0f} 秒后重试"
                    f"（剩余 {max_query_retries} 次）: {exc}"
                )
                time.sleep(query_retry_delay)
                continue
            raise

        if not alive:
            L.ok(f"{prefix}全部 {len(job_ids)} 个作业已离开队列")
            return

        L.info(
            f"{prefix}仍有 {len(alive)}/{len(job_ids)} "
            f"个作业在队列中…"
        )
        time.sleep(poll_interval)


def check_flags(workdirs, kind):
    """
    检查 <kind>_done.flag / <kind>_failed.flag。

    返回：
        (done_dirs, failed_dirs, missing_dirs)
    """
    done_dirs = []
    failed_dirs = []
    missing_dirs = []

    for workdir in workdirs:
        workdir = Path(workdir)

        if (workdir / f"{kind}_done.flag").exists():
            done_dirs.append(workdir)
        elif (workdir / f"{kind}_failed.flag").exists():
            failed_dirs.append(workdir)
        else:
            missing_dirs.append(workdir)

    return done_dirs, failed_dirs, missing_dirs
