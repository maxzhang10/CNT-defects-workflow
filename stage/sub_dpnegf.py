#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import time
import glob

import logkit as L


def check_required_files(workdir: Path):
    """
    检查 DPNEGF 工作目录中必须存在的文件。

    最低要求：
        input.json
        run.py
        run.sh
        至少一个 *.xyz
        至少一个 nnenv*.pth
    """

    required = [
        "input.json",
        "run.py",
        "run.sh",
    ]

    missing = []

    for name in required:
        if not (workdir / name).exists():
            missing.append(name)

    # 检查 xyz 结构文件
    xyz_files = list(workdir.glob("*.xyz"))
    if len(xyz_files) == 0:
        missing.append("*.xyz")

    # 检查模型文件
    model_files = list(workdir.glob("nnenv*.pth"))
    if len(model_files) == 0:
        missing.append("nnenv*.pth")

    if missing:
        raise FileNotFoundError(
            f"Missing files in {workdir}:\n" + "\n".join(missing)
        )


def run_dpnegf_local(workdir: Path) -> str:
    """
    容器节点本地运行 DPNEGF。
    等价于在 workdir 里执行：
        bash run.sh
    """

    L.info(f"本地运行 DPNEGF: {workdir}")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "dpnegf_started.flag").write_text(start_time + "\n")

    stdout_path = workdir / "local_run.stdout"
    stderr_path = workdir / "local_run.stderr"

    with open(stdout_path, "w", encoding="utf-8") as fout, \
         open(stderr_path, "w", encoding="utf-8") as ferr:

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
            f"Return code: {result.returncode}\n"
        )

        raise RuntimeError(
            f"DPNEGF local run failed in {workdir}\n"
            f"Return code: {result.returncode}\n"
            f"See:\n"
            f"  {stdout_path}\n"
            f"  {stderr_path}"
        )

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # Python 侧写 done flag
    # run.sh 里也可以 touch dpnegf_done.flag，双保险
    (workdir / "dpnegf_done.flag").write_text(end_time + "\n")

    return "local"


def write_submit_record(workdir: Path, job_id: str):
    (workdir / "job_id.txt").write_text(job_id + "\n")
    (workdir / "dpnegf_submitted.flag").write_text("local\n")


def main():
    if len(sys.argv) != 2:
        L.error("用法: python sub_dpnegf.py /path/to/dpnegf_workdir")
        sys.exit(1)

    workdir = Path(sys.argv[1]).resolve()

    if not workdir.exists():
        raise FileNotFoundError(f"Workdir does not exist: {workdir}")

    done_flag = workdir / "dpnegf_done.flag"
    failed_flag = workdir / "dpnegf_failed.flag"
    submitted_flag = workdir / "dpnegf_submitted.flag"
    job_id_file = workdir / "job_id.txt"

    # 已经成功完成：直接跳过
    if done_flag.exists():
        L.skip(f"DPNEGF 已完成，跳过: {workdir}")
        return

    # 上一次失败：默认不自动重跑
    if failed_flag.exists():
        L.warn(f"上次 DPNEGF 运行失败: {workdir}")
        L.warn(f"如需重跑请删除 {failed_flag}")
        return

    # 已经启动过但没有 done：本地模式下通常说明上次中断
    if submitted_flag.exists() and job_id_file.exists():
        job_id = job_id_file.read_text().strip()
        L.warn(f"DPNEGF 之前已启动。job_id = {job_id}")
        L.warn("未找到 dpnegf_done.flag。")
        L.warn("如需重跑请删除 dpnegf_submitted.flag 和 job_id.txt。")
        return

    check_required_files(workdir)

    job_id = run_dpnegf_local(workdir)

    write_submit_record(workdir, job_id)

    L.ok("DPNEGF 运行成功。")
    L.info(f"workdir = {workdir}")
    L.info(f"job_id  = {job_id}")
    L.info(f"done    = {workdir / 'dpnegf_done.flag'}")


if __name__ == "__main__":
    main()