#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import time


def check_required_files(workdir: Path):
    required = [
        "data.lmp",
        "in.lammps",
        "CH.airebo-m",
        "run.sh",
    ]

    missing = []
    for name in required:
        if not (workdir / name).exists():
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"Missing files in {workdir}:\n" + "\n".join(missing)
        )


def run_lammps_local(workdir: Path) -> str:
    """
    容器节点本地运行 LAMMPS。
    等价于在 workdir 里执行：
        bash run.sh
    """

    print(f"[INFO] Running LAMMPS locally in {workdir}")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (workdir / "lammps_started.flag").write_text(start_time + "\n")

    result = subprocess.run(
        ["bash", "run.sh"],
        cwd=workdir,
        text=True,
        capture_output=True,
    )

    # 保存 Python 捕获到的标准输出和错误输出
    (workdir / "local_run.stdout").write_text(result.stdout)
    (workdir / "local_run.stderr").write_text(result.stderr)

    if result.returncode != 0:
        fail_time = time.strftime("%Y-%m-%d %H:%M:%S")
        (workdir / "lammps_failed.flag").write_text(
            f"Failed at {fail_time}\n"
            f"Return code: {result.returncode}\n"
        )

        raise RuntimeError(
            f"LAMMPS local run failed in {workdir}\n"
            f"Return code: {result.returncode}\n"
            f"See:\n"
            f"  {workdir / 'local_run.stdout'}\n"
            f"  {workdir / 'local_run.stderr'}"
        )

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # 这里写 done flag
    # run.sh 里也可以 touch lammps_done.flag，双保险
    (workdir / "lammps_done.flag").write_text(end_time + "\n")

    return "local"


def write_submit_record(workdir: Path, job_id: str):
    (workdir / "job_id.txt").write_text(job_id + "\n")
    (workdir / "lammps_submitted.flag").write_text("local\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python sub_lmps.py /path/to/lammps_workdir")
        sys.exit(1)

    workdir = Path(sys.argv[1]).resolve()

    if not workdir.exists():
        raise FileNotFoundError(f"Workdir does not exist: {workdir}")

    done_flag = workdir / "lammps_done.flag"
    failed_flag = workdir / "lammps_failed.flag"
    submitted_flag = workdir / "lammps_submitted.flag"
    job_id_file = workdir / "job_id.txt"

    # 已经成功完成：直接跳过
    if done_flag.exists():
        print(f"[SKIP] LAMMPS already finished: {workdir}")
        return

    # 上一次失败：默认不自动重跑，避免反复炸
    if failed_flag.exists():
        print(f"[WARN] Previous LAMMPS run failed: {workdir}")
        print(f"[WARN] Remove {failed_flag} if you want to rerun.")
        return

    # 已经提交/启动过但没 done：容器本地模式下，这通常说明上次中断了
    if submitted_flag.exists() and job_id_file.exists():
        job_id = job_id_file.read_text().strip()
        print(f"[WARN] LAMMPS was started before. job_id = {job_id}")
        print("[WARN] No lammps_done.flag found.")
        print("[WARN] Remove lammps_submitted.flag and job_id.txt if you want to rerun.")
        return

    check_required_files(workdir)

    job_id = run_lammps_local(workdir)

    write_submit_record(workdir, job_id)

    print("LAMMPS finished successfully.")
    print(f"workdir = {workdir}")
    print(f"job_id  = {job_id}")
    print(f"done    = {workdir / 'lammps_done.flag'}")


if __name__ == "__main__":
    main()