#!/usr/bin/env python3
"""重新提交 dpnegf 目录中未成功的任务。

复用 sub_dpnegf.py 的统一提交路径（含提交记录、重试、flag 协议），
而不是直接 sbatch run.sh —— 后者会绕过 job_id.txt /
dpnegf_submitted.flag 记录，并残留旧的 done/failed flag 造成假成功。
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

# 让本脚本无论从哪里调用都能 import stage 下的 sub_dpnegf。
_STAGE_DIR = Path(__file__).resolve().parent / "stage"
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

import sub_dpnegf  # noqa: E402


def negf_succeeded(workdir: Path) -> bool:
    """以非空的 output/negf.out.pth 作为成功标志。"""
    result_file = workdir / "output" / "negf.out.pth"
    return result_file.is_file() and result_file.stat().st_size > 0


def find_dpnegf_workdirs(root: Path):
    """返回所有含 run.sh 的 dpnegf leaf 工作目录。"""
    seen = set()
    for dpnegf_dir in sorted(root.rglob("dpnegf")):
        if not dpnegf_dir.is_dir():
            continue
        for run_sh in sorted(dpnegf_dir.rglob("run.sh")):
            workdir = run_sh.parent.resolve()
            if workdir not in seen:
                seen.add(workdir)
                yield workdir


def reset_workdir_state(workdir: Path) -> None:
    """删除旧的状态 flag / 提交记录 / output，使 sub_dpnegf 视为全新提交。

    sub_dpnegf.py 会因 done/failed/submitted flag 而跳过或报错，必须先清理。
    """
    for name in (
        "dpnegf_done.flag",
        "dpnegf_failed.flag",
        "dpnegf_started.flag",
        "dpnegf_submitted.flag",
        "dpnegf_submit_failed.flag",
        "job_id.txt",
    ):
        (workdir / name).unlink(missing_ok=True)

    output_dir = workdir / "output"
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="重新提交 dpnegf 目录中未成功的任务"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="搜索根目录，默认为当前目录",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="删除失败任务的 output 并重新提交；不加时只检查",
    )
    parser.add_argument(
        "--max-submit",
        type=int,
        default=None,
        help="本次最多重新提交多少个任务",
    )
    parser.add_argument(
        "--scheduler",
        choices=("local", "slurm"),
        default="slurm",
        help="提交方式，默认 slurm；local 直接阻塞运行",
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

    root = Path(args.root).resolve()

    failed_count = 0
    submit_count = 0
    dpnegf_count = 0

    for workdir in find_dpnegf_workdirs(root):
        dpnegf_count += 1
        print(f"\n[DPNEGF] {workdir}")

        if negf_succeeded(workdir):
            print(f"[SUCCESS] {workdir}")
            continue

        failed_count += 1
        print(f"[FAILED ] {workdir}")

        if not args.submit:
            continue

        if (
            args.max_submit is not None
            and submit_count >= args.max_submit
        ):
            print(f"[LIMIT  ] 已达到提交上限：{args.max_submit}")
            print_summary(
                dpnegf_count,
                failed_count,
                submit_count,
                args.submit,
            )
            return

        # 清理旧状态（flag / 提交记录 / output），否则 sub_dpnegf 会跳过。
        reset_workdir_state(workdir)
        print(f"          已重置状态：{workdir}")

        # 复用 sub_dpnegf.py 的统一提交路径，保证 flag 协议与提交记录一致。
        cmd = [
            sys.executable,
            str(Path(sub_dpnegf.__file__).resolve()),
            str(workdir),
            "--scheduler",
            args.scheduler,
            "--submit-retry-interval",
            str(args.submit_retry_interval),
            "--max-submit-retries",
            str(args.max_submit_retries),
        ]

        try:
            result = subprocess.run(
                cmd,
                text=True,
                check=True,
            )
            print(f"          sub_dpnegf 退出码={result.returncode}")
            submit_count += 1

        except subprocess.CalledProcessError as exc:
            print(f"[ERROR  ] 重新提交失败：{workdir}")
            print(f"          returncode={exc.returncode}")

    print_summary(
        dpnegf_count,
        failed_count,
        submit_count,
        args.submit,
    )


def print_summary(dpnegf_count, failed_count, submit_count, submitted):
    print()
    print(f"找到的 dpnegf 任务数：{dpnegf_count}")
    print(f"未成功任务数：{failed_count}")

    if submitted:
        print(f"成功重新提交数：{submit_count}")
    else:
        print("当前为检查模式，没有删除 output 或提交任务。")
        print("确认无误后添加 --submit")


if __name__ == "__main__":
    main()
