#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import subprocess


def negf_succeeded(workdir: Path) -> bool:
    """以非空的 output/negf.out.pth 作为成功标志。"""
    result_file = workdir / "output" / "negf.out.pth"
    return result_file.is_file() and result_file.stat().st_size > 0


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
    args = parser.parse_args()

    root = Path(args.root).resolve()

    failed_count = 0
    submit_count = 0
    dpnegf_count = 0

    # 只有名字恰好为 dpnegf 的目录才会被处理
    for dpnegf_dir in sorted(root.rglob("dpnegf")):
        if not dpnegf_dir.is_dir():
            continue

        dpnegf_count += 1
        print(f"\n[DPNEGF] {dpnegf_dir}")

        # 只在该 dpnegf 目录内部查找 run.sh
        for run_sh in sorted(dpnegf_dir.rglob("run.sh")):
            workdir = run_sh.parent

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

            output_dir = workdir / "output"

            if output_dir.is_dir():
                shutil.rmtree(output_dir)
                print(f"          已删除：{output_dir}")
            elif output_dir.exists():
                output_dir.unlink()
                print(f"          已删除：{output_dir}")
            else:
                print("          output 不存在，无需删除")

            try:
                result = subprocess.run(
                    ["sbatch", "run.sh"],
                    cwd=workdir,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                print(f"          {result.stdout.strip()}")
                submit_count += 1

            except subprocess.CalledProcessError as exc:
                print(f"[ERROR  ] 提交失败：{workdir}")

                if exc.stdout:
                    print(f"          stdout: {exc.stdout.strip()}")
                if exc.stderr:
                    print(f"          stderr: {exc.stderr.strip()}")

    print_summary(
        dpnegf_count,
        failed_count,
        submit_count,
        args.submit,
    )


def print_summary(dpnegf_count, failed_count, submit_count, submitted):
    print()
    print(f"找到的 dpnegf 目录数：{dpnegf_count}")
    print(f"未成功任务数：{failed_count}")

    if submitted:
        print(f"成功重新提交数：{submit_count}")
    else:
        print("当前为检查模式，没有删除 output 或提交任务。")
        print("确认无误后添加 --submit")


if __name__ == "__main__":
    main()