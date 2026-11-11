#!/bin/bash
#SBATCH -p v6_384            # TODO: 改成你集群的 DPNEGF 计算分区名
#SBATCH -N 1
#SBATCH -n 48               # 必须与 run.py 里 NEGF(..., n_cpus=48) 保持一致
#SBATCH -J cnt_negf
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -t 08:00:00

# ============================================================
# DPNEGF 输运计算作业（SLURM 版本）
#
# 由 sub_dpnegf.py --scheduler slurm 通过 `sbatch run.sh` 提交到计算节点。
# 本脚本会被 copy_input_dpnegf.py 原样拷进每个 leaf 目录，因此
# flag 落盘逻辑必须写在这里：
#   成功 -> touch dpnegf_done.flag
#   失败 -> touch dpnegf_failed.flag
# 编排层 run.py 等 squeue 队列清空后，靠这两个 flag 判断结果。
#
# 注意：#SBATCH -n 与 run.py 的 n_cpus 要对齐（当前都是 32）。
# ============================================================

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# 清理本次运行的旧状态，避免上一轮的 done/failed/started flag 干扰判断。
rm -f dpnegf_done.flag
rm -f dpnegf_failed.flag
rm -f dpnegf_started.flag

on_exit() {
    rc=$?

    if (( rc != 0 )); then
        {
            echo "Failed at: $(date '+%F %T')"
            echo "Return code: ${rc}"
            echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
            echo "Host: $(hostname)"
        } > dpnegf_failed.flag

        echo "[ERROR] DPNEGF failed, return code=${rc}" >&2
        return
    fi

    # 成功：写入 provenance hash（与 output/negf.out.pth 同层，
    # 用于后续校验 done 结果是否与当前配置一致），并落 done flag。
    if [ -f expected_negf_config_hash.txt ]; then
        cp expected_negf_config_hash.txt output/negf_config_hash.txt
    fi
    touch dpnegf_done.flag
    rm -f dpnegf_failed.flag
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source ~/software-t6s008517/DeePTB/.venv/bin/activate

date '+%F %T' > dpnegf_started.flag

echo "======================================"
echo "DPNEGF slurm run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "======================================"

python run.py

echo "DPNEGF finished successfully at $(date)"
