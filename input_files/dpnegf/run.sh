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

echo "======================================"
echo "DPNEGF slurm run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "======================================"

source ~/software-t6s008517/DeePTB/.venv/bin/activate

if python run.py; then
    echo "DPNEGF finished successfully at $(date)"
    touch dpnegf_done.flag
else
    echo "[ERROR] DPNEGF 失败，写 dpnegf_failed.flag"
    touch dpnegf_failed.flag
    exit 1
fi
