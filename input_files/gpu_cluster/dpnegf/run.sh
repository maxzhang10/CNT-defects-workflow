#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --job-name=cnt_negf
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=08:00:00

# ============================================================
# DPNEGF 输运计算作业（GPU SLURM 版本）
#
# 由 sub_dpnegf.py --scheduler slurm 通过：
#
#     sbatch run.sh
#
# 提交到 GPU 计算节点。
#
# 本脚本会被 copy_input_dpnegf.py 原样复制到每个 DPNEGF leaf。
#
# 状态文件（落盘逻辑见下方）：
#   作业开始：started flag
#   计算成功：done flag
#   计算失败：failed flag
#
# CPU 配置：
#   --ntasks=1
#   --cpus-per-task=6
#
# run.py 中的：
#   NEGF(..., n_cpus=6)
#
# 必须与这里保持一致。
#
# GPU 配置：
#   --gpus=1
# ============================================================

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "======================================"
echo "DPNEGF GPU SLURM run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-N/A}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
echo "======================================"

# 清理上一次运行留下的最终状态文件。
# submitted.flag 和 job_id.txt 由 sub_dpnegf.py 管理。
rm -f dpnegf_done.flag
rm -f dpnegf_failed.flag
rm -f dpnegf_started.flag

on_exit() {
    rc=$?

    if (( rc != 0 )); then
        {
            echo "Failed at $(date)"
            echo "Exit code: ${rc}"
            echo "Host: $(hostname)"
            echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
        } > dpnegf_failed.flag

        echo "[ERROR] DPNEGF failed, exit code=${rc}" >&2
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

# 标记作业已经真正开始运行。
date "+%Y-%m-%d %H:%M:%S" > dpnegf_started.flag

# 加载 DeePTB、DPNEGF 和 CUDA 环境。
source ~/dptb.sh

echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"

# 当前作业只申请了 6 个 CPU 核，避免底层数值库超额开线程。
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "======================================"
    echo "Allocated GPU"
    nvidia-smi -L
    nvidia-smi \
        --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
    echo "======================================"
else
    echo "[WARNING] nvidia-smi command not found."
fi

echo "Starting DPNEGF calculation..."

python run.py

echo "======================================"
echo "DPNEGF finished successfully"
echo "Finish time: $(date)"
echo "======================================"