#!/bin/bash
#SBATCH --job-name=cnt_lmp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# 清理旧标记
rm -f lammps_done.flag lammps_failed.flag

# 无论在哪里失败，都生成失败标记
on_exit() {
    exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        echo "[ERROR] LAMMPS job failed at $(date), exit code=${exit_code}"
        rm -f lammps_done.flag
        touch lammps_failed.flag
    fi
}

trap on_exit EXIT

echo "======================================"
echo "LAMMPS GPU run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
echo "======================================"

source ~/dpmd.sh

# DeePMD/PyTorch CPU 线程设置

if [[ ! -f "in.lammps" ]]; then
    echo "[ERROR] Cannot find input file: $(pwd)/in.lammps"
    exit 1
fi




LMP_CMD="/data02/run01/scv7sc4/test/lammps/lammps-build-gpu/lmp"

NPROC="${SLURM_NTASKS:-1}"

echo "LAMMPS executable: ${LMP_CMD}"
echo "MPI processes: ${NPROC}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Allocated GPU:"
    nvidia-smi -L
    nvidia-smi \
        --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
else
    echo "[WARNING] nvidia-smi is unavailable."
fi

echo "======================================"
echo "Running LAMMPS"
echo "======================================"

/data02/run01/scv7sc4/test/lammps/lammps-build-gpu/lmp -nonbuf -in in.lammps > lammps.log 2>&1

touch lammps_done.flag
rm -f lammps_failed.flag

echo "======================================"
echo "Finish time: $(date)"
echo "LAMMPS finished successfully"
echo "LAMMPS log: $(pwd)/lammps.log"
echo "Done flag: $(pwd)/lammps_done.flag"
echo "======================================"