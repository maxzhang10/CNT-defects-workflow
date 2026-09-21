#!/bin/bash
#SBATCH --job-name=cnt_lmp
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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

module load deepmdkit/v3.2.0b0_pytorch

# DeePMD/PyTorch CPU 线程设置
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DP_INTER_OP_PARALLELISM_THREADS=1

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "DP_INTRA_OP_PARALLELISM_THREADS=${DP_INTRA_OP_PARALLELISM_THREADS}"
echo "DP_INTER_OP_PARALLELISM_THREADS=${DP_INTER_OP_PARALLELISM_THREADS}"

if [[ ! -f "in.lammps" ]]; then
    echo "[ERROR] Cannot find input file: $(pwd)/in.lammps"
    exit 1
fi

if command -v lmp_mpi >/dev/null 2>&1; then
    LMP_CMD="$(command -v lmp_mpi)"
elif command -v lmp >/dev/null 2>&1; then
    LMP_CMD="$(command -v lmp)"
elif command -v lammps_mpi >/dev/null 2>&1; then
    LMP_CMD="$(command -v lammps_mpi)"
else
    echo "[ERROR] Cannot find LAMMPS executable."
    echo "[INFO] Current PATH=${PATH}"
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] Cannot find mpirun."
    exit 1
fi

NPROC="${SLURM_NTASKS:-1}"

echo "LAMMPS executable: ${LMP_CMD}"
echo "MPI executable: $(command -v mpirun)"
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

mpirun -np "${NPROC}" \
    "${LMP_CMD}" \
    -in in.lammps \
    > lammps.log 2>&1

touch lammps_done.flag
rm -f lammps_failed.flag

echo "======================================"
echo "Finish time: $(date)"
echo "LAMMPS finished successfully"
echo "LAMMPS log: $(pwd)/lammps.log"
echo "Done flag: $(pwd)/lammps_done.flag"
echo "======================================"