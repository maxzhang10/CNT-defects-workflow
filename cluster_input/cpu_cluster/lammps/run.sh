#!/bin/bash
#SBATCH --partition=v6_384
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --job-name=cnt_lmp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

# 清理本次运行的旧状态
rm -f lammps_done.flag
rm -f lammps_failed.flag
rm -f lammps_started.flag

on_exit() {
    rc=$?

    if (( rc != 0 )); then
        {
            echo "Failed at: $(date '+%F %T')"
            echo "Return code: ${rc}"
            echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
            echo "Host: $(hostname)"
        } > lammps_failed.flag

        echo "[ERROR] LAMMPS failed, return code=${rc}" >&2
    fi
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source /public5/soft/modules/module.sh
module load deepmdkit/v3.2.0b0_pytorch

# DeePMD/PyTorch CPU 线程设置
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DP_INTER_OP_PARALLELISM_THREADS=1

date '+%F %T' > lammps_started.flag

echo "======================================"
echo "LAMMPS Slurm run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "DP_INTRA_OP_PARALLELISM_THREADS=${DP_INTRA_OP_PARALLELISM_THREADS}"
echo "DP_INTER_OP_PARALLELISM_THREADS=${DP_INTER_OP_PARALLELISM_THREADS}"
echo

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
echo "======================================"

mpirun -np "${NPROC}" \
    "${LMP_CMD}" \
    -in in.lammps \
    > lammps.log 2>&1

date '+%F %T' > lammps_done.flag
rm -f lammps_failed.flag

echo "======================================"
echo "Finish time: $(date)"
echo "LAMMPS finished successfully"
echo "======================================"