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
module load lammps/oneAPI.2022.1/7Feb2024-no_lib

export OMP_NUM_THREADS=1

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
echo
echo "LAMMPS executable:"
which lmp_intel_cpu_intelmpi
echo "======================================"

srun \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    lmp_intel_cpu_intelmpi \
    -in in.lammps \
    -log lammps.log

date '+%F %T' > lammps_done.flag
rm -f lammps_failed.flag

echo "======================================"
echo "Finish time: $(date)"
echo "LAMMPS finished successfully"
echo "======================================"