#!/bin/bash
set -e

echo "======================================"
echo "LAMMPS local run"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
echo "======================================"

# 如果你已经在 lammps_cpu 环境里运行，就不要在这里 conda activate
# conda activate lammps_cpu

if command -v lmp >/dev/null 2>&1; then
    LMP_CMD="lmp"
elif command -v lmp_mpi >/dev/null 2>&1; then
    LMP_CMD="lmp_mpi"
elif command -v lammps >/dev/null 2>&1; then
    LMP_CMD="lammps"
elif command -v lammps_mpi >/dev/null 2>&1; then
    LMP_CMD="lammps_mpi"
else
    echo "[ERROR] Cannot find LAMMPS executable."
    exit 1
fi

echo "Using LAMMPS command: ${LMP_CMD}"

if command -v mpirun >/dev/null 2>&1; then
    echo "Using mpirun"
    mpirun --allow-run-as-root -np 16 ${LMP_CMD} -in in.lammps > lammps.log
else
    echo "Using serial command"
    ${LMP_CMD} -in in.lammps > lammps.log
fi

echo "======================================"
echo "Finish time: $(date)"
echo "LAMMPS finished successfully"
echo "======================================"

touch lammps_done.flag