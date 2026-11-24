#!/usr/bin/env bash
#SBATCH -J cnt_band_gap
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eu
trap 'rc=$?; if (( rc )); then echo "$rc" > band_gap_failed.flag; fi' EXIT
if [[ -n "${CNT_BAND_GAP_SETUP:-}" ]]; then
    source "$CNT_BAND_GAP_SETUP"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python -u run.py
test -s output/results/gap.json
touch band_gap_done.flag
