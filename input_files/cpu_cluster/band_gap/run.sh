#!/bin/bash
#SBATCH -p v6_384            # TODO: 改成你集群的 DPNEGF 计算分区名
#SBATCH -N 1
#SBATCH -n 96               # 必须与 run.py 里 NEGF(..., n_cpus=48) 保持一致
#SBATCH -J cnt_negf
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -t 08:00:00

set -eu
trap 'rc=$?; if (( rc )); then echo "$rc" > band_gap_failed.flag; fi' EXIT
if [[ -n "${CNT_BAND_GAP_SETUP:-}" ]]; then
    source "$CNT_BAND_GAP_SETUP"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python -u run.py __BAND_GAP_INPUTS__
test -s output/results/gap.json
touch band_gap_done.flag
