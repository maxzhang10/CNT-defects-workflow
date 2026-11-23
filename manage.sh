#!/usr/bin/env bash
set -euo pipefail

# 把 manage.sh 和 batch.py 复制到数据目录后，将 WORKFLOW_DIR 设为集群上
# scattering_slurm 仓库的绝对路径，也可以在运行前 export WORKFLOW_DIR。
data_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="${WORKFLOW_DIR:-$data_dir}"
export BATCH_SCHEDULER="${BATCH_SCHEDULER:-slurm}"  # slurm 或 local

if [[ ! -f "$WORKFLOW_DIR/manage_batch.sh" ]]; then
    echo "请在 manage.sh 或环境变量中设置 WORKFLOW_DIR，指向 scattering_slurm 软件目录。" >&2
    exit 2
fi
export BATCH_DATA_DIR="$data_dir"
exec bash "$WORKFLOW_DIR/manage_batch.sh" "$@"
