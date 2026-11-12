#!/usr/bin/env bash
# 与 batch.py 放在数据目录中，使用 ./manage.sh start / status / stop / log。

WORKFLOW_DIR="/personal/CNT_defects_package_slurm"

# LAMMPS 资源池：最多占用 BATCH_PARA 张 GPU。
export BATCH_PHASE="lammps"
export BATCH_SCHEDULER="pool"
export BATCH_PARA=4

# 复用原 run.sh 的环境和 LAMMPS 程序；由计算节点加载环境。
export BATCH_POOL_SETUP="$HOME/dpmd.sh"
export CNT_LAMMPS_CMD="/data02/run01/scv7sc4/test/lammps/lammps-build-gpu/lmp"

export BATCH_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_MANAGER_CMD="$0"

exec bash "$WORKFLOW_DIR/manage_batch.sh" "$@"
