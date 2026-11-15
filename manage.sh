#!/usr/bin/env bash
WORKFLOW_DIR="/data02/run01/scv7sc4/soft/assemble_workflow"

export BATCH_PHASE="all"
export BATCH_SCHEDULER="pool"
export BATCH_PARA=16
export BATCH_POOL_CPUS=6
export BATCH_WORKER_TIME="24:00:00"

# 与原 LAMMPS run.sh 一致，先加载包含 DeePMD LAMMPS 插件的环境。
export BATCH_POOL_SETUP="$HOME/dpmd.sh"
export CNT_LAMMPS_CMD="/data02/run01/scv7sc4/test/lammps/lammps-build-gpu/lmp"
# DPNEGF local 阶段单独使用 DeePTB 环境的 Python。
export CNT_DPTB_PYTHON="/data02/run01/scv7sc4/soft/dptb-migrated/env/bin/python"

export BATCH_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_MANAGER_CMD="$0"

exec bash "$WORKFLOW_DIR/manage_batch.sh" "$@"
