#!/usr/bin/env bash
# manage.sh —— 数据文件夹 wrapper（模板）
#
# 用法：把本文件和 batch.py 一起放进任意数据文件夹，然后：
#     chmod +x manage.sh
#     ./manage.sh start        # 以及 status / stop / log / ...
#
# 数据（batch.py、日志、pid/status、record.log、replica 结果）全部
# 留在本文件夹；工作流软件始终调用下面 WORKFLOW_DIR 指向的仓库副本。
# 阶段设置由此文件导出；已有旧 wrapper 可补上 BATCH_PHASE。

WORKFLOW_DIR="/personal/CNT_defects_package_slurm"

# A: lammps (includes FDF/XYZ conversion); B: dpnegf; legacy: all.
export BATCH_PHASE="lammps"

export BATCH_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_MANAGER_CMD="$0"

exec bash "$WORKFLOW_DIR/manage_batch.sh" "$@"
