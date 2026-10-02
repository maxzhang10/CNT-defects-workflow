#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 用法：manage_batch.sh <batch脚本> <命令> [batch参数...]
# 例如：
#   manage_batch.sh batch_generate_1.py start --data-root ./data_1 --lammps-repeats 20
#   manage_batch.sh batch_generate_2.py status
#   manage_batch.sh batch_generate_2.py stop
# 以 batch 脚本名区分实例，同一文件夹可并发运行多个不同名字的 batch 脚本。
#
# 文件策略（尽量减少中间文件）：
#   只生成 <基名>.pids 一个文件，记录运行中的进程号（供 stop/status 使用），
#   批次结束后自动删除。不再生成 .log/.status/.manager.log 等文件：
#   batch 输出直接回显到终端（建议 nohup 或 tee 保存），
#   进度记录在 <data_root>/record.log，细节在各 replica 的 workflow.log。

usage() {
    echo "用法："
    echo "  $0 <batch脚本.py> <命令> [batch参数...]"
    echo
    echo "  命令: start | status | stop | restart"
    echo
    echo "  start        启动一次该 batch 脚本；成功或失败后自动退出（不自动重试）"
    echo "  status       查看运行状态（进程树 + Slurm 队列）"
    echo "  stop         停止本地调度器及 batch 进程（已提交的 Slurm 作业不受影响）"
    echo "  restart      重启"
    echo
    echo "示例："
    echo "  nohup $0 batch_generate.py start > batch_generate.out 2>&1 &"
    echo "  $0 batch_generate_2.py start --data-root ./data_2 --lammps-repeats 20"
    echo "  $0 batch_generate_2.py status"
    echo
    echo "说明：只生成 <基名>.pids 记录运行中的进程，结束后自动删除；"
    echo "      进度见 <data_root>/record.log，细节见各 replica 的 workflow.log。"
}

BATCH_SCRIPT="${1:-}"
if [[ -z "$BATCH_SCRIPT" || "$BATCH_SCRIPT" == "-h" || "$BATCH_SCRIPT" == "--help" ]]; then
    usage
    exit $([[ -z "$BATCH_SCRIPT" ]] && echo 2 || echo 0)
fi
if [[ $# -gt 0 ]]; then
    shift
fi

# 统一解析到 manage 所在目录
BATCH_NAME="$(basename "$BATCH_SCRIPT")"
BATCH_PATH="$SCRIPT_DIR/$BATCH_NAME"
if [[ ! -f "$BATCH_PATH" ]]; then
    echo "错误：找不到 batch 脚本: $BATCH_PATH"
    exit 2
fi
if [[ "$BATCH_NAME" != *.py ]]; then
    echo "错误：batch 脚本必须是 .py 文件: $BATCH_NAME"
    exit 2
fi

# 实例标识 = 脚本基名（去掉 .py）
BASE="${BATCH_NAME%.py}"
if [[ ! "$BASE" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "错误：脚本基名只能包含字母、数字、下划线和连字符: $BASE"
    exit 2
fi

CMD="${1:-}"
if [[ ! "$CMD" =~ ^(start|status|stop|restart)$ ]]; then
    echo "错误：缺少有效命令，当前: ${CMD:-<空>}"
    echo
    usage
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi
EXTRA_ARGS=("$@")

if [[ "$CMD" != "start" && "$CMD" != "restart" && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "错误：命令 [$CMD] 不接受额外参数: ${EXTRA_ARGS[*]}"
    exit 2
fi

PID_FILE="$SCRIPT_DIR/${BASE}.pids"


is_running() {
    [[ -f "$PID_FILE" ]] || return 1

    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}


self_cmd() {
    echo "$0 $BATCH_NAME"
}


start_job() {
    if is_running; then
        echo "实例 [$BATCH_NAME] 已经运行，PID=$(cat "$PID_FILE")"
        echo "查看状态：$(self_cmd) status"
        return 1
    fi

    # 清理失效的旧 pid 文件
    rm -f "$PID_FILE"

    # setsid 让 batch 进程自成进程组，stop 时可整组终止；
    # 输出不回收到 manage，直接随调用者的终端/nohup 走。
    setsid python "$BATCH_PATH" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    echo "[$(date "+%F %T")] 实例 [$BATCH_NAME] 已启动，PID=$pid"
    echo "  进度：$(self_cmd) status"
    echo "  停止：$(self_cmd) stop"

    # 前台等待 batch 结束，并清理 pid 文件
    wait "$pid" || true
    rm -f "$PID_FILE"
    echo "[$(date "+%F %T")] 实例 [$BATCH_NAME] 已结束（PID=$pid）"
}


status_job() {
    echo "实例 [$BATCH_NAME]："
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"

        echo "正在运行："
        ps -o pid,ppid,pgid,stat,etime,pcpu,pmem,args -p "$pid"

        echo
        echo "进程树："
        ps -o pid,ppid,pgid,stat,etime,pcpu,pmem,args \
            --forest -g "$pid" 2>/dev/null || true

        echo
        echo "Slurm 作业："
        squeue -u "$USER" 2>/dev/null || true
    else
        echo "未运行"
        rm -f "$PID_FILE"
        return 1
    fi
}


stop_job() {
    if ! is_running; then
        echo "实例 [$BATCH_NAME] 未运行"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"

    echo "正在停止实例 [$BATCH_NAME]，进程组 PGID=$pid"

    # 停止 batch 及其本地子进程（整组）
    kill -TERM -- "-$pid" 2>/dev/null || true

    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "普通停止失败，强制终止进程组"
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"

    echo "实例 [$BATCH_NAME] 已停止"
    echo "注意：已经提交到 Slurm 的作业不会被自动取消。"
}


case "$CMD" in
    start)
        start_job
        ;;

    status)
        status_job
        ;;

    stop)
        stop_job
        ;;

    restart)
        stop_job
        start_job
        ;;
esac
