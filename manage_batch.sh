#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PID_FILE="$SCRIPT_DIR/batch_generate.pid"
STATUS_FILE="$SCRIPT_DIR/batch_generate.status"

# 当前轮和上一轮 batch_generate.py 日志
LOG_FILE="$SCRIPT_DIR/batch_generate.log"
PREV_LOG_FILE="$SCRIPT_DIR/batch_generate.log.prev"

# manage 调度器自身日志
MANAGER_LOG="$SCRIPT_DIR/manage_batch.log"

# batch_generate.py 失败后，等待10分钟再次运行
RETRY_INTERVAL=600


is_running() {
    [[ -f "$PID_FILE" ]] || return 1

    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}


start_job() {
    if is_running; then
        echo "任务已经运行，PID=$(cat "$PID_FILE")"
        echo "查看状态：$0 status"
        return 0
    fi

    # 清理失效的旧PID文件
    rm -f "$PID_FILE"

    # 每次重新启动manage时，清空旧的manage日志
    : > "$MANAGER_LOG"

    nohup setsid bash -c '
        script_dir="$1"
        log_file="$2"
        prev_log_file="$3"
        status_file="$4"
        retry_interval="$5"
        pid_file="$6"

        cd "$script_dir" || exit 1

        cleanup() {
            echo "[$(date "+%F %T")] 调度器停止，PID=$$"

            printf "%s stopped PID=%s\n" \
                "$(date "+%F %T")" "$$" > "$status_file"

            rm -f "$pid_file"
            exit 0
        }

        trap cleanup TERM INT

        echo "[$(date "+%F %T")] 调度器启动，PID=$$"
        echo "[$(date "+%F %T")] 工作目录：$script_dir"
        echo "[$(date "+%F %T")] 失败重试间隔：${retry_interval}s"

        round=0

        while true; do
            round=$((round + 1))
            start_time="$(date "+%F %T")"

            printf "%s running PID=%s round=%s\n" \
                "$start_time" "$$" "$round" > "$status_file"

            echo
            echo "============================================================"
            echo "[$start_time] 第 ${round} 轮运行 batch_generate.py"
            echo "============================================================"

            # 当前轮日志转为上一轮日志
            if [[ -f "$log_file" ]]; then
                mv -f "$log_file" "$prev_log_file"
            fi

            # 创建新的当前轮日志
            : > "$log_file"

            echo "[$(date "+%F %T")] 当前轮日志：$log_file"

            # manage不控制并行参数，完全交给batch_generate.py自身配置
            if python batch_generate.py >"$log_file" 2>&1; then
                rc=0
            else
                rc=$?
            fi

            end_time="$(date "+%F %T")"

            # 返回0：工作流完成，调度器自动退出
            if [[ "$rc" -eq 0 ]]; then
                echo "[$end_time] 第 ${round} 轮正常结束，全部任务完成"
                echo "[$end_time] 调度器自动退出"

                printf "%s completed PID=%s round=%s result=success\n" \
                    "$end_time" "$$" "$round" > "$status_file"

                rm -f "$pid_file"
                exit 0
            fi

            # 返回非0：等待一段时间后重试
            echo "[$end_time] 第 ${round} 轮失败，exit=$rc"
            echo "[$end_time] ${retry_interval} 秒后重新运行"

            printf "%s sleeping PID=%s round=%s result=failed next_retry_in=%ss\n" \
                "$end_time" "$$" "$round" "$retry_interval" \
                > "$status_file"

            sleep "$retry_interval"
        done
    ' bash \
        "$SCRIPT_DIR" \
        "$LOG_FILE" \
        "$PREV_LOG_FILE" \
        "$STATUS_FILE" \
        "$RETRY_INTERVAL" \
        "$PID_FILE" \
        >>"$MANAGER_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "启动失败，请查看：$MANAGER_LOG"
        rm -f "$PID_FILE"
        return 1
    fi

    echo "自动检查任务已启动"
    echo "PID          : $pid"
    echo "失败重试间隔: ${RETRY_INTERVAL} 秒"
    echo
    echo "状态         : $0 status"
    echo "当前日志     : $0 log"
    echo "上一轮日志   : $PREV_LOG_FILE"
    echo "调度器日志   : $0 manager-log"
    echo "停止         : $0 stop"
}


status_job() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"

        echo "调度器正在运行："
        ps -o pid,ppid,pgid,stat,etime,pcpu,pmem,args -p "$pid"

        echo
        echo "调度器及其子进程："
        ps -o pid,ppid,pgid,stat,etime,pcpu,pmem,args \
            --forest -g "$pid" 2>/dev/null || true

        if [[ -f "$STATUS_FILE" ]]; then
            echo
            echo "当前状态："
            cat "$STATUS_FILE"
        fi

        echo
        echo "Slurm作业："
        squeue -u "$USER" 2>/dev/null || true

        echo
        echo "当前轮最近日志："
        tail -n 20 "$LOG_FILE" 2>/dev/null || true
    else
        echo "调度器未运行"

        if [[ -f "$PID_FILE" ]]; then
            echo "清理失效PID文件：$(cat "$PID_FILE")"
            rm -f "$PID_FILE"
        fi

        if [[ -f "$STATUS_FILE" ]]; then
            echo
            echo "最后状态："
            cat "$STATUS_FILE"
        fi

        return 1
    fi
}


stop_job() {
    if ! is_running; then
        echo "调度器未运行"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"

    echo "正在停止进程组 PGID=$pid"

    # 停止调度器及其本地子进程
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

    printf "%s stopped\n" "$(date "+%F %T")" > "$STATUS_FILE"

    echo "调度器已停止"
    echo "注意：已经提交到Slurm的作业不会被自动取消。"
}


show_log() {
    touch "$LOG_FILE"
    tail -F "$LOG_FILE"
}


show_previous_log() {
    if [[ ! -f "$PREV_LOG_FILE" ]]; then
        echo "暂时没有上一轮日志：$PREV_LOG_FILE"
        return 1
    fi

    less +G "$PREV_LOG_FILE"
}


show_manager_log() {
    touch "$MANAGER_LOG"
    tail -F "$MANAGER_LOG"
}


case "${1:-start}" in
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

    log)
        show_log
        ;;

    previous-log)
        show_previous_log
        ;;

    manager-log)
        show_manager_log
        ;;

    *)
        echo "用法：$0 {start|status|stop|restart|log|previous-log|manager-log}"
        echo
        echo "  start        启动自动检查任务"
        echo "  status       查看运行状态"
        echo "  stop         停止本地调度器"
        echo "  restart      重启本地调度器"
        echo "  log          实时查看当前轮日志"
        echo "  previous-log 查看上一轮日志"
        echo "  manager-log  实时查看调度器日志"
        exit 2
        ;;
esac