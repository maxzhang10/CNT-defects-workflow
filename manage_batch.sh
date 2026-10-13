#!/usr/bin/env bash
set -euo pipefail

# SCRIPT_DIR = 软件目录（batch_generate.py 所在处，按脚本自身位置定位）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DATA_DIR = 数据目录（batch.py、日志、pid/status 所在处）。
# 缺省 = SCRIPT_DIR，即仓库内自用时行为与以前完全一致；
# 数据文件夹通过 wrapper（manage.sh）设置 BATCH_DATA_DIR 指向自己，
# 实现"数据参数在一起、软件分开"。
DATA_DIR="${BATCH_DATA_DIR:-$SCRIPT_DIR}"
# 统一成绝对路径：手动 export 相对路径时，start/status/stop 从不同
# cwd 调用也能找到同一套 pid/log 文件（目录不存在则留给 start 守卫报错）
if [[ -d "$DATA_DIR" ]]; then
    DATA_DIR="$(cd "$DATA_DIR" && pwd)"
fi

# 提示语中显示的命令名：经 wrapper 调用时显示 wrapper 路径（如
# ./manage.sh status），避免误导用户直接运行仓库里的本脚本
# （那样会缺 BATCH_DATA_DIR，把仓库当成数据目录）。
CMD="${BATCH_MANAGER_CMD:-$0}"

PID_FILE="$DATA_DIR/batch_generate.pid"
STATUS_FILE="$DATA_DIR/batch_generate.status"

# 当前轮和上一轮 batch_generate.py 日志
LOG_FILE="$DATA_DIR/batch_generate.log"
PREV_LOG_FILE="$DATA_DIR/batch_generate.log.prev"

# manage 调度器自身日志
MANAGER_LOG="$DATA_DIR/manage_batch.log"


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
        echo "查看状态：$CMD status"
        return 0
    fi

    # 数据目录必须存在且含批配置，否则启动后立刻失败
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "数据目录不存在：$DATA_DIR"
        return 1
    fi
    if [[ ! -f "$DATA_DIR/batch.py" && ! -f "$DATA_DIR/batch.json" ]]; then
        echo "数据目录下找不到 batch.py / batch.json：$DATA_DIR"
        return 1
    fi

    # 清理失效的旧PID文件和上一轮状态文件
    # （状态文件必须清掉：下面靠"本轮是否写了新状态"区分
    # 快速完成与启动失败，残留的旧状态会造成误判）
    rm -f "$PID_FILE" "$STATUS_FILE"

    # 每次重新启动manage时，清空旧的manage日志
    : > "$MANAGER_LOG"

    nohup setsid bash -c '
        workflow_dir="$1"
        data_dir="$2"
        log_file="$3"
        prev_log_file="$4"
        status_file="$5"
        pid_file="$6"

        # 在数据目录运行：batch_generate.py 缺省探测 ./batch.py，
        # data_root 相对路径也按数据目录解析
        cd "$data_dir" || exit 1

        cleanup() {
            echo "[$(date "+%F %T")] 调度器停止，PID=$$"

            printf "%s stopped PID=%s\n" \
                "$(date "+%F %T")" "$$" > "$status_file"

            rm -f "$pid_file"
            exit 0
        }

        trap cleanup TERM INT

        echo "[$(date "+%F %T")] 调度器启动，PID=$$"
        echo "[$(date "+%F %T")] 工作目录：$data_dir"
        echo "[$(date "+%F %T")] 软件目录：$workflow_dir"

        round=1
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
        # （并行度写在批配置顶层 max_parallel，可用 --max-parallel 覆盖）
        if python "$workflow_dir/batch_generate.py" >"$log_file" 2>&1; then
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

        # 返回非0：仅记录失败、保留日志与全部任务目录，然后退出。
        # 绝不自动重试、重交或重新启动 batch_generate.py。
        echo "[$end_time] 第 ${round} 轮失败，exit=$rc"
        echo "[$end_time] 调度器退出；未自动重试或重交，请查看 $log_file"

        printf "%s completed PID=%s round=%s result=failed exit=%s auto_retry=false\n" \
            "$end_time" "$$" "$round" "$rc" > "$status_file"

        rm -f "$pid_file"
        exit "$rc"
    ' bash \
        "$SCRIPT_DIR" \
        "$DATA_DIR" \
        "$LOG_FILE" \
        "$PREV_LOG_FILE" \
        "$STATUS_FILE" \
        "$PID_FILE" \
        >>"$MANAGER_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 1

    if ! kill -0 "$pid" 2>/dev/null; then
        # 进程已退出。区分两种情况：
        # 1) 本轮写了 completed 状态 = batch_generate.py 已完整跑完一轮
        #    （成功：如无新任务秒退；失败：如配置错误秒挂），如实报告；
        # 2) 没有新状态 = 调度器自身没能启动（如 python 不存在）。
        if [[ -f "$STATUS_FILE" ]] && grep -q "completed" "$STATUS_FILE"; then
            echo "本轮已在 1 秒内结束，结果："
            cat "$STATUS_FILE"
            echo
            echo "当前日志     : $CMD log"
            echo "调度器日志   : $CMD manager-log"
            if grep -q "result=success" "$STATUS_FILE"; then
                return 0
            fi
            return 1
        fi

        echo "启动失败，请查看：$MANAGER_LOG"
        rm -f "$PID_FILE"
        return 1
    fi

    echo "批处理管理器已启动"
    echo "PID          : $pid"
    echo "数据目录     : $DATA_DIR"
    echo "软件目录     : $SCRIPT_DIR"
    echo
    echo "状态         : $CMD status"
    echo "当前日志     : $CMD log"
    echo "上一轮日志   : $PREV_LOG_FILE"
    echo "调度器日志   : $CMD manager-log"
    echo "停止         : $CMD stop"
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
        echo "用法：$CMD {start|status|stop|restart|log|previous-log|manager-log}"
        echo
        echo "  start        启动一次 batch_generate.py；成功或失败后自动退出"
        echo "  status       查看运行状态"
        echo "  stop         停止本地调度器"
        echo "  restart      重启本地调度器"
        echo "  log          实时查看当前轮日志"
        echo "  previous-log 查看上一轮日志"
        echo "  manager-log  实时查看调度器日志"
        echo
        echo "数据目录：$DATA_DIR（设 BATCH_DATA_DIR 可指向其他数据文件夹，"
        echo "或直接用数据文件夹里的 manage.sh wrapper）"
        exit 2
        ;;
esac