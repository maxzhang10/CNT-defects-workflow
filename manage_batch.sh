#!/usr/bin/env bash
set -euo pipefail

workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "${BATCH_DATA_DIR:-$workflow_dir}"
data_dir="$(pwd -P)"
pid_file=batch_generate.pid
status_file=batch_generate.status
log=batch_generate.log
manager_log=manage_batch.log

write_status() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" > "$status_file.tmp"
    mv -f -- "$status_file.tmp" "$status_file"
}

run_batch() {
    trap 'write_status "stopped PID=$$"; rm -f -- "$pid_file"; exit 0' TERM INT
    echo "$$" > "$pid_file"
    write_status "running PID=$$"
    echo "数据目录 $data_dir"
    echo "工作流目录 $workflow_dir"
    if [[ -f "$log" ]]; then mv -f -- "$log" "$log.prev"; fi
    local rc=0
    python -u "$workflow_dir/batch_generate.py" --workflow-dir "$workflow_dir" \
        --batch-config "$data_dir/batch.py" --scheduler "${BATCH_SCHEDULER:-slurm}" \
        > "$log" 2>&1 || rc=$?
    if (( rc == 0 )); then
        write_status "completed PID=$$ result=success"
    else
        write_status "completed PID=$$ result=failed exit=$rc"
    fi
    cat -- "$status_file"
    rm -f -- "$pid_file"
    return "$rc"
}

# 只管理本数据目录的 setsid 会话，避免旧 PID 被其他进程复用。
running_pid() {
    local pid
    [[ -f "$pid_file" ]] && read -r pid < "$pid_file" || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ "$(readlink "/proc/$pid/cwd")" == "$data_dir" ]] || return 1
    [[ "$(ps -o sid= -p "$pid")" =~ ^[[:space:]]*$pid$ ]] || return 1
    grep -Fxzq -- '--run-batch' "/proc/$pid/cmdline" || return 1
    echo "$pid"
}

start_job() {
    local pid
    if pid="$(running_pid)"; then echo "批处理已在运行，PID=$pid"; return; fi
    [[ -f batch.py ]] || { echo "找不到 $data_dir/batch.py" >&2; return 1; }
    case "${BATCH_SCHEDULER:-slurm}" in
        local|slurm) ;;
        *) echo "BATCH_SCHEDULER 仅支持 local 或 slurm" >&2; return 2 ;;
    esac
    rm -f -- "$pid_file" "$status_file"
    nohup setsid bash "$workflow_dir/manage_batch.sh" --run-batch \
        9>&- </dev/null > "$manager_log" 2>&1 &
    pid=$!
    sleep 1
    if running_pid >/dev/null; then
        echo "批处理已启动，PID=$pid，日志 $data_dir/$log"
    elif wait "$pid"; then
        cat -- "$status_file"
    else
        rm -f -- "$pid_file"
        echo "启动或计算失败，请查看 $data_dir/$manager_log 和 $data_dir/$log" >&2
        return 1
    fi
}

stop_job() {
    local pid
    if ! pid="$(running_pid)"; then echo "调度器未运行"; return 0; fi
    kill -TERM -- "-$pid" || return 1
    for _ in {1..10}; do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid"; fi
    rm -f -- "$pid_file"
    write_status "stopped PID=$pid"
    echo "已停止本地调度器。已提交的 Slurm 作业仍由 Slurm 管理。"
}

usage() {
    echo "用法：bash manage.sh {start|status|stop|restart|log|previous-log|manager-log}"
}

case "${1:-start}" in
    --run-batch) run_batch ;;
    start|stop|restart)
        # 同一数据目录的启动和停止操作串行；后台进程关闭锁描述符。
        exec 9>>.manage_batch.lock
        flock -x 9
        case "${1:-start}" in
            start) start_job ;;
            stop) stop_job ;;
            restart) stop_job; start_job ;;
        esac ;;
    status)
        if pid="$(running_pid)"; then
            echo "批处理运行中，PID=$pid"
            tail -n 20 -- "$log" 2>/dev/null || true
        else
            echo "调度器未运行"
            [[ ! -f "$status_file" ]] || cat -- "$status_file"
            exit 1
        fi ;;
    log) tail -n 20 -F -- "$log" ;;
    previous-log) cat -- "$log.prev" ;;
    manager-log) tail -n 20 -F -- "$manager_log" ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
