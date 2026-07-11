#!/usr/bin/env python3
"""
统一日志工具（零依赖，单文件）。

设计目标：
- 全流程输出统一前缀 [LEVEL HH:MM:SS]，方便从 stdout 判断阶段与耗时。
- ERROR 走 stderr，其余走 stdout，便于 `python run.py 2>err.log` 单独抓错。
- 面向用户的状态行统一中文；[INFO]/[RUN] 这类标签作为格式符号保留英文。
- debug() 默认静默，仅在环境变量 CNT_VERBOSE=1 时打印，供底层高频函数使用。

编排层（run.py / run_multi.py，位于仓库根）与各 stage 脚本共用同一份：
    stage 脚本内直接 `import logkit`；
    编排层先把 stage 目录加入 sys.path 再 import。
"""
import os
import sys
import time

# 时间戳默认开启；如需关闭可设 CNT_LOG_TS=0
_TS_ON = os.environ.get("CNT_LOG_TS", "1") != "0"
# debug() 是否输出：默认静默
_VERBOSE = os.environ.get("CNT_VERBOSE", "0") == "1"

# 走 stderr 的级别
_STDERR_LEVELS = {"ERROR"}


def _now():
    return time.strftime("%H:%M:%S")


def log(msg, level="INFO", *, ts=None):
    """
    打印一行统一格式日志。

        ts=True  -> [LEVEL HH:MM:SS] msg
        ts=False -> [LEVEL] msg
        ts=None  -> 跟随全局默认（CNT_LOG_TS）
    """
    show_ts = _TS_ON if ts is None else ts
    tag = f"[{level} {_now()}]" if show_ts else f"[{level}]"
    stream = sys.stderr if level in _STDERR_LEVELS else sys.stdout
    print(f"{tag} {msg}", file=stream)


def info(msg, **kw):
    log(msg, "INFO", **kw)


def warn(msg, **kw):
    log(msg, "WARN", **kw)


def error(msg, **kw):
    log(msg, "ERROR", **kw)


def ok(msg, **kw):
    log(msg, "OK", **kw)


def skip(msg, **kw):
    log(msg, "SKIP", **kw)


def clean(msg, **kw):
    log(msg, "CLEAN", **kw)


def run(msg, **kw):
    """打印即将执行的命令行。"""
    log(msg, "RUN", **kw)


def debug(msg, **kw):
    """底层高频函数用；默认静默，CNT_VERBOSE=1 时才打印。"""
    if _VERBOSE:
        log(msg, "DEBUG", **kw)


def phase(n, total, title):
    """
    阶段分隔行，例如：
        ==================== [2/5] dump -> fdf ====================
    """
    head = f" [{n}/{total}] {title} "
    bar = head.center(64, "=")
    print("\n" + bar)


def item(i, total, msg):
    """
    列表内逐项进度行，例如：
          [3/8] data/300K/5_5/DV_DV/dpnegf/40000
    """
    print(f"    [{i}/{total}] {msg}")
