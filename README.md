# CNT 缺陷工作流批量运行说明

本文档介绍 `manage_batch.sh`（批处理管理器）与 `batch_generate.py` 的用法。

## 快速开始

```bash
# 启动一个批次（前台等待，建议 nohup 挂后台并保存输出）
nohup ./manage_batch.sh batch_generate.py start > batch_generate.out 2>&1 &

# 查看状态 / 停止
./manage_batch.sh batch_generate.py status
./manage_batch.sh batch_generate.py stop
```

批次参数（并发数、replica 数、数据目录）通过 `start` 后面的参数透传给
batch 脚本：

```bash
./manage_batch.sh batch_generate.py start --lammps-repeats 20 --max-parallel 50 --data-root ./
```

## 多批次并发：用不同的脚本名区分实例

`manage_batch.sh` 以 **batch 脚本名**作为实例标识。想跑第二个批次，
复制一份脚本、改个名字（改里面的配置也行），用各自的脚本名管理：

```bash
cp batch_generate.py batch_generate_1.py
cp batch_generate.py batch_generate_2.py

# 两个批次并发运行
nohup ./manage_batch.sh batch_generate_1.py start --data-root ./data_1 --lammps-repeats 20 > b1.out 2>&1 &
nohup ./manage_batch.sh batch_generate_2.py start --data-root ./data_2 --lammps-repeats 20 > b2.out 2>&1 &

# 分别管理
./manage_batch.sh batch_generate_2.py status
./manage_batch.sh batch_generate_2.py stop
```

脚本基名只能包含字母、数字、下划线和连字符。

**注意**：并发实例必须使用不同的 `--data-root`，否则 replica 目录重叠会互相
写坏；所有实例的 `--max-parallel` 之和不要超过 Slurm 队列配额。

## 产生的文件（刻意保持最少）

manage 自身**只生成一个 `<基名>.pids`**，记录运行中的进程号（供
status/stop 使用），批次结束后自动删除。不再生成
`.log / .status / .manager.log / .log.prev` 等文件。

batch 的输出直接回显到终端，想留档就在启动时重定向（如上例的
`> b1.out`）。其余信息都在数据目录里：

- **进度**：`<data_root>/record.log`（每个 replica 完成一行 + 汇总）；
- **细节**：每个 replica 目录下的 `workflow.log`（完整内部日志含报错）；
- **参数**：每个 replica 目录下的 `workflow_config.json`（启动时落盘的
  完整配置，参数溯源以它为准）。

## 命令一览

```
manage_batch.sh <batch脚本.py> <命令> [batch参数...]

  start        启动一次该 batch 脚本；成功或失败后自动退出（不自动重试）
  status       查看运行状态（进程树 + Slurm 队列）
  stop         停止本地调度器及 batch 进程（已提交的 Slurm 作业不会被取消）
  restart      重启
```

## batch_generate.py 常用参数

```
python batch_generate.py \
    --scheduler slurm          # 或 local
    --lammps-repeats 20        # 每个物理配置的 replica（种子）数
    --max-parallel 50          # 同时启动的 workflow 数
    --data-root ./             # 数据根目录
    --skip-conductance         # 跳过批次结束后的跨 replica 电导汇总
```

物理配置（手性、缺陷类型、缺陷数、温度等）在 `batch_generate.py` 顶部的
`CHIRAL_CONFIGS` / `TEMPERATURES` 中定义；每个配置会展开成
`--lammps-repeats` 个 replica，写入
`data_root/<温度>K/<m>_<n>/<缺陷配置>/replica_XXX/`。

## record.log：replica 进度记录

每次批次运行时，batch 脚本会在 `data_root/record.log` 追加记录每个
物理配置的 replica 完成进度：

```
===== batch start 2026-10-02 14:01:53 (20 tasks, 1 configurations) =====
[2026-10-02 14:05:12] 500K/5_5/5775_L008_0.1016A-1: 1/20 done (500K_5_5_5775_L008_0.1016A-1_rep003 OK)
[2026-10-02 14:06:40] 500K/5_5/5775_L008_0.1016A-1: 2/20 done (500K_5_5_5775_L008_0.1016A-1_rep001 FAIL)
...
----- batch end 2026-10-02 16:20:00 (success 19/20) -----
  500K/5_5/5775_L008_0.1016A-1: 20/20 done
```

- 每个 replica 的 workflow 跑完一轮（无论 OK 还是 FAIL）都会追加一行；
- 结尾汇总块给出每个配置的最终 `完成数/总数` 与整批成功数；
- 文件为追加模式，同一 data_root 多次批次不会清掉历史记录；
- 每个并发实例的 record.log 写在各自的 `--data-root` 下。

## 参数溯源

想知道某次运行用了什么参数，按此顺序查：

1. **每个 replica 目录的 `workflow_config.json`**（最准确）：任务启动时实际
   落盘的完整配置，不会被后续修改 batch 脚本影响；
2. **启动时保存的输出**（如 `b1.out`）：开头有总任务数、replica 数、并发、
   scheduler、data_root；
3. **运行中的进程命令行**：`status` 拿到 PID 后 `ps -p <PID> -o args`。

## 失败处理

`manage_batch.sh` **不会自动重试**：batch 脚本返回非 0 时直接结束，
失败任务的目录与日志全部保留。排查方式：

1. 看启动输出末尾的失败任务列表（或 `record.log` 里的 FAIL 行）；
2. 每个 replica 的详细日志在其目录下的 `workflow.log`；
3. 修复后手动重跑（注意：若 config 有变更，旧结果的 provenance hash 不一致
   会触发 P0-4 保护，需删除对应 `dpnegf_done.flag` 与 `output/` 再重算）。
