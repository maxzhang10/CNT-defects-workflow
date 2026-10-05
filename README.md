# CNT 缺陷输运工作流

本仓库用于批量生成 CNT 缺陷结构，运行 LAMMPS 分子动力学或结构优化，生成 DPNEGF 输入并计算输运性质。批量入口是 `batch_generate.py`，单个目录入口是 `run_multi.py`。

## 1. 运行前准备

计算节点需要具备：

- Python 环境及仓库依赖（DPNEGF、DeePTB、PyTorch 等）；
- LAMMPS、MPI；
- 使用 Slurm 时可用的 `sbatch`、`squeue`、`scancel`；
- 与 `input_files/` 或 `cluster_input/` 模板匹配的 `run.sh`、模型文件和输入文件。

`input_files/` 是默认模板，`cluster_input/` 保存集群相关模板。脚本中的 `run.sh` 会加载节点上的 `~/dpmd.sh` 或 `~/dptb.sh`，如集群环境不同，需要先修改对应模板。

## 2. 批量运行（推荐）

批量参数写在 `batch.json` 或 `batch.py` 中。默认探测顺序是当前目录下的 `batch.py`、`batch.json`，也可以用 `--config` 显式指定。

### 最小示例

当前示例配置可直接检查：

```bash
python batch_generate.py --config batch.json --scheduler slurm --max-parallel 2
```

正式运行建议保存日志：

```bash
nohup python batch_generate.py \
  --config batch.json \
  --scheduler slurm \
  --max-parallel 20 \
  > batch_generate.out 2>&1 &
```

调试时可使用本地串行调度，或只打印命令：

```bash
python batch_generate.py --config batch.json --scheduler local --max-parallel 1
python batch_generate.py --config batch.json --scheduler slurm --max-parallel 2 --skip-conductance
```

`--max-parallel` 是同时运行的独立 workflow 数；Slurm 模式下每个 workflow 内部会按 LAMMPS、DPNEGF 阶段提交作业并等待阶段结束。所有 workflow 成功后，默认自动进行跨 replica 电导汇总；`--skip-conductance` 可跳过该步骤。

### 配置字段

`batch.json` 的核心结构如下：

```json
{
  "template": {
    "r_max": 6.5,
    "md_steps": 50000,
    "save_self_energy": false,
    "self_energy_cache": {
      "use_saved": true,
      "save_path": "{data_root}/self_energy/{chirality}/"
    },
    "md_sampling": {"n_samples": 1}
  },
  "temperatures": [500],
  "lammps_mode": "opt",
  "base_structure_seed": 20260705,
  "base_lammps_seed": 23456789,
  "data_root": ".",
  "configs": [
    {
      "chirality": [5, 5],
      "l_def": 8,
      "N_defects": 2,
      "structures": ["5775"],
      "replicas": 2
    }
  ]
}
```

说明：

- `chirality`、`l_def`、`N_defects`、`structures`、`replicas` 是每个配置的必需字段；
- `lammps_mode` 为 `opt`（结构优化）或 `md`（分子动力学）；
- 电导模式由手性自动判断：`(m - n) % 3 == 0` 的金属管使用 `fermi`，其余半导体管使用 `band_edge_bias`；
- 每个配置单独设置 `replicas`，不同长度可以使用不同 replica 数；
- 两个 base seed 分别控制缺陷结构和 LAMMPS 轨迹。相同配置的 replica 序号会得到稳定的 seed；
- `self_energy_cache.save_path` 支持 `{m}`、`{n}`、`{chirality}`、`{data_root}` 占位符。若多个 replica 共享自能，建议使用包含 `{data_root}` 的路径。

需要循环、条件或批量生成多个配置时，可使用 `batch.py`。它必须导出与 `batch.json` 相同的配置字段，具体格式和校验规则见 `stage/batch_config.py` 顶部说明。

## 3. 批量目录和输出

每个任务写入：

```text
<data_root>/<temperature>K/<m>_<n>/<defect_name>/replica_XXX/
├── lammps/
├── dpnegf/<md_step>/
├── workflow_config.json
└── workflow.log
```

缺陷目录名由结构类型、缺陷长度和密度组成，例如 `5775_L008_0.1016A-1`。批次根目录下的 `record.log` 记录每个 replica 的完成进度。DPNEGF 工作目录中的常见状态文件包括：

- `lammps_done.flag` / `lammps_failed.flag`；
- `dpnegf_done.flag` / `dpnegf_failed.flag`；
- `job_id.txt`、`slurm-<jobid>.out`、`slurm-<jobid>.err`；
- `output/negf.out.pth` 和电导结果文件。

任务参数以对应 replica 目录内的 `workflow_config.json` 为准。结果检查可运行：

```bash
./check.sh
```

## 4. 管理批处理进程

`manage_batch.sh` 适合让批量脚本在后台运行。它管理的是本地 Python 调度器，不会自动取消已经提交到 Slurm 的作业。

```bash
./manage_batch.sh start
./manage_batch.sh status
./manage_batch.sh log
./manage_batch.sh manager-log
./manage_batch.sh stop
./manage_batch.sh restart
```

日志文件包括 `batch_generate.log`、`batch_generate.log.prev` 和 `manage_batch.log`。批次成功或失败后调度器都会退出，不会自动重试。

## 5. 单个结构或已有数据的工作流

`run_multi.py` 会执行以下阶段：生成缺陷结构（eledefects）→ 运行 LAMMPS → dump 转 FDF/XYZ → 准备 DPNEGF → 运行 DPNEGF → 汇总电导。

对已有数据目录运行：

```bash
python run_multi.py --root ./data --scheduler slurm
```

常用选项：

```bash
python run_multi.py --root ./data --scheduler local
python run_multi.py --root ./data --skip-ele
python run_multi.py --root ./data --skip-dpnegf
python run_multi.py --root ./data --skip-conductance
python run_multi.py --root ./data --dry-run
python run_multi.py --root ./data --config ./config.json
```

`run.py` 是不包含最终跨 replica 汇总的单目录版本，参数与 `run_multi.py` 的基础选项相同：

```bash
python run.py --root ./data/500K/5_5/5775_L008_0.1016A-1/replica_001
```

`run.py` 只负责生成并执行单个目录的后处理流程；需要选择 `local` 或 `slurm` 调度方式时使用 `run_multi.py`。

## 6. 失败任务重提

先检查，不修改任何任务：

```bash
python resubmit_negf.py ./data
```

确认失败目录后再重置 `output/`、清理旧状态并重新提交：

```bash
python resubmit_negf.py ./data --submit --max-submit 10 --scheduler slurm
```

该脚本默认使用 Slurm；本地调试可以指定 `--scheduler local`。重提前应先查看对应目录的 `workflow.log`、`slurm-*.err` 和 `dpnegf_failed.flag`。如果修改了输入配置或模型，旧的 `output/` 和 done flag 不能直接复用。

## 7. 常用维护命令

```bash
# 查看当前用户的 Slurm 作业
squeue -u "$USER"

# 按仓库提供的作业名取消作业
./cancel.sh

# 删除共享自能缓存（执行前确认路径）
./rm_self_energy.sh
```

不要让两个并发批次使用相同的 `data_root`，否则 replica 目录和 `record.log` 会相互覆盖。修改 Slurm 分区、GPU、CPU 或环境加载方式时，应同步修改相应的 `run.sh` 模板。
