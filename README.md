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
    "defect_count_mode": "balanced_count",
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
      "length_angstrom": 10.0,
      "defect_density": 0.051210,
      "structures": ["5775"],
      "replicas": 20
    }
  ]
}
```

说明：

- `chirality`、`structures`、`replicas` 是每个配置的必需字段；长度维度必须从 `l_def` / `length_angstrom` 中选择一个，缺陷维度必须从 `N_defects` / `defect_density` 中选择一个；
- `buffer_pl` 是每侧缓冲层厚度，单位为 PL，可在每个 `configs` 配置中单独指定；省略时沿用按手性选择的默认值（金属管 2 PL，半导体管 4 PL）；
- `lammps_mode` 为 `opt`（结构优化）或 `md`（分子动力学）；
- 使用 `defect_density` 时，`defect_count_mode` 可设为 `same_count` 或 `balanced_count`。缺省 `same_count` 保持旧行为：所有 replica 使用同一个四舍五入后的缺陷数（至少 1 个）；推荐的 `balanced_count` 在相邻两个整数间分配，使整个 replica ensemble 的平均密度逼近目标密度，并允许短管 replica 为 0 缺陷；
- 电导模式由手性自动判断：`(m - n) % 3 == 0` 的金属管使用 `fermi`，其余半导体管使用 `band_edge_bias`；
- 每个配置单独设置 `replicas`，不同长度可以使用不同 replica 数；
- 两个 base seed 分别控制缺陷结构和 LAMMPS 轨迹。相同配置的 replica 序号会得到稳定的 seed；
- `self_energy_cache.save_path` 支持 `{m}`、`{n}`、`{chirality}`、`{data_root}` 占位符。若多个 replica 共享自能，建议使用包含 `{data_root}` 的路径。

### 使用结构优化后的 UC

批配置的 `template` 或单个 `configs` 项可以设置 `"uc_file": "uc_library/{chirality}/POSCAR"`（也支持 `{m}`、`{n}`）。路径相对批配置文件解析；未设置时继续由 ASE 生成 UC。文件须是单个、纯碳 CNT 原胞的 VASP5 POSCAR（元素行为 `C`）：CNT 轴沿正 z，轴向晶格向量不得倾斜，原子数应与所选手性的一 UC 一致；z 方向必须是实际周期，不能包含真空层。工作流从文件读取原子坐标和轴向晶格常数，统一用于扩胞、目标长度与密度换算、PL 划分以及电极原子范围。启动批次时会记录文件 SHA-256，运行期间若文件内容被改动则报错；外部 UC 的配置目录会带有哈希后缀，避免与 ASE 结构或其他 UC 混合。

仓库当前的 `batch.py` 在 `(8,2)` 的 `CHIRAL_CONFIGS` 项中设置 `"uc_file": "POSCAR"`，只生成这一手性的任务。它沿用原 `(12,3)` 的长度、缺陷数与 replica 网格，共 15 组配置、740 个 replica；运行前可直接修改 `batch.py` 缩小批量。根目录的 `POSCAR` 必须随配置一同保留。

推荐先独立建立和审核按手性组织的 UC 库，再让普通批次与 `self_energy_generate.py` 都引用同一文件。UC 库只描述优化后的结构和晶格常数；自能另按 UC 内容、电子温度 `ele_T`、模型、截断半径、能量网格等计算参数管理缓存。尤其当多个任务共享绝对路径的自能缓存时，应按这些参数分开目录，不应仅以手性命名。外部 UC 使用共享缓存时，工作流要求 `save_path` 含 `{uc_sha256}` 和 `{ele_T}`，并显式设置 `ele_T`，例如 `"{data_root}/self_energy/{chirality}/{uc_sha256}/eleT_{ele_T}/"`。结构的 DFT 优化条件可记录在 UC 库旁的元数据中，不作为运行时的 `ele_T`。

### 长度、密度与缺陷数换算

ASE 模式下，手性 `(m,n)` 的轴向 UC 长度由下式确定；外部 UC 模式直接使用 POSCAR 的轴向晶格常数 `T`。随后两种模式使用同一套长度换算：

```text
T = 3 × 1.42 × sqrt(m² + mn + n²) / gcd(2m+n, 2n+m)  Å
l_def = max(1, floor(length_angstrom / T + 0.5))
L_actual = l_def × T
```

取整采用正数 half-up，即 `11.5 → 12`、`12.5 → 13`，不使用 Python 的银行家舍入。`same_count` 使用：

```text
N_defects = max(1, floor(defect_density × L_actual + 0.5))
```

`balanced_count` 令 `λ = defect_density × L_actual = q + f`。第 `i` 个 replica（从 1 开始）的缺陷数为：

```text
N_i = q + floor(i × f + 0.5) - floor((i-1) × f + 0.5)
```

因此任意前 `R` 个 replica 的缺陷总数都等于 `floor(R × λ + 0.5)`。增加 `replicas` 只追加新任务，不会改变已有 replica 的缺陷数或 seed。每个 `workflow_config.json` 会记录该 replica 的实际 `N_defects`、实际密度、目标密度和 ensemble 密度；`replica_conductance.csv` 也保留每个 replica 的实际缺陷数。

例如 `(5,5)`、目标长度 `10 Å`、目标密度 `0.051210 Å⁻¹` 会解析为 `l_def=4`、实际长度约 `9.838 Å`。20 个 replica 使用 `balanced_count` 时，10 个分配 0 个缺陷、10 个分配 1 个缺陷，ensemble 实际密度约为 `0.050823 Å⁻¹`；`same_count` 会让全部 replica 各有 1 个缺陷。

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

已没有独立的 `resubmit_negf.py`，重提就是对同一根目录重新运行 `run_multi.py`（或由 `batch_generate.py` 重跑对应批次）：

```bash
python run_multi.py --root ./data/.../replica_XXX --config <对应 config.json> --scheduler slurm
```

`run_multi.py` 会跳过带 `dpnegf_done.flag` 且 provenance 一致的已完成 leaf，只处理未完成的目录。重提前应先查看对应目录的 `workflow.log`、`slurm-*.err` 和 `dpnegf_failed.flag` 确认失败原因，然后：

- DPNEGF 真正运行失败（存在 `dpnegf_failed.flag`）：先删除该 flag 再重跑，否则 `sub_dpnegf.py` 会拒绝重复提交；
- 上次 sbatch 提交阶段失败（存在 `dpnegf_submit_failed.flag`，无 `dpnegf_failed.flag`）：直接重跑即可重新提交；
- 修改了输入配置或模型：provenance hash 校验（P0-4）会阻止复用与新配置不一致的旧结果，需删除对应 `dpnegf_done.flag` 与 `output/` 后显式重算。

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

# POSCAR 模块拼接（搭积木）

独立批配置入口是 [`batch_assemble.py`](batch_assemble.py)。先把其中的完美模块和各缺陷模块路径改为自己准备的 POSCAR，然后在集群环境运行 `python batch_assemble.py --scheduler slurm`；本地预检可运行 `python -m unittest tests.test_assemble`。它调用现有 `batch_generate.py` / `run_multi.py` 调度和下游流程；原 `batch.py`、ASE 缺陷生成和外部单 UC 模式保持原样。配置中的 `md_steps = 0` 表示跳过 LAMMPS，直接从拼接 POSCAR 经 `sgeom` 进入 DPNEGF；设为正数则运行原有 LAMMPS MD 流程。

每个 `configs` 条目填写 `perfect_module`（严格一个 UC）、`defect_modules`（名称到 `(POSCAR 路径, 占用 UC 数)`，例如 `{"MVH": ("./modules/MVH/POSCAR", 2)}`）、`defect_counts`（同名键到非负整数）、`scatter_uc_count`（散射区 UC 总数）、`buffer_pl`、`rotation_angles_deg`、`replicas`，可选 `module_sequence`。每出现一次缺陷模块计为一个缺陷，因此请准备每模块一个缺陷的 POSCAR。手动序列中用 `"perfect"` 表示完美模块，其余名称对应 `defect_modules`；不填时按 `base_structure_seed + replica` 随机排列缺陷模块，再以完美模块填满。缺陷模块的轴向晶胞长度必须等于声明的 UC 数乘完美 UC 的晶胞长度（容差 1e-5 Å）；长度取 POSCAR 晶胞向量，不取原子坐标跨度。缺陷模块占用的 UC 总数超过散射区时直接报错。

完美 POSCAR 必须恰好包含一个对应手性的 UC。沿用原工作流的 `l_PL = floor(r_max / T) + 1`：每侧先取 2 PL 电极，再取 `buffer_pl` 个 PL 缓冲；省略 `buffer_pl` 时金属管为 2、半导体管为 4。LAMMPS 固定端沿用最内侧放开 1 UC 的规则。电极和缓冲均仅由完美 UC 重复构造。`assembly.json` 同时记录散射区 UC 数和由 POSCAR 晶胞长度得到的实际 Å 长度；线密度只以散射区实际 Å 长度计算。

所有模块需为轴沿正 z、横向晶胞相同、接口原子已配准的 C 或 C/H POSCAR；完美电极模块仅含 C。旋转只从 `rotation_angles_deg` 的离散列表抽取；列表必须包含 0，程序先检验完美模块在每个角度的原子配准，再检验缺陷模块在两端的接口原子与完美模块一致。不做优化、缺陷构造、裁剪或自动修复。实际序列、逐模块旋转角、长度、缺陷数和只按散射区计算的线密度写到每个 replica 的 `assembly.json`。已存在内容的 replica 目录会报错，避免覆盖计算结果。配置和 seed 不变时，序列及角度可复现。
