# bug.md — CNT 缺陷输运流水线 Bug / 缺陷审查报告

> 审查范围：`run.py` / `run_multi.py` / `workflow_status.py` 及 `stage/` 下全部模块
> 审查方法：逐文件通读 + 对关键疑点做最小复现验证
> 约定：`文件:行号` 可点击跳转；严重程度分 Critical / High / Medium / Low

---

## Critical（会产生"看似成功、实则错误"的结果）

### C1. `copy_input_dpnegf.py` 不从 config 读取 `r_max`，导致电极原子索引区间算错
- **位置**：`stage/copy_input_dpnegf.py:400-416`（`main()`）、`:139,153`（`update_input_json_for_leaf` / `geo_info`）
- **现象**：`main()` 只从 config 读了 `l_def`，`r_max` 仍取 argparse 默认值 `6.5`（`--r-max` 默认 6.5）。而 `run.py` / `run_multi.py` 调用 `copy_input` 时**并未传 `--r-max`**（`grep -n r-max run.py` 无结果）。
- **后果**：一旦 `config.json` 的 `r_max ≠ 6.5`，`cnt_geometry.geo_info` 会算出**错误的 `l_PL`** → 错误的 `n_elec = n_lead_pl * N_uc * l_PL` → 写进 `input.json` 的 `lead_L / device / lead_R` 的 `id` 区间全部错位。DPNEGF 会照常跑完并输出 `negf.out.pth`，但左右电极/散射区划分是错的——**静默产出错误物理结果**。
- **与 C3 同源**：`fdf2xyz.py:469-472` 是从 config 读 `r_max` 的（正确），`copy_input` 却用 6.5，二者对同一体系用**不同的 `r_max`** → PL 划分不一致（见 C3）。
- **修复建议**：`copy_input` 的 `main()` 中改为 `r_max = float(config.get("r_max", args.r_max))`（或在 `run.py` 里显式把 `--r-max` 透传下去，与 `fdf2xyz` 保持一致）。

### C2. `copy_input_dpnegf.py` 单个 leaf 失败不影响退出码，且会残留"错误但完整"的 `input.json`
- **位置**：`stage/copy_input_dpnegf.py:292-331`（per-leaf `try/except` + 结尾 `print`，**无非零退出**）
- **现象**：每个 leaf 先复制模板文件（含 `input.json` 模板），再 `update_input_json_for_leaf` 改写。若改写抛异常（如 `geo_info`、原子数校验、手性解析失败），异常被吞掉，`n_failed++`，但**模板 `input.json`（硬编码 id `0-144 / 144-394 / 394-538`）已经躺在 leaf 里**。脚本最终仍 `return 0`。
- **后果**：
  1. `run.py` 用 `subprocess.run(..., check=True)` 也发现不了失败（copy_input 返回 0）。
  2. 该 leaf 依旧满足 `is_dpnegf_workdir`（`input.json`+`run.py`+`run.sh`+`*.xyz`+`nnenv*.pth` 齐全）→ 被 `sub_dpnegf` 拉起，用**模板里错误的电极区间**跑 DPNEGF。
- **对比**：`fdf2xyz.py:511-512` 在 `n_failed>0` 时 `raise SystemExit(1)`，行为正确；`copy_input` 缺这一步，**两者不一致**。
- **修复建议**：`copy_inputs_to_leaf_dirs` 结束时若 `n_failed > 0` 则 `sys.exit(1)`；并考虑失败时删除刚复制进去的模板 `input.json`，避免残留完整但错误的 leaf。

### C3. `fdf2xyz` 与 `copy_input` 对同一体系使用不一致的 `r_max` → PL 划分不一致
- **位置**：`stage/fdf2xyz.py:469-472`（用 config `r_max`）vs `stage/copy_input_dpnegf.py:412`（用默认 6.5）
- **后果**：`fdf2xyz` 里 `swap_left_two_pl_order` 用 config 的 `r_max` 算 `n_atoms_per_pl`，而 `copy_input` 用 6.5 算电极 id。两者若不一致，会出现**xyz 里交换的 PL 边界**与 **input.json 里电极 id 边界**对不上——电极原子集合与实际几何错配。根因同 C1，修复 C1 即可消除。

---

## High（破坏某项功能或工具，但通常不至于静默出错）

### H1. `workflow_status.py` 的 LAMMPS dump 检查漏了 `lammps/` 前缀 → 每个跑完的算例都被误判
- **位置**：`stage/../workflow_status.py:25,52`
- **现象**：`LMP_DUMP_PATTERNS = ("*.dump", "dump*", "*.lammpstrj")`，在 `case_dir.glob(pat)` 里直接对**结构目录**匹配。但 dump 实际在 `case_dir/lammps/traj.dump`。已复现：`case_dir.glob('*.dump')` 返回空，`case_dir.glob('lammps/*.dump')` 才命中。
- **后果**：S2 的 dump 检查永远 `MISSING`；而 `done_flag = lammps/lammps_done.flag`（路径正确）存在 → `_evaluate_stage` 判为 **SUSPICIOUS**。即**每一个 LAMMPS 已完成的算例都会被状态工具错误地标成"可疑：done 标志在但输出缺失"**。
- **修复建议**：把三个 pattern 改成 `"lammps/*.dump", "lammps/dump*", "lammps/*.lammpstrj"`（与 S0/S1 的 `lammps/POSCAR` 写法一致）。

### H2. `workflow_status.py` 的 `parse_args` 永远忽略传入参数
- **位置**：`workflow_status.py:433`
- **现象**：`parsed = parser.parse_args(args if args is None else None)`。已复现：无论 `args` 是 `None` 还是 `['--root','x']`，传给 `parse_args` 的永远是 `None`。
- **后果**：`run_status(args=...)` 想以编程方式传参时被完全忽略，只会去读 `sys.argv`；这是明显的逻辑错误（三元表达式恒为 `None`）。CLI 直接跑时"碰巧能用"，但作为函数调用即失效。
- **修复建议**：直接写 `parsed = parser.parse_args(args)`。

### H3. `ele_multi_defects_ele.py` 的 `load_config` 默认写死绝对路径，忽略 `CNT_CONFIG` / `--config`
- **位置**：`stage/ele_multi_defects_ele.py:165`
- **现象**：`def load_config(config_path="/personal/.../config_multi.json")`——默认值是**非 None 的绝对路径**，于是 `if config_path is None:`（读环境变量的分支）永远不执行。调用处 `load_config()` 也不传参。
- **后果**：`run_multi.py --config X` 会 `env["CNT_CONFIG"]=X` 并传下去，但 `ele_multi` **完全无视**，永远读死写的 `config_multi.json`。这违反 CLAUDE.md 明确规定的"`--config` → `CNT_CONFIG` → 默认"优先级。对照 `ele_defects_ele.py:21` 的实现（`config_path=None` 默认、读 env）是正确的。
- **修复建议**：改成 `def load_config(config_path=None):`，与 `ele_defects_ele.py` 保持一致。

### H4. `copy_input_dpnegf.py` 仍在用"路径正则解析手性"，与 CLAUDE.md 显式禁令冲突
- **位置**：`stage/copy_input_dpnegf.py:16-39,151`（`get_chirality_from_fdf` → `update_input_json_for_leaf`）
- **现象**：CLAUDE.md 的 Gotcha 明确要求"手性从 config 读，不要重新引入 `\d+_\d+` 路径正则，否则会把 `5775_5775` 之类结构名误解析"。但 `copy_input` 主流程仍调用 `get_chirality_from_fdf`，它对 FDF 第一行（`from <abs_dump_path> ...`）和路径做 `/(\d+)_(\d+)/` 正则。
- **后果**：当前"能用"仅仅因为路径里手性目录（如 `5_5`）恰好排在结构名（`5775_5775`）之前、且 `data_root` 名不含 `\d+_\d+`。这是**脆弱的巧合**——若 `data_root` 或上层目录出现类似 `123_456` 的片段，就会取错手性，进而 `geo_info` 全错（叠加 C1 的影响）。
- **修复建议**：像 `fdf2xyz` 一样从 config 读手性并传入 `update_input_json_for_leaf`，删除路径正则依赖。

---

## Medium

### M1. 模型文件发现要求 `nnenv*.pth`，但复制环节接受任意 `*.pth` → 非 `nnenv` 命名的模型会被静默跳过
- **位置**：`run.py:162` / `stage/sub_dpnegf.py:39` 要求 `nnenv*.pth`；`copy_input_dpnegf.py:241` 与 `run.py:455` 只按 `*.pth` 选模型
- **后果**：若把模型命名为非 `nnenv...`（例如 `model.pth`），`copy_input` 会正常软链进 leaf，但 `is_dpnegf_workdir` 因 `glob("nnenv*.pth")` 为空而判定该 leaf 不完整 → **DPNEGF 阶段静默不运行该目录**，且不报错。
- **修复建议**：统一发现口径（都用 `*.pth`），或在 CLAUDE.md 里把"模型必须 `nnenv*.pth`"作为硬约束在 `run.py` 选模型时校验并报错。

### M2. `run_multi.py` 在 per-workdir 循环内对 `root` 做全树缓存清理 → O(N²) 且会误删其他 workdir 的缓存
- **位置**：`run_multi.py:514-524`（`for workdir ...: clean_dpnegf_output_cache(root)`）
- **现象**：CLAUDE.md 说明"multi 版按 root 而非 workdir 清理"是有意的，但清理调用被放在**每个 workdir 的循环体内**，每跑完一个就 `rglob("output")` 遍历整棵 `root` 树。
- **后果**：N 个 workdir → N 次全树遍历（O(N²)）；并且会删掉尚未处理/其它 workdir 的 `self_energy`、`HS_*.h5`。当前串行执行只是"浪费"，但若将来改并行就会误删活跃缓存。
- **修复建议**：把 `clean_dpnegf_output_cache(root)` 移到 for 循环**之外**、全部跑完后执行一次。

### M3. `dump2fdf` 的 `--every` 默认 40000 与 LAMMPS `run 40000` 强耦合，易静默产出 0 帧
- **位置**：`stage/dump2fdf_batch.py:215-221`（`timestep % every == 0` 且默认跳过 0）、`run.py:407-418`（不传 `--every`）、`input_files/lammps/in.lammps:57`（`run 40000`）
- **现象**：当前只会抽到 `timestep=40000` 这一帧（0 被跳过）。若有人把 `in.lammps` 的 `run` 改小（< 40000）或改成非 40000 的整数，`every=40000` 会导致**抽 0 帧** → 没有任何 `STRUCT.fdf` → 后续 `fdf2xyz`/`copy_input`/DPNEGF 全部无事可做，而 `run.py` 不会报错（`dump2fdf` 返回 0）。
- **修复建议**：把 `--every` 纳入 config，并在 `total_frames == 0` 时让 `dump2fdf` 以非零码退出或至少醒目告警。

---

## Low / 稳健性

### L1. 原子 `type` 映射 `1→C, 2→H` 依赖"C 原子排在最前"的隐式约定
- **位置**：`stage/dump2fdf_batch.py:8-11`（写死映射）＋ `stage/exporters.py:6-36`（`reposition_hydrogens` 保证 C 在前）
- **说明**：已实测 ASE `lammps-data` 按"首次出现顺序"分配 type（C 先出现 → type 1）。当前 `reposition_hydrogens` 让 C 恒在最前，故成立。但这是**跨模块的隐式契约**，无任何断言保护；若哪天有结构 H 排到最前，`dump2fdf` 会把 C/H 张冠李戴且不报错。建议在 `dump2fdf` 里对 type→元素做一次一致性校验（如按 `data.lmp` 的 Masses 段核对）。

### L2. multi 流水线的目录名内含非 ASCII 字符 `Å`
- **位置**：`stage/ele_multi_defects_ele.py:279`（`folder_name = f"{type_name}_Dens_{Dens:.2f}Å-1"`）
- **说明**：路径含 `Å` 在多数 Linux 上可用，但对某些下游工具/编码环境（尤其跨机器、非 UTF-8 locale）是隐患。建议用 `A-1` 或 `inv_ang` 之类纯 ASCII。

### L3. `dump2fdf` 默认不 `--wrap`，坐标可能未回盒
- **位置**：`stage/dump2fdf_batch.py:256`（`--wrap` 默认关）、`run.py` 不传 `--wrap`
- **说明**：冻结电极 + NVT 下一般无碍，但若中心区原子跨越 z 周期边界，未 wrap 的坐标可能让 DPNEGF 的电极/散射区判定异常。属设计取舍，记录备查。

### L4. `ele_*_defects_ele.py` 在模块导入作用域直接执行几何构建
- **位置**：`stage/ele_defects_ele.py`、`stage/ele_multi_defects_ele.py`（无 `if __name__=="__main__"`）
- **说明**：CLAUDE.md 已注明这是刻意为之（notebook 风格）。作为子进程 `python xxx.py` 运行没问题，但任何 `import` 都会触发副作用；`dump2fdf`/`copy_input` 里 `from fdf2xyz import ...` 时须确保不会误连带执行到这两个脚本（目前没有直接 import 它们，安全）。

### L5. `fdf2xyz.build_output_dir` 为死代码
- **位置**：`stage/fdf2xyz.py:56-115`
- **说明**：`main()` 采用"原地输出"（`output_dir = current_dir`），`build_output_dir` 未被调用。它内部反而实现了正确的"从 config 读手性、只删第一个手性目录"逻辑。留着无害，但易误导后续维护者以为输出走的是重建目录逻辑。建议删除或标注 deprecated。

---

## 附：已确认"没问题/属设计"的点（避免重复排查）
- `run.py` 的 `has_dump_files` 不会把 `dump2fdf_batch.py` 误判为 dump（`.py` 后缀、且不以 `dump.` 开头）——OK。
- `sub_lmps` / `sub_dpnegf` 的 flag 幂等逻辑（done 跳过、failed 不自动重跑、dangling submitted 警告跳过）与 CLAUDE.md 一致——OK。
- `swap_left_two_pl_order` 只动最前 `2*n_atoms_per_pl` 个（纯 C 电极），H 被 `reposition_hydrogens` 放在中间，交换安全——OK（前提是 `r_max` 一致，见 C3）。
- `run.py` vs `run_multi.py` 的差异仅三处（默认 config、ele 脚本、清理目标），与 CLAUDE.md 描述吻合——OK（但清理位置见 M2）。

---

## 建议修复优先级
1. **C1 / C3**（copy_input 读 config 的 r_max）—— 直接影响物理结果正确性，改动最小、收益最大。
2. **C2**（copy_input 失败传播 + 不残留错误 input.json）。
3. **H3**（ele_multi 尊重 CNT_CONFIG）、**H1 / H2**（修 workflow_status，恢复状态可信度）。
4. **H4 / M1 / M2 / M3**。
5. Low 项按需清理。
