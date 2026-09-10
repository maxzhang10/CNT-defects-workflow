#!/usr/bin/env python3
import os
import re
import json
import math
import copy
import shutil
import argparse
from pathlib import Path
from fdf2xyz import load_config, get_chirality_from_config
import cnt_geometry

import logkit as L
from negf_provenance import (
    compute_negf_config_hash,
    expected_hash_path,
    write_hash_file,
)


def clean_line(line):
    return line.split("#", 1)[0].strip()


def read_fdf_block(lines, block_name):
    block_name = block_name.lower()
    inside = False
    data = []

    for line in lines:
        line = clean_line(line)

        if not line:
            continue

        parts = line.split()
        lower = [x.lower() for x in parts]

        if len(lower) >= 2 and lower[0] == "%block" and lower[1] == block_name:
            inside = True
            continue

        if inside and lower[0] == "%endblock":
            return data

        if inside:
            data.append(line)

    raise ValueError(f"未找到 block：{block_name}")


def count_atoms_from_struct_fdf(fdf_path):
    with open(fdf_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    coord_block = read_fdf_block(lines, "AtomicCoordinatesAndAtomicSpecies")

    n_atoms = 0
    for row in coord_block:
        parts = row.split()
        if len(parts) >= 4:
            n_atoms += 1

    if n_atoms <= 0:
        raise ValueError(f"未能从坐标 block 中读取原子数：{fdf_path}")

    return n_atoms




def update_run_py_for_leaf(leaf_dir, model_filename, m, n):
    """
    修改 leaf 目录下 dpnegf run.py 里的：
        model_path = "xxx.pth"
        structure  = "xxx.xyz"
    """
    leaf_dir = Path(leaf_dir).resolve()
    run_py_path = leaf_dir / "run.py"

    if not run_py_path.is_file():
        raise FileNotFoundError(f"找不到 run.py：{run_py_path}")

    model_path = f"./{model_filename}"
    structure_path = f"./{m}_{n}.xyz"

    with open(run_py_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text, n_model = re.subn(
        r'(^\s*model_path\s*=\s*)(["\'])(.*?)\2',
        lambda x: f'{x.group(1)}"{model_path}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    text, n_structure = re.subn(
        r'(^\s*structure\s*=\s*)(["\'])(.*?)\2',
        lambda x: f'{x.group(1)}"{structure_path}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if n_model == 0:
        raise KeyError(f"run.py 中未找到 model_path = ... ：{run_py_path}")

    if n_structure == 0:
        raise KeyError(f"run.py 中未找到 structure = ... ：{run_py_path}")

    with open(run_py_path, "w", encoding="utf-8") as f:
        f.write(text)

    L.debug(
        f"Updated run.py -> {run_py_path}\n"
        f"  model_path = {model_path}\n"
        f"  structure  = {structure_path}"
    )


def update_input_json_for_leaf(
    leaf_dir,
    model_filename,
    chirality,
    espacing,
    negf_energy_window,
    self_energy_cache,
    r_max=6.5,
    n_lead_pl=2,
    l_def=5,
):
    leaf_dir = Path(leaf_dir).resolve()

    fdf_path = leaf_dir / "STRUCT.fdf"
    input_json_path = leaf_dir / "input.json"

    if not fdf_path.is_file():
        raise FileNotFoundError(f"找不到 STRUCT.fdf：{fdf_path}")

    if not input_json_path.is_file():
        raise FileNotFoundError(f"找不到 input.json：{input_json_path}")

    m, n = chirality

    T, N_uc, l_PL, length = cnt_geometry.geo_info(m, n, r_max, l_def)

    n_atoms_per_pl = N_uc * l_PL
    n_elec = n_lead_pl * n_atoms_per_pl
    n_total = count_atoms_from_struct_fdf(fdf_path)

    if n_total <= 2 * n_elec:
        raise ValueError(
            f"体系原子数不足，无法定义左右电极：\n"
            f"  fdf_path        = {fdf_path}\n"
            f"  n_total         = {n_total}\n"
            f"  n_atoms_per_pl  = {n_atoms_per_pl}\n"
            f"  n_elec          = {n_elec}\n"
            f"  need >          = {2 * n_elec}"
        )

    lead_L_id = f"0-{n_elec}"
    device_id = f"{n_elec}-{n_total - n_elec}"
    lead_R_id = f"{n_total - n_elec}-{n_total}"

    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    task_options = input_data["task_options"]
    stru_options = task_options["stru_options"]

    # 保留模板 input.json 中的 ele_T。工作流的 temperature 用于结构/MD，
    # 不应再隐式覆盖 DPNEGF 电极温度。
    task_options["espacing"] = espacing
    task_options["emin"] = negf_energy_window[0]
    task_options["emax"] = negf_energy_window[1]
    task_options["self_energy_options"]["cache"] = copy.deepcopy(self_energy_cache)

    stru_options["lead_L"]["id"] = lead_L_id
    stru_options["device"]["id"] = device_id
    stru_options["lead_R"]["id"] = lead_R_id

    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=4)

    update_run_py_for_leaf(
        leaf_dir=leaf_dir,
        model_filename=model_filename,
        m=m,
        n=n,
    )

    L.debug(
        f"Updated input.json -> {leaf_dir}\n"
        f"  m={m}, n={n}\n"
        f"  N_uc={N_uc}, l_PL={l_PL}, atoms_per_PL={n_atoms_per_pl}\n"
        f"  n_total={n_total}, n_elec={n_elec}\n"
        f"  lead_L.id = {lead_L_id}\n"
        f"  device.id = {device_id}\n"
        f"  lead_R.id = {lead_R_id}"
        f"\n  espacing = {espacing} eV"
        f"\n  energy range = [{negf_energy_window[0]}, {negf_energy_window[1]}] eV"
        f"\n  self-energy cache = {self_energy_cache}"
    )

def copy_or_link_file(src: Path, dst: Path, overwrite: bool = True):
    """
    普通文件复制。
    .pth 文件软链接。

    overwrite=True 时：
        如果目标已存在或是坏软链，先删除再重新创建。
    """
    src = src.resolve()

    if overwrite and (dst.exists() or dst.is_symlink()):
        dst.unlink()

    if src.suffix == ".pth":
        dst.symlink_to(src)
        L.debug(f"软链接模型: {dst} -> {src}")
    else:
        shutil.copy2(src, dst)
        L.debug(f"复制: {src} -> {dst}")


def collect_input_files(input_dir: Path, files_to_copy, model_file=None):
    """
    收集需要放入 leaf 的文件：
      1. files_to_copy 里的普通模板文件
      2. input_dir 下所有 .pth 模型文件

    同时返回用于写入 input.json 的模型文件名。
    """
    input_dir = Path(input_dir).resolve()

    input_files = []

    for filename in files_to_copy:
        src = input_dir / filename
        if not src.is_file():
            raise FileNotFoundError(f"源文件不存在：{src}")
        input_files.append(src)

    model_files = sorted(input_dir.glob("*.pth"))

    if len(model_files) == 0:
        raise FileNotFoundError(f"未在 input_dir 中找到 .pth 模型文件：{input_dir}")

    if model_file is not None:
        selected_model = input_dir / model_file
        if not selected_model.is_file():
            raise FileNotFoundError(f"指定的模型文件不存在：{selected_model}")
        selected_model = selected_model.resolve()
    else:
        if len(model_files) > 1:
            names = "\n".join(f"  {p.name}" for p in model_files)
            raise ValueError(
                f"input_dir 中找到多个 .pth 模型文件，请用 --model-file 指定一个：\n{names}"
            )
        selected_model = model_files[0].resolve()

    input_files.extend(model_files)

    return input_files, selected_model.name

def copy_inputs_to_leaf_dirs(
    input_dir,
    root_dir,
    files_to_copy,
    chirality,
    model_file=None,
    r_max=6.5,
    n_lead_pl=2,
    l_def=5,
    espacing=0.1,
    negf_energy_window=(-0.5, 0.5),
    self_energy_cache=None,
    overwrite=True,
    negf_config_hash=None,
):
    input_dir = Path(input_dir).resolve()
    root_dir = Path(root_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在：{input_dir}")

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir 不存在：{root_dir}")

    input_files, model_filename = collect_input_files(
        input_dir=input_dir,
        files_to_copy=files_to_copy,
        model_file=model_file,
    )

    n_leaf = 0
    n_success = 0
    n_failed = 0
    n_skipped = 0

    for current_dir, subdirs, files in os.walk(root_dir):
        if "STRUCT.fdf" not in files:
            continue

        current_dir = Path(current_dir).resolve()
        n_leaf += 1

        # 无论该 leaf 是否已完成，都把本次运行期望的 config hash 写到 leaf 下，
        # 作为 provenance 校验的期望值（run.sh 成功时复制为 provenance；
        # sub_dpnegf.py 据此校验 done 结果是否与当前配置一致）。
        if negf_config_hash is not None:
            try:
                write_hash_file(expected_hash_path(current_dir), negf_config_hash)
            except OSError as e:
                L.warn(f"写入 expected hash 失败: {current_dir}: {e}")

        # 已完成的 DPNEGF leaf：绝不覆盖其输入文件。
        # 原因：input.json / run.py 的电极 id、model_path 是按 config+几何重算的，
        # 用不同 config 重跑会把输入改写成与已算好的 output/negf.out.pth 不一致；
        # 且下方失败分支会删除 input.json，一旦重处理已完成 leaf 时抛异常，
        # 会把跑完的 leaf 破坏成 discovery 都识别不到的残缺状态。
        # done flag 由 sub_dpnegf.py 写在同一层 leaf 目录下。
        if (current_dir / "dpnegf_done.flag").exists():
            L.skip(f"DPNEGF 已完成，跳过覆盖输入: {current_dir}")
            n_skipped += 1
            continue

        try:
            L.debug("=" * 80)
            L.debug(f"Leaf: {current_dir}")

            for src in input_files:
                dst = current_dir / src.name
                copy_or_link_file(src, dst, overwrite=overwrite)

            update_input_json_for_leaf(
                leaf_dir=current_dir,
                model_filename=model_filename,
                chirality=chirality,
                espacing=espacing,
                negf_energy_window=negf_energy_window,
                self_energy_cache=self_energy_cache,
                r_max=r_max,
                n_lead_pl=n_lead_pl,
                l_def=l_def,
            )

            n_success += 1

        except Exception as e:
            L.error(f"leaf 处理失败: {current_dir}")
            L.error(f"原因: {e}")
            n_failed += 1

            # 删除刚复制进去的模板 input.json，避免残留"完整但错误"的 leaf
            # （硬编码电极 id 的模板会被 is_dpnegf_workdir 误判为可运行）
            stale_input_json = current_dir / "input.json"
            if stale_input_json.is_file():
                try:
                    stale_input_json.unlink()
                    L.warn(f"已删除残留模板 input.json: {stale_input_json}")
                except OSError as unlink_err:
                    L.warn(f"无法删除残留 input.json：{stale_input_json}：{unlink_err}")

    L.ok(f"copy_input 完成  成功={n_success}  失败={n_failed}  跳过={n_skipped}  (共 {n_leaf} 个 leaf)")
    L.info(f"输入目录 = {input_dir}")
    L.info(f"根目录   = {root_dir}")
    L.info(f"模型文件 = {model_filename}")

    return n_failed

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy DPNEGF input files to all STRUCT.fdf leaf directories, "
            "symlink .pth model files, and update input.json lead/device atom ids."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.json path. If not given, use CNT_CONFIG or work_flow_test/config.json."
    )
    parser.add_argument(
        "--input-dir",
        default="../input_files/dpnegf",
        help="Directory containing input.json, run.py, run.sh, and *.pth model files."
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to search for STRUCT.fdf leaf directories."
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "input.json",
            "run.py",
            "run.sh",
        ],
        help=(
            "Non-model files copied from input-dir to each leaf directory. "
            "All *.pth files in input-dir are automatically symlinked."
        )
    )

    parser.add_argument(
    "--model-file",
    default=None,
    help=(
        "Model .pth filename in input-dir used to write input.json model_path. "
        "If omitted, input-dir must contain exactly one .pth file."
    )
    )

    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="CNT r_max parameter used by cnt_geometry.geo_info. If not given, read from config.json."
    )

    parser.add_argument(
        "--n-lead-pl",
        type=int,
        default=2,
        help="Number of PLs used for each lead."
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files or links in leaf directories."
    )

    args = parser.parse_args()
    config = load_config(args.config)
    l_def = int(config.get("l_def", 5))
    chirality = get_chirality_from_config(config)

    # r_max 从 config 读取，保证与 fdf2xyz 对同一体系用相同的 PL 划分。
    # 仅当显式传入 --r-max 时才覆盖（默认 None 表示不覆盖）。
    if args.r_max is None:
        r_max = float(config.get("r_max", 6.5))
    else:
        r_max = float(args.r_max)

    L.info("[CONFIG]")
    L.info(f"l_def     = {l_def}")
    L.info(f"chirality = {chirality}")
    L.info(f"r_max     = {r_max}")
    espacing = float(config.get("espacing", 0.1))
    if not math.isfinite(espacing) or espacing <= 0.0:
        raise ValueError(
            f"espacing must be a finite positive value in eV, got {espacing}"
        )
    L.info(f"espacing = {espacing} eV")
    raw_negf_window = config.get("negf_energy_window", [-0.5, 0.5])
    if not isinstance(raw_negf_window, (list, tuple)) or len(raw_negf_window) != 2:
        raise ValueError("negf_energy_window must be [emin, emax] in eV")
    try:
        negf_energy_window = (float(raw_negf_window[0]), float(raw_negf_window[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("negf_energy_window must contain two numeric values") from exc
    if (
        not all(math.isfinite(value) for value in negf_energy_window)
        or negf_energy_window[0] >= negf_energy_window[1]
    ):
        raise ValueError("negf_energy_window requires finite emin < emax")
    L.info(
        "negf_energy_window = "
        f"[{negf_energy_window[0]}, {negf_energy_window[1]}] eV"
    )
    raw_cache = config.get("self_energy_cache")
    if raw_cache is None:
        raise ValueError("self_energy_cache must be configured")
    if not isinstance(raw_cache, dict):
        raise ValueError("self_energy_cache must be an object with use_saved and save_path")
    use_saved = raw_cache.get("use_saved")
    save_path = raw_cache.get("save_path")
    if not isinstance(use_saved, bool) or not isinstance(save_path, str) or not save_path:
        raise ValueError("self_energy_cache requires boolean use_saved and non-empty string save_path")
    self_energy_cache = {"use_saved": use_saved, "save_path": save_path}
    L.info(f"self_energy_cache = {self_energy_cache}")

    n_failed = copy_inputs_to_leaf_dirs(
        input_dir=args.input_dir,
        root_dir=args.root,
        files_to_copy=args.files,
        chirality=chirality,
        model_file=args.model_file,
        r_max=r_max,
        n_lead_pl=args.n_lead_pl,
        overwrite=(not args.no_overwrite),
        l_def=l_def,
        espacing=espacing,
        negf_energy_window=negf_energy_window,
        self_energy_cache=self_energy_cache,
        negf_config_hash=compute_negf_config_hash(config),
    )

    if n_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
