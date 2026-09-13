#!/usr/bin/env python3
"""
NEGF 结果 provenance：配置哈希与一致性校验。

用于修复"新配置元数据 + 旧计算结果"的静默错配（P0-4）。

config hash 由所有会影响结构 / Hamiltonian / NEGF 结果（即最终
negf.out.pth）的 workflow_config 参数稳定地计算得到。只有当前 hash
与旧结果保存的 provenance hash 完全一致时，才允许复用 done 结果。

文件约定（每个 DPNEGF leaf 工作目录内）
    expected_negf_config_hash.txt   —— 本次运行期望的 hash，由
                                      copy_input_dpnegf.py 写入。
    output/negf_config_hash.txt    —— 实际运行 provenance hash，由
                                      run.sh / sub_dpnegf.py 在 DPNEGF
                                      成功后写入，与 negf.out.pth 同目录层。

默认行为 fail-safe：无法确认旧结果与当前配置一致时，不得复用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


# 影响 negf.out.pth 的 workflow_config 参数：
#   结构：chirality / l_def / N_defects / structures / seed / r_max /
#         md_steps / md_sampling / temperature
#   MD 轨迹：lammps_seed
#   NEGF / Hamiltonian：espacing / negf_energy_window / self_energy_cache
NEGF_CONFIG_KEYS = (
    "chirality",
    "l_def",
    "N_defects",
    "structures",
    "seed",
    "lammps_seed",
    "r_max",
    "md_steps",
    "md_sampling",
    "temperature",
    "espacing",
    "negf_energy_window",
    "self_energy_cache",
)

# leaf 工作目录内期望 hash 文件名（copy_input 写入，作为本次运行的期望值）。
EXPECTED_HASH_FILE = "expected_negf_config_hash.txt"

# output/ 下与 negf.out.pth 同层的 provenance hash 文件名。
PROVENANCE_HASH_FILE = "negf_config_hash.txt"

# DPNEGF 结果文件名。
RESULT_FILE = "negf.out.pth"


def _canonicalize(value: Any) -> str:
    """返回带排序键的 JSON 字符串，保证哈希跨运行稳定。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def compute_negf_config_hash(config: Mapping[str, Any]) -> str:
    """对影响 negf.out.pth 的 workflow_config 参数计算稳定 SHA-256。"""
    subset: dict[str, Any] = {}
    for key in NEGF_CONFIG_KEYS:
        if key in config:
            subset[key] = config[key]
    payload = _canonicalize(subset)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_hash_file(path: Path) -> str | None:
    """读取单行 hash 文件；不存在或为空则返回 None。"""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def write_hash_file(path: Path, hash_value: str) -> None:
    """原子写入单行 hash 文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(hash_value + "\n", encoding="utf-8")
    tmp.replace(path)


def expected_hash_path(leaf_dir: Path) -> Path:
    return Path(leaf_dir) / EXPECTED_HASH_FILE


def provenance_hash_path(leaf_dir: Path) -> Path:
    return Path(leaf_dir) / "output" / PROVENANCE_HASH_FILE


def result_file_path(leaf_dir: Path) -> Path:
    return Path(leaf_dir) / "output" / RESULT_FILE


def result_is_nonempty(leaf_dir: Path) -> bool:
    """结果文件存在且非空。仅做存在性/大小检查，不读取内容。"""
    result = result_file_path(Path(leaf_dir))
    return result.is_file() and result.stat().st_size > 0


def result_is_readable(leaf_dir: Path) -> bool:
    """结果文件存在、非空，且能用 torch.load 真正读出（可读取）。

    用于把"done flag + 损坏/缺失结果"识别为失败，使其进入重试，
    而不是等到批次末尾电导汇总才暴露。

    torch 不在 orchestrator 环境时退化为非空检查 —— 此时由
    collect 阶段（在带 torch 的环境里运行）做最终的读取校验。
    """
    leaf_dir = Path(leaf_dir)
    if not result_is_nonempty(leaf_dir):
        return False
    try:
        import torch  # type: ignore
    except ImportError:
        # orchestrator 环境无 torch：只能确认到非空，读取校验交给下游。
        return True
    try:
        try:
            torch.load(
                result_file_path(leaf_dir),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            torch.load(result_file_path(leaf_dir), map_location="cpu")
    except Exception:
        return False
    return True


def provenance_is_valid(leaf_dir: Path, expected_hash: str) -> bool:
    """校验 leaf 的 provenance：结果文件存在、非空，且 provenance hash 与
    期望 hash 完全一致。任一条件不满足返回 False（fail-safe，不得复用）。
    """
    leaf_dir = Path(leaf_dir)
    result = result_file_path(leaf_dir)
    if not result.is_file() or result.stat().st_size == 0:
        return False
    saved = read_hash_file(provenance_hash_path(leaf_dir))
    if saved is None:
        return False
    return saved == expected_hash
