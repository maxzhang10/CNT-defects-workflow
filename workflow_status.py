"""
Workflow status checker for CNT defects pipeline.

Only reads the filesystem; does not run any external commands.
Allowed stdlib imports: pathlib, argparse, json, dataclasses, enum, typing, re, sys
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Candidate glob patterns for LAMMPS dump files (any_of)
LMP_DUMP_PATTERNS = ("*.dump", "dump*", "*.lammpstrj")

# Stage definitions
STRUCTURE_STAGE_DEFS = {
    "S0": {
        "name": "结构生成",
        "checks": [
            {"pattern": "lammps/POSCAR", "required": True},
            {"pattern": "lammps/data.lmp", "required": True},
        ],
        "done_flag": None,
        "fail_flag": None,
    },
    "S1": {
        "name": "LAMMPS 输入准备",
        "checks": [
            {"pattern": "lammps/in.lammps", "required": True},
            {"pattern": "lammps/CH.airebo-m", "required": True},
            {"pattern": "lammps/run.sh", "required": True},
        ],
        "done_flag": None,
        "fail_flag": None,
    },
    "S2": {
        "name": "LAMMPS 运行",
        "checks": [
            {"pattern": "lammps/lammps_done.flag", "required": True},
            {"pattern_any_of": LMP_DUMP_PATTERNS, "required": True},
        ],
        "done_flag": "lammps/lammps_done.flag",
        "fail_flag": "lammps/lammps_failed.flag",
    },
}

TIMESTEP_STAGE_DEFS = {
    "T3": {
        "name": "dump→fdf 转换",
        "checks": [
            {"pattern": "dpnegf/{timestep}/STRUCT.fdf", "required": True},
        ],
        "done_flag": None,
        "fail_flag": None,
    },
    "T4": {
        "name": "fdf→xyz 转换",
        "checks": [
            {"pattern": "dpnegf/{timestep}/{m}_{n}.xyz", "required": True},
        ],
        "done_flag": None,
        "fail_flag": None,
    },
    "T5": {
        "name": "DPNEGF 输入复制",
        "checks": [
            {"pattern": "dpnegf/{timestep}/input.json", "required": True},
            {"pattern": "dpnegf/{timestep}/run.py", "required": True},
            {"pattern": "dpnegf/{timestep}/run.sh", "required": True},
            {"pattern": "dpnegf/{timestep}/nnenv*.pth", "required": True},
        ],
        "done_flag": None,
        "fail_flag": None,
    },
    "T6": {
        "name": "DPNEGF 运行",
        "checks": [
            {"pattern": "dpnegf/{timestep}/dpnegf_done.flag", "required": True},
            {"pattern": "dpnegf/{timestep}/output/negf.out.pth", "required": True},
        ],
        "done_flag": "dpnegf/{timestep}/dpnegf_done.flag",
        "fail_flag": "dpnegf/{timestep}/dpnegf_failed.flag",
    },
}


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class StageState(Enum):
    DONE = "DONE"
    FAILED = "FAILED"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class FileCheck:
    pattern: str
    found_paths: List[Path]
    is_required: bool
    is_satisfied: bool
    tried_paths: List[str] = field(default_factory=list)


@dataclass
class StageResult:
    stage_id: str
    name: str
    state: StageState
    checks: List[FileCheck]
    done_flag: Optional[Path] = None
    fail_flag: Optional[Path] = None
    extra_note: str = ""

    @property
    def missing_files(self) -> List[str]:
        return [c.pattern for c in self.checks if c.is_required and not c.is_satisfied]


@dataclass
class TimestepStatus:
    timestep: int
    stages: Dict[str, StageResult]
    overall_state: StageState


@dataclass
class CaseStatus:
    case_path: Path
    temperature: str
    chirality: str
    structure_name: str
    struct_stages: Dict[str, StageResult]
    timesteps: Dict[int, TimestepStatus]
    overall_state: StageState


@dataclass
class WorkflowStatus:
    root: Path
    config: dict
    cases: List[CaseStatus]
    summary: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _check_pattern(case_dir: Path, pattern: str) -> Tuple[List[Path], List[str]]:
    """Return (found_paths, tried_paths) for a single glob pattern."""
    found = list(case_dir.glob(pattern))
    tried = [str(case_dir / pattern)]
    return found, tried


def _check_any_of(case_dir: Path, patterns: Tuple[str, ...]) -> Tuple[List[Path], List[str]]:
    """Return (found_paths, tried_paths) for any_of patterns."""
    found: List[Path] = []
    tried: List[str] = []
    for pat in patterns:
        f, t = _check_pattern(case_dir, pat)
        found.extend(f)
        tried.extend(t)
    return found, tried


def _evaluate_stage(
    checks: List[FileCheck],
    done_flag: Optional[Path] = None,
    fail_flag: Optional[Path] = None,
) -> StageState:
    if fail_flag and fail_flag.exists():
        return StageState.FAILED

    required_satisfied = all(c.is_satisfied for c in checks if c.is_required)
    done_flag_exists = done_flag is not None and done_flag.exists()

    if done_flag_exists and not required_satisfied:
        return StageState.SUSPICIOUS

    if required_satisfied:
        if done_flag_exists or done_flag is None:
            return StageState.DONE
        return StageState.PARTIAL

    if done_flag_exists:
        return StageState.SUSPICIOUS

    return StageState.MISSING


def _check_structure_stages(case_dir: Path) -> Dict[str, StageResult]:
    results: Dict[str, StageResult] = {}
    for stage_id, sdef in STRUCTURE_STAGE_DEFS.items():
        checks: List[FileCheck] = []
        for cdef in sdef["checks"]:
            if "pattern_any_of" in cdef:
                found, tried = _check_any_of(case_dir, cdef["pattern_any_of"])
                is_satisfied = len(found) > 0
                # Represent the any_of as a combined description
                pattern_desc = f"any_of({', '.join(cdef['pattern_any_of'])})"
                checks.append(FileCheck(
                    pattern=pattern_desc,
                    found_paths=found,
                    is_required=cdef["required"],
                    is_satisfied=is_satisfied,
                    tried_paths=tried,
                ))
            else:
                found, tried = _check_pattern(case_dir, cdef["pattern"])
                checks.append(FileCheck(
                    pattern=cdef["pattern"],
                    found_paths=found,
                    is_required=cdef["required"],
                    is_satisfied=len(found) > 0,
                    tried_paths=tried,
                ))

        done_flag = None
        if sdef.get("done_flag"):
            done_flag = case_dir / sdef["done_flag"]
        fail_flag = None
        if sdef.get("fail_flag"):
            fail_flag = case_dir / sdef["fail_flag"]

        state = _evaluate_stage(checks, done_flag, fail_flag)
        results[stage_id] = StageResult(
            stage_id=stage_id,
            name=sdef["name"],
            state=state,
            checks=checks,
            done_flag=done_flag,
            fail_flag=fail_flag,
        )
    return results


def _check_timestep_stages(case_dir: Path, timestep: int, m: int, n: int) -> Dict[str, StageResult]:
    results: Dict[str, StageResult] = {}
    for stage_id, sdef in TIMESTEP_STAGE_DEFS.items():
        checks: List[FileCheck] = []
        for cdef in sdef["checks"]:
            pattern = cdef["pattern"].format(timestep=timestep, m=m, n=n)
            found, tried = _check_pattern(case_dir, pattern)
            checks.append(FileCheck(
                pattern=pattern,
                found_paths=found,
                is_required=cdef["required"],
                is_satisfied=len(found) > 0,
                tried_paths=tried,
            ))

        done_flag = None
        if sdef.get("done_flag"):
            done_flag = case_dir / sdef["done_flag"].format(timestep=timestep, m=m, n=n)
        fail_flag = None
        if sdef.get("fail_flag"):
            fail_flag = case_dir / sdef["fail_flag"].format(timestep=timestep, m=m, n=n)

        state = _evaluate_stage(checks, done_flag, fail_flag)
        results[stage_id] = StageResult(
            stage_id=stage_id,
            name=sdef["name"],
            state=state,
            checks=checks,
            done_flag=done_flag,
            fail_flag=fail_flag,
        )
    return results


def _discover_timesteps(case_dir: Path) -> List[int]:
    timesteps: List[int] = []
    dpnegf_dir = case_dir / "dpnegf"
    if dpnegf_dir.exists():
        for subdir in dpnegf_dir.iterdir():
            if subdir.is_dir():
                try:
                    timesteps.append(int(subdir.name))
                except ValueError:
                    pass
    return sorted(timesteps)


def _scan_case(case_dir: Path, m: int, n: int, temperature: str) -> CaseStatus:
    struct_stages = _check_structure_stages(case_dir)
    timesteps = _discover_timesteps(case_dir)

    timestep_status: Dict[int, TimestepStatus] = {}
    for ts in timesteps:
        ts_stages = _check_timestep_stages(case_dir, ts, m, n)
        if any(s.state == StageState.FAILED for s in ts_stages.values()):
            overall = StageState.FAILED
        elif all(s.state == StageState.DONE for s in ts_stages.values()):
            overall = StageState.DONE
        elif any(s.state == StageState.SUSPICIOUS for s in ts_stages.values()):
            overall = StageState.SUSPICIOUS
        else:
            overall = StageState.PARTIAL

        timestep_status[ts] = TimestepStatus(
            timestep=ts,
            stages=ts_stages,
            overall_state=overall,
        )

    all_states = [s.state for s in struct_stages.values()]
    if timestep_status:
        all_states.extend([ts.overall_state for ts in timestep_status.values()])

    if StageState.FAILED in all_states:
        case_state = StageState.FAILED
    elif StageState.SUSPICIOUS in all_states:
        case_state = StageState.SUSPICIOUS
    elif all(s == StageState.DONE for s in all_states):
        case_state = StageState.DONE
    else:
        case_state = StageState.PARTIAL

    return CaseStatus(
        case_path=case_dir,
        temperature=temperature,
        chirality=f"{m}_{n}",
        structure_name=case_dir.name,
        struct_stages=struct_stages,
        timesteps=timestep_status,
        overall_state=case_state,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _state_icon(state: StageState) -> str:
    icons = {
        StageState.DONE: "✓",
        StageState.FAILED: "✗",
        StageState.MISSING: "○",
        StageState.PARTIAL: "◐",
        StageState.SKIPPED: "⊘",
        StageState.SUSPICIOUS: "⚠",
    }
    return icons.get(state, "?")


def _print_status_report(status: WorkflowStatus) -> None:
    print("=" * 80)
    print("Workflow Status Report")
    print(f"Root: {status.root}")
    print(f"Total Cases: {len(status.cases)}")
    print("=" * 80)

    for case in status.cases:
        print(f"\n{'─' * 80}")
        print(f"Case: {case.structure_name}")
        print(f"Path: {case.case_path}")
        print(f"Overall: {case.overall_state.value}")

        # Structure stages
        print("\n  [Structure Stages]")
        for stage_id in ("S0", "S1", "S2"):
            if stage_id in case.struct_stages:
                stage = case.struct_stages[stage_id]
                icon = _state_icon(stage.state)
                print(f"    {icon} {stage_id}: {stage.name} [{stage.state.value}]")
                if stage.missing_files:
                    for mf in stage.missing_files:
                        print(f"       Missing: {mf}")
                # Print tried paths for missing files
                for check in stage.checks:
                    if check.is_required and not check.is_satisfied:
                        for tp in check.tried_paths:
                            print(f"       Tried: {tp}")
                if stage.state == StageState.SUSPICIOUS:
                    print("       Note: done flag exists but critical output missing")

        # Timestep stages
        if case.timesteps:
            print("\n  [Timestep Stages]")
            for ts_num, ts in sorted(case.timesteps.items()):
                print(f"\n    Timestep: {ts_num}")
                for stage_id in ("T3", "T4", "T5", "T6"):
                    if stage_id in ts.stages:
                        stage = ts.stages[stage_id]
                        icon = _state_icon(stage.state)
                        print(f"      {icon} {stage_id}: {stage.name} [{stage.state.value}]")
                        if stage.missing_files:
                            for mf in stage.missing_files:
                                print(f"         Missing: {mf}")
                        for check in stage.checks:
                            if check.is_required and not check.is_satisfied:
                                for tp in check.tried_paths:
                                    print(f"         Tried: {tp}")
                        if stage.state == StageState.SUSPICIOUS:
                            print("         Note: done flag exists but critical output missing")
        else:
            # Check if LAMMPS is done but no timesteps found
            s2 = case.struct_stages.get("S2")
            if s2 and s2.state == StageState.DONE:
                print("\n  [Timestep Stages]")
                print("    READY_FOR_DUMP2FDF (LAMMPS done, no timesteps found)")
            else:
                print("\n  [No timesteps found]")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_status(args: Optional[argparse.Namespace] = None) -> None:
    """Entry point for the status subcommand."""
    parser = argparse.ArgumentParser(description="Check workflow status")
    parser.add_argument("--root", default=None, help="Data root directory")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parsed = parser.parse_args(args if args is None else None)

    config = _load_config(parsed.config)
    root = Path(parsed.root or config.get("data_root", "../data")).resolve()
    temperature = config.get("temperature", "")
    chirality = config.get("chirality", [0, 0])
    m, n = int(chirality[0]), int(chirality[1])

    cases: List[CaseStatus] = []
    base_dir = root / f"{temperature}K" / f"{m}_{n}"
    if base_dir.exists():
        for case_dir in base_dir.iterdir():
            if case_dir.is_dir():
                cases.append(_scan_case(case_dir, m, n, str(temperature)))

    status = WorkflowStatus(
        root=root,
        config=config,
        cases=cases,
    )

    _print_status_report(status)


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load config.json with fallback logic."""
    if config_path is None:
        # Try environment variable
        env_config = __import__("os").environ.get("CNT_CONFIG")
        if env_config:
            config_path = env_config
        else:
            # Default: look for config.json in parent of this file
            config_path = Path(__file__).resolve().parent / "config.json"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    run_status()
