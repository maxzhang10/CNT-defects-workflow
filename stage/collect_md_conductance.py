#!/usr/bin/env python3
"""
Collect zero-bias conductance across independent LAMMPS replicas.

Expected layout
---------------
<root>/
└── <temperature>K/
    └── <m>_<n>/
        └── <configuration>/
            ├── replica_001/
            │   ├── workflow_config.json
            │   └── dpnegf/<md_steps>/output/negf.out.pth
            ├── replica_002/
            └── ...

Outputs
-------
Only inside each configuration directory:
    replica_conductance.csv
    replica_conductance.log

No cross-configuration CSV files are written at root.
Legacy conductance_all.csv / conductance_summary.csv files are removed.

The dimensionless linear-response conductance is evaluated from the
Landauer formula:
    G/G0 = integral T(E) [-df(E; E_F, T)/dE] dE.
At zero transport temperature this reduces to G/G0 = T(E_F).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "ERROR: collect_replica_conductance.py requires PyTorch. "
        "Run it with the DeePTB/DPNEGF Python environment."
    ) from exc


ENERGY_KEYS = (
    "uni_grid",
    "energy_grid",
    "energies",
    "energy",
    "E",
)

TRANSMISSION_KEYS = (
    "T_avg",
    "transmission_avg",
    "transmission",
    "T",
)

REPLICA_PATTERN = re.compile(r"^replica_(\d+)$")
DENSITY_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?)A-1$")
TEMPERATURE_PATTERN = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)K$")
CHIRALITY_PATTERN = re.compile(r"^(-?\d+)_(-?\d+)$")

KB_EV_PER_K = 8.617333262145e-5
COVERAGE_HARD_MIN = 0.95


@dataclass(frozen=True)
class ReplicaResult:
    configuration_dir: str
    configuration_name: str
    temperature_K: float | None
    chirality_m: int | None
    chirality_n: int | None
    structures: str
    l_def: int | None
    n_defects: int | None
    density_A_inv: float | None
    replica: int
    expected_replicas: int | None
    structure_seed: int | None
    lammps_seed: int | None
    md_steps: int | None
    dpnegf_sample: str
    conductance_G0: float
    ln_conductance_G0: float | None
    e_fermi_eV: float
    transport_temperature_K: float
    conductance_method: str
    grid_min_eV: float
    grid_max_eV: float
    n_energy_points: int
    replica_dir: str
    result_file: str


@dataclass(frozen=True)
class ConfigurationSummary:
    configuration_dir: str
    configuration_name: str
    temperature_K: float | None
    chirality_m: int | None
    chirality_n: int | None
    structures: str
    l_def: int | None
    n_defects: int | None
    density_A_inv: float | None
    expected_replicas: int | None
    valid_replicas: int
    missing_or_failed_replicas: int
    mean_G0: float
    std_G0: float
    median_G0: float
    min_G0: float
    max_G0: float
    mean_ln_G0: float | None
    std_ln_G0: float | None
    geometric_mean_G0: float | None


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def ok(message: str) -> None:
    print(f"[OK]   {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_named_value(
    obj: Any,
    candidate_keys: Sequence[str],
) -> tuple[Any, str]:
    if isinstance(obj, Mapping):
        for key in candidate_keys:
            if key in obj:
                return obj[key], str(key)

        lower_map = {str(key).lower(): key for key in obj.keys()}
        for key in candidate_keys:
            matched = lower_map.get(key.lower())
            if matched is not None:
                return obj[matched], str(matched)

        for value in obj.values():
            if isinstance(value, (Mapping, list, tuple)):
                try:
                    return find_named_value(value, candidate_keys)
                except KeyError:
                    pass

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, (Mapping, list, tuple)):
                try:
                    return find_named_value(value, candidate_keys)
                except KeyError:
                    pass

    raise KeyError(
        "none of the expected keys were found: "
        + ", ".join(candidate_keys)
    )


def to_numpy(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{label} cannot be converted to an array") from exc

    if array.size == 0:
        raise ValueError(f"{label} is empty")

    if np.iscomplexobj(array):
        array = np.real_if_close(array)
        if np.iscomplexobj(array):
            max_imag = float(np.max(np.abs(np.imag(array))))
            raise ValueError(
                f"{label} contains non-negligible complex values "
                f"(max |Im|={max_imag:.3e})"
            )

    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")

    return array


def normalize_energy_grid(value: Any) -> np.ndarray:
    energy = np.squeeze(to_numpy(value, "energy grid"))
    if energy.ndim != 1:
        raise ValueError(
            "energy grid must be one-dimensional after squeeze; "
            f"got shape {energy.shape}"
        )
    if energy.size < 2:
        raise ValueError("energy grid must contain at least two points")
    return energy


def normalize_transmission(
    value: Any,
    n_energy: int,
    reduction: str,
) -> np.ndarray:
    transmission = np.squeeze(to_numpy(value, "transmission"))

    if transmission.ndim == 0:
        raise ValueError(
            "transmission is scalar; expected an energy-dependent array"
        )

    if transmission.ndim == 1:
        if transmission.size != n_energy:
            raise ValueError(
                f"transmission length {transmission.size} does not match "
                f"energy-grid length {n_energy}"
            )
        return transmission

    matching_axes = [
        axis
        for axis, size in enumerate(transmission.shape)
        if size == n_energy
    ]

    if not matching_axes:
        raise ValueError(
            f"no transmission axis matches energy-grid length {n_energy}; "
            f"transmission shape is {transmission.shape}"
        )
    if len(matching_axes) > 1:
        raise ValueError(
            f"ambiguous transmission shape {transmission.shape}: "
            f"multiple axes match energy-grid length {n_energy}"
        )

    energy_axis = matching_axes[0]
    transmission = np.moveaxis(transmission, energy_axis, -1)
    component_shape = transmission.shape[:-1]
    n_components = int(np.prod(component_shape))
    transmission = transmission.reshape(n_components, n_energy)

    if n_components == 1:
        return transmission[0]
    if reduction == "error":
        raise ValueError(
            f"transmission contains {n_components} components with shape "
            f"{component_shape}. Specify --component-reduction sum, mean, "
            "or first explicitly."
        )
    if reduction == "sum":
        return np.sum(transmission, axis=0)
    if reduction == "mean":
        return np.mean(transmission, axis=0)
    if reduction == "first":
        return transmission[0]

    raise ValueError(f"unsupported component reduction: {reduction}")


def prepare_curve(
    energy: np.ndarray,
    transmission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(energy)
    energy = energy[order]
    transmission = transmission[order]

    unique_energy, inverse, counts = np.unique(
        energy,
        return_inverse=True,
        return_counts=True,
    )

    if unique_energy.size != energy.size:
        accumulated = np.zeros(unique_energy.size, dtype=float)
        np.add.at(accumulated, inverse, transmission)
        transmission = accumulated / counts
        energy = unique_energy

    if energy.size < 2:
        raise ValueError("fewer than two unique energy points remain")

    return energy, transmission


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """
    NumPy compatibility helper: np.trapezoid is unavailable in older versions.
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def interpolate_conductance(
    energy: np.ndarray,
    transmission: np.ndarray,
    e_fermi: float,
    temperature_k: float = 0.0,
) -> float:
    grid_min = float(energy[0])
    grid_max = float(energy[-1])

    tolerance = max(1.0, abs(grid_min), abs(grid_max)) * 1.0e-12
    if e_fermi < grid_min - tolerance or e_fermi > grid_max + tolerance:
        raise ValueError(
            f"E_F={e_fermi:g} eV lies outside the energy grid "
            f"[{grid_min:g}, {grid_max:g}] eV"
        )

    e_eval = min(max(e_fermi, grid_min), grid_max)

    if temperature_k <= 0.0:
        conductance = float(np.interp(e_eval, energy, transmission))
    else:
        kbt = KB_EV_PER_K * temperature_k
        if not math.isfinite(kbt) or kbt <= 0.0:
            raise ValueError(f"invalid transport temperature: {temperature_k}")

        # Resample to resolve the narrow Fermi window at low temperatures.
        delta_e = np.diff(energy)
        min_step = float(np.min(delta_e))
        target_step = min(min_step, kbt / 20.0)
        if not math.isfinite(target_step) or target_step <= 0.0:
            target_step = min_step

        n_dense = int(math.ceil((grid_max - grid_min) / target_step)) + 1
        n_dense = min(max(n_dense, energy.size), 200001)

        if n_dense > energy.size:
            energy_eval = np.linspace(grid_min, grid_max, n_dense)
            transmission_eval = np.interp(energy_eval, energy, transmission)
        else:
            energy_eval = energy
            transmission_eval = transmission
        #energy_eval为变量E   e_eval为常数E_F
        u = (energy_eval - e_eval) / (2.0 * kbt)
        #u_clipped为变量u的截断值，限制在[-50, 50]范围内，避免数值溢出
        u_clipped = np.clip(u, -50.0, 50.0)
        kernel = 0.25 / kbt / np.cosh(u_clipped) ** 2
        kernel[np.abs(u) > 50.0] = 0.0

        coverage = trapezoid_integral(kernel, energy_eval)
        if not math.isfinite(coverage) or coverage <= 0.0:
            raise ValueError(
                "invalid finite-temperature kernel coverage; check energy grid"
            )
        if coverage < COVERAGE_HARD_MIN:   #COVERAGE_HARD_MIN = 0.95，用于判断透射谱能量范围是否足够宽，覆盖费米窗口
            raise ValueError(
                    "energy range is too narrow for finite-temperature conductance: "
                    f"kernel coverage={coverage:.4f}. Increase the energy range "
                    "around E_F, typically to at least +/-10 k_B T."
            )
        if coverage < 0.995:
            warn(
                "energy range is not wide enough for finite-temperature conductance: "
                f"(coverage={coverage:.4f}). Increase the energy range around E_F, typically to at least +/-10 k_B T."
            )
            kernel = kernel / coverage

        conductance = trapezoid_integral(
            transmission_eval * kernel,
            energy_eval,
        )

    if not math.isfinite(conductance):
        raise ValueError("interpolated conductance is not finite")

    return conductance


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return value


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_replica_number(replica_dir: Path) -> int:
    match = REPLICA_PATTERN.fullmatch(replica_dir.name)
    if match is None:
        raise ValueError(f"invalid replica directory name: {replica_dir.name}")
    return int(match.group(1))


def parse_path_metadata(configuration_dir: Path) -> dict[str, Any]:
    temperature: float | None = None
    chirality_m: int | None = None
    chirality_n: int | None = None
    density: float | None = None

    if len(configuration_dir.parents) >= 2:
        chirality_match = CHIRALITY_PATTERN.fullmatch(
            configuration_dir.parent.name
        )
        if chirality_match:
            chirality_m = int(chirality_match.group(1))
            chirality_n = int(chirality_match.group(2))

        temperature_match = TEMPERATURE_PATTERN.fullmatch(
            configuration_dir.parent.parent.name
        )
        if temperature_match:
            temperature = float(temperature_match.group(1))

    density_match = DENSITY_PATTERN.search(configuration_dir.name)
    if density_match:
        density = float(density_match.group(1))

    return {
        "temperature_K": temperature,
        "chirality_m": chirality_m,
        "chirality_n": chirality_n,
        "density_A_inv": density,
    }


def find_configuration_dirs(root: Path) -> list[Path]:
    configuration_dirs: set[Path] = set()

    for path in root.rglob("replica_*"):
        if not path.is_dir():
            continue
        if REPLICA_PATTERN.fullmatch(path.name) is None:
            continue
        configuration_dirs.add(path.parent.resolve())

    return sorted(configuration_dirs)


def find_replica_dirs(configuration_dir: Path) -> list[Path]:
    replica_dirs = [
        path
        for path in configuration_dir.iterdir()
        if path.is_dir() and REPLICA_PATTERN.fullmatch(path.name)
    ]
    return sorted(replica_dirs, key=parse_replica_number)


def find_result_files(replica_dir: Path) -> list[Path]:
    dpnegf_root = replica_dir / "dpnegf"
    if not dpnegf_root.is_dir():
        return []

    return sorted(
        {
            path.resolve()
            for path in dpnegf_root.rglob("negf.out.pth")
            if path.is_file()
        }
    )


def choose_result_file(
    replica_dir: Path,
    config: Mapping[str, Any],
) -> Path:
    candidates = find_result_files(replica_dir)
    if not candidates:
        raise FileNotFoundError(
            f"negf.out.pth not found beneath {replica_dir / 'dpnegf'}"
        )

    md_steps = optional_int(config.get("md_steps"))
    if md_steps is not None:
        preferred_workdir = replica_dir / "dpnegf" / str(md_steps)
        preferred = [
            preferred_workdir / "output" / "negf.out.pth",
            preferred_workdir / "negf.out.pth",
        ]
        for path in preferred:
            if path.is_file():
                if len(candidates) > 1:
                    warn(
                        f"{replica_dir}: found {len(candidates)} result files; "
                        f"using last-frame result {path}"
                    )
                return path.resolve()

    if len(candidates) == 1:
        return candidates[0]

    listing = "\n".join(f"  {path}" for path in candidates)
    raise RuntimeError(
        f"multiple negf.out.pth files found in {replica_dir}, but no unique "
        f"md_steps result could be selected:\n{listing}"
    )


def process_result_file(
    result_file: Path,
    e_fermi: float,
    transport_temperature_k: float,
    reduction: str,
) -> tuple[float, float, float, int, str, str]:
    payload = load_torch_file(result_file)

    energy_value, energy_key = find_named_value(payload, ENERGY_KEYS)
    transmission_value, transmission_key = find_named_value(
        payload,
        TRANSMISSION_KEYS,
    )

    energy = normalize_energy_grid(energy_value)
    transmission = normalize_transmission(
        transmission_value,
        n_energy=energy.size,
        reduction=reduction,
    )
    energy, transmission = prepare_curve(energy, transmission)
    conductance = interpolate_conductance(
        energy,
        transmission,
        e_fermi=e_fermi,
        temperature_k=transport_temperature_k,
    )

    return (
        conductance,
        float(energy[0]),
        float(energy[-1]),
        int(energy.size),
        energy_key,
        transmission_key,
    )


def get_structures_text(config: Mapping[str, Any]) -> str:
    structures = config.get("structures")
    if isinstance(structures, (list, tuple)):
        return "_".join(str(item) for item in structures)
    if structures is None:
        return ""
    return str(structures)


def build_replica_result(
    configuration_dir: Path,
    replica_dir: Path,
    e_fermi: float,
    transport_temperature_k: float | None,
    reduction: str,
    expected_replicas_override: int | None,
) -> ReplicaResult:
    config = read_json(replica_dir / "workflow_config.json")
    path_meta = parse_path_metadata(configuration_dir)

    effective_transport_temperature_k = transport_temperature_k
    if effective_transport_temperature_k is None:
        effective_transport_temperature_k = optional_float(
            config.get("temperature")
        )
        if effective_transport_temperature_k is None:
            effective_transport_temperature_k = path_meta["temperature_K"]
        if effective_transport_temperature_k is None:
            effective_transport_temperature_k = 300.0

    replica_number = parse_replica_number(replica_dir)
    result_file = choose_result_file(replica_dir, config)

    (
        conductance,
        grid_min,
        grid_max,
        n_energy,
        energy_key,
        transmission_key,
    ) = process_result_file(
        result_file=result_file,
        e_fermi=e_fermi,
        transport_temperature_k=effective_transport_temperature_k,
        reduction=reduction,
    )

    conductance_method = (
        "zero-temperature interpolation"
        if effective_transport_temperature_k <= 0.0
        else "finite-temperature Fermi-window integral"
    )

    chirality = config.get("chirality")
    chirality_m = path_meta["chirality_m"]
    chirality_n = path_meta["chirality_n"]
    if isinstance(chirality, (list, tuple)) and len(chirality) >= 2:
        chirality_m = optional_int(chirality[0])
        chirality_n = optional_int(chirality[1])

    expected_replicas = expected_replicas_override
    if expected_replicas is None:
        expected_replicas = optional_int(config.get("lammps_repeats"))

    density = optional_float(config.get("density_A_inv"))
    if density is None:
        density = path_meta["density_A_inv"]

    dpnegf_sample = result_file.parent.name
    if result_file.parent.name == "output":
        dpnegf_sample = result_file.parent.parent.name

    ln_conductance = (
        math.log(conductance)
        if conductance > 0.0
        else None
    )

    info(
        f"{replica_dir}: {energy_key}/{transmission_key}, "
        f"G/G0({e_fermi:g} eV, {effective_transport_temperature_k:g} K)={conductance:.10g}"
    )

    return ReplicaResult(
        configuration_dir=str(configuration_dir),
        configuration_name=configuration_dir.name,
        temperature_K=optional_float(config.get("temperature"))
        if config.get("temperature") is not None
        else path_meta["temperature_K"],
        chirality_m=chirality_m,
        chirality_n=chirality_n,
        structures=get_structures_text(config),
        l_def=optional_int(config.get("l_def")),
        n_defects=optional_int(config.get("N_defects")),
        density_A_inv=density,
        replica=replica_number,
        expected_replicas=expected_replicas,
        structure_seed=optional_int(config.get("seed")),
        lammps_seed=optional_int(config.get("lammps_seed")),
        md_steps=optional_int(config.get("md_steps")),
        dpnegf_sample=dpnegf_sample,
        conductance_G0=conductance,
        ln_conductance_G0=ln_conductance,
        e_fermi_eV=e_fermi,
        transport_temperature_K=effective_transport_temperature_k,
        conductance_method=conductance_method,
        grid_min_eV=grid_min,
        grid_max_eV=grid_max,
        n_energy_points=n_energy,
        replica_dir=str(replica_dir),
        result_file=str(result_file),
    )


def format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_csv_value(row.get(key))
                    for key in fieldnames
                }
            )

    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def summarize_configuration(
    configuration_dir: Path,
    results: Sequence[ReplicaResult],
    n_problems: int,
    expected_replicas_override: int | None,
) -> ConfigurationSummary:
    if not results:
        raise ValueError(f"cannot summarize empty result set: {configuration_dir}")

    values = np.asarray(
        [result.conductance_G0 for result in results],
        dtype=float,
    )
    positive = values[values > 0.0]

    expected = expected_replicas_override
    if expected is None:
        expected_values = {
            result.expected_replicas
            for result in results
            if result.expected_replicas is not None
        }
        if len(expected_values) == 1:
            expected = expected_values.pop()

    first = results[0]
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    if positive.size:
        log_values = np.log(positive)
        mean_log = float(np.mean(log_values))
        std_log = (
            float(np.std(log_values, ddof=1))
            if log_values.size > 1
            else 0.0
        )
        geometric_mean = float(np.exp(mean_log))
    else:
        mean_log = None
        std_log = None
        geometric_mean = None

    inferred_missing = max(0, expected - len(results)) if expected is not None else 0
    total_problems = max(n_problems, inferred_missing)

    return ConfigurationSummary(
        configuration_dir=str(configuration_dir),
        configuration_name=configuration_dir.name,
        temperature_K=first.temperature_K,
        chirality_m=first.chirality_m,
        chirality_n=first.chirality_n,
        structures=first.structures,
        l_def=first.l_def,
        n_defects=first.n_defects,
        density_A_inv=first.density_A_inv,
        expected_replicas=expected,
        valid_replicas=len(results),
        missing_or_failed_replicas=total_problems,
        mean_G0=float(np.mean(values)),
        std_G0=std,
        median_G0=float(np.median(values)),
        min_G0=float(np.min(values)),
        max_G0=float(np.max(values)),
        mean_ln_G0=mean_log,
        std_ln_G0=std_log,
        geometric_mean_G0=geometric_mean,
    )


def build_configuration_log(
    summary: ConfigurationSummary,
    results: Sequence[ReplicaResult],
    problems: Sequence[str],
) -> str:
    first = results[0]

    lines = [
        "Replica conductance collection",
        "=" * 80,
        f"configuration_dir: {summary.configuration_dir}",
        f"temperature_K: {summary.temperature_K}",
        f"transport_temperature_K: {first.transport_temperature_K}",
        f"conductance_method: {first.conductance_method}",
        f"chirality: ({summary.chirality_m}, {summary.chirality_n})",
        f"structures: {summary.structures}",
        f"l_def: {summary.l_def}",
        f"N_defects: {summary.n_defects}",
        f"density_A^-1: {summary.density_A_inv}",
        f"expected replicas: {summary.expected_replicas}",
        f"valid replicas: {summary.valid_replicas}",
        f"missing/failed replicas: {summary.missing_or_failed_replicas}",
        "",
        "Summary (G/G0 = integral T(E) [-df/dE] dE; at T=0, G/G0 = T(E_F))",
        f"mean:             {summary.mean_G0:.16g}",
        f"std:              {summary.std_G0:.16g}",
        f"median:           {summary.median_G0:.16g}",
        f"min:              {summary.min_G0:.16g}",
        f"max:              {summary.max_G0:.16g}",
        f"mean ln(G/G0):    {summary.mean_ln_G0}",
        f"std ln(G/G0):     {summary.std_ln_G0}",
        f"geometric mean:   {summary.geometric_mean_G0}",
        "",
        "Replicas",
    ]

    for result in sorted(results, key=lambda item: item.replica):
        lines.append(
            f"replica_{result.replica:03d}  "
            f"seed={result.lammps_seed}  "
            f"G/G0={result.conductance_G0:.16g}  "
            f"ln={result.ln_conductance_G0}  "
            f"file={result.result_file}"
        )

    if problems:
        lines.extend(["", "Problems"])
        lines.extend(f"- {message}" for message in problems)

    return "\n".join(lines) + "\n"


def replica_fieldnames() -> list[str]:
    return list(ReplicaResult.__dataclass_fields__.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect linear-response conductance from replica_XXX directories "
            "using the Landauer Fermi-window integral (or T(E_F) at T=0), "
            "then write only per-configuration CSV and log files."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Batch data root containing temperature/chirality/configuration trees.",
    )
    parser.add_argument(
        "--configuration-dir",
        action="append",
        default=[],
        help=(
            "Collect only this configuration directory. May be supplied "
            "multiple times. If omitted, scan every replica_XXX beneath root."
        ),
    )
    parser.add_argument(
        "--e-fermi",
        type=float,
        default=0.0,
        help="Fermi energy in eV. Default: 0.0.",
    )
    parser.add_argument(
        "--transport-temperature-k",
        type=float,
        default=None,
        help=(
            "Transport temperature in K. If omitted, use temperature from "
            "workflow_config.json, then path metadata, then 300 K fallback. "
            "Set 0 to force zero-temperature interpolation G/G0=T(E_F); "
            "values >0 use finite-temperature Fermi-window integration."
        ),
    )
    parser.add_argument(
        "--expected-replicas",
        type=int,
        default=None,
        help=(
            "Expected replica count per configuration. If omitted, read "
            "lammps_repeats from workflow_config.json."
        ),
    )
    parser.add_argument(
        "--component-reduction",
        choices=("error", "sum", "mean", "first"),
        default="error",
        help=(
            "How to combine extra transmission components if T_avg is not "
            "one-dimensional. Default: error."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write available results and exit successfully when replicas are "
            "missing or invalid."
        ),
    )
    parser.add_argument(
        "--per-config-csv",
        default="replica_conductance.csv",
    )
    parser.add_argument(
        "--per-config-log",
        default="replica_conductance.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        error(f"root directory does not exist: {root}")
        return 1
    if (
        args.transport_temperature_k is not None
        and args.transport_temperature_k < 0.0
    ):
        error("--transport-temperature-k must be non-negative")
        return 1
    if args.expected_replicas is not None and args.expected_replicas < 1:
        error("--expected-replicas must be greater than zero")
        return 1

    if args.configuration_dir:
        configuration_dirs = sorted(
            {
                Path(raw).expanduser().resolve()
                for raw in args.configuration_dir
            }
        )
        invalid = [
            path for path in configuration_dirs
            if not path.is_dir()
        ]
        if invalid:
            for path in invalid:
                error(f"configuration directory does not exist: {path}")
            return 1
    else:
        configuration_dirs = find_configuration_dirs(root)

    if not configuration_dirs:
        error(
            f"no configuration directories containing replica_XXX were found: {root}"
        )
        return 1

    info(f"found {len(configuration_dirs)} configuration directories")

    total_valid_results = 0
    completed_configurations = 0
    all_problems: list[str] = []

    # Remove legacy cross-configuration outputs left by older versions.
    for legacy_name in ("conductance_all.csv", "conductance_summary.csv"):
        legacy_path = root / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            info(f"removed legacy global output: {legacy_path}")

    for configuration_dir in configuration_dirs:
        replica_dirs = find_replica_dirs(configuration_dir)
        info(
            f"{configuration_dir}: found {len(replica_dirs)} replica directories"
        )

        results: list[ReplicaResult] = []
        problems: list[str] = []

        for replica_dir in replica_dirs:
            try:
                result = build_replica_result(
                    configuration_dir=configuration_dir,
                    replica_dir=replica_dir,
                    e_fermi=args.e_fermi,
                    transport_temperature_k=args.transport_temperature_k,
                    reduction=args.component_reduction,
                    expected_replicas_override=args.expected_replicas,
                )
            except Exception as exc:
                message = (
                    f"{replica_dir}: {type(exc).__name__}: {exc}"
                )
                error(message)
                problems.append(message)
                continue

            results.append(result)
            ok(
                f"{configuration_dir.name}/replica_{result.replica:03d}: "
                f"G/G0={result.conductance_G0:.10g}"
            )

        if not results:
            message = f"{configuration_dir}: no valid replica results"
            error(message)
            all_problems.append(message)
            all_problems.extend(problems)
            continue

        results.sort(key=lambda item: item.replica)
        summary = summarize_configuration(
            configuration_dir=configuration_dir,
            results=results,
            n_problems=len(problems),
            expected_replicas_override=args.expected_replicas,
        )

        atomic_write_csv(
            configuration_dir / args.per_config_csv,
            [asdict(result) for result in results],
            replica_fieldnames(),
        )
        atomic_write_text(
            configuration_dir / args.per_config_log,
            build_configuration_log(summary, results, problems),
        )

        info(
            f"{configuration_dir.name}: n={summary.valid_replicas}, "
            f"mean={summary.mean_G0:.10g}, std={summary.std_G0:.10g} G0"
        )

        total_valid_results += len(results)
        completed_configurations += 1
        all_problems.extend(problems)

        if (
            summary.expected_replicas is not None
            and summary.valid_replicas != summary.expected_replicas
        ):
            message = (
                f"{configuration_dir}: expected {summary.expected_replicas} "
                f"replicas, collected {summary.valid_replicas}"
            )
            warn(message)
            all_problems.append(message)

    if total_valid_results == 0:
        error("no valid conductance results were collected")
        return 1

    print("=" * 80, flush=True)
    info(f"valid replica results: {total_valid_results}")
    info(f"configuration outputs: {completed_configurations}")
    info(f"problems: {len(all_problems)}")
    info("only per-configuration replica_conductance.csv/.log were written")

    if all_problems and not args.allow_partial:
        error(
            "collection completed with missing/invalid replicas; "
            "use --allow-partial to accept partial results"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
