"""POSCAR module assembly. No geometry optimization or defect creation is done here."""

import hashlib
import math
import random
from pathlib import Path


def module_metadata(path):
    """Read the axial *cell* length, not the extent of atomic coordinates."""
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    lines = data.decode("utf-8-sig").splitlines()
    try:
        scale = float(lines[1].split()[0])
        if len(lines[1].split()) != 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("POSCAR requires one positive scale")
        cell = [[float(x) * scale for x in lines[i].split()] for i in (2, 3, 4)]
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in cell):
            raise ValueError("invalid cell vectors")
        if (abs(cell[0][2]) > 1e-6 or abs(cell[1][2]) > 1e-6
                or abs(cell[2][0]) > 1e-6 or abs(cell[2][1]) > 1e-6
                or cell[2][2] <= 0):
            raise ValueError("transport axis must be positive, untilted z")
        if abs(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0]) < 1e-8:
            raise ValueError("transverse cell vectors must not be collinear")
        symbols = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        if not symbols or len(symbols) != len(counts) or any(n <= 0 for n in counts):
            raise ValueError("invalid VASP5 element/count lines")
        if not set(symbols) <= {"C", "H"} or "C" not in symbols:
            raise ValueError("only C and H modules are supported")
        mode_index = 8 if lines[7].strip().lower().startswith("s") else 7
        mode = lines[mode_index].strip().lower()
        if not mode.startswith(("d", "c", "k")):
            raise ValueError("coordinates must be Direct or Cartesian")
        if len(lines) < mode_index + 1 + sum(counts):
            raise ValueError("missing atomic coordinates")
        for line in lines[mode_index + 1:mode_index + 1 + sum(counts)]:
            xyz = [float(x) for x in line.split()[:3]]
            if len(xyz) != 3 or not all(math.isfinite(x) for x in xyz):
                raise ValueError("invalid atomic coordinate")
            z = xyz[2] * (cell[2][2] if mode.startswith("d") else scale)
            # Optimized POSCARs may retain tiny signed boundary roundoff.
            # Accept it without moving or wrapping atoms.
            if not -1e-3 <= z < cell[2][2] + 1e-3:
                raise ValueError("atomic z coordinate lies outside its module cell")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid module POSCAR {path}: {exc}") from exc
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "length_angstrom": cell[2][2], "cell": cell,
            "atom_count": sum(counts), "symbols": symbols, "counts": counts}


def lead_layout(perfect, m, n, r_max, buffer_pl, release_uc=1):
    """Apply the existing floor(r_max/T)+1 UC/PL and 2PL lead rule."""
    if type(buffer_pl) is not int or buffer_pl < 0:
        raise ValueError("buffer_pl must be a nonnegative integer")
    if type(release_uc) is not int or release_uc < 0:
        raise ValueError("release_uc must be a nonnegative integer")
    if isinstance(r_max, bool) or not isinstance(r_max, (int, float)) or not math.isfinite(r_max) or r_max <= 0:
        raise ValueError("r_max must be positive and finite")
    expected = 4 * (m*m + m*n + n*n) // math.gcd(2*m+n, 2*n+m)
    if perfect["atom_count"] != expected:
        raise ValueError(
            f"perfect module has {perfect['atom_count']} C atoms; ({m},{n}) requires "
            f"exactly {expected} for one UC"
        )
    uc_length = perfect["length_angstrom"]
    l_pl = math.floor(r_max / uc_length) + 1
    electrode_uc = 2 * l_pl
    buffer_uc = buffer_pl * l_pl
    side_uc = electrode_uc + buffer_uc
    if side_uc - release_uc <= 0:
        raise ValueError("release_uc leaves no fixed atoms on either side")
    return {
        "uc_length_angstrom": uc_length, "atoms_per_uc": expected,
        "l_PL_uc": l_pl, "buffer_pl": buffer_pl,
        "electrode_uc_per_side": electrode_uc,
        "buffer_uc_per_side": buffer_uc,
        "side_uc": side_uc, "side_repeats": side_uc,
        "atoms_per_pl": l_pl * expected,
        "electrode_atoms_per_side": electrode_uc * expected,
        "side_atoms_per_side": side_uc * expected,
        "fixed_atoms_per_side": (side_uc - release_uc) * expected,
        "electrode_length_angstrom": electrode_uc * uc_length,
        "buffer_length_angstrom": buffer_uc * uc_length,
        "release_uc": release_uc,
    }


def plan(config, seed):
    """Return a deterministic sequence; reject impossible lengths/counts."""
    modules = config["modules"]
    perfect = modules["perfect"]
    defects = modules["defects"]
    target_uc = config["scatter_uc_count"]
    counts = config["defect_counts"]
    manual = config.get("module_sequence")
    if manual is not None:
        sequence = list(manual)
        if {key: sequence.count(key) for key in defects} != counts:
            raise ValueError("module_sequence defect counts differ from defect_counts")
        if any(key != "perfect" and key not in defects for key in sequence):
            raise ValueError("module_sequence contains an unknown module")
    else:
        sequence = [key for key, count in counts.items() for _ in range(count)]
        occupied_uc = sum(defects[key]["uc_count"] for key in sequence)
        remaining_uc = target_uc - occupied_uc
        if remaining_uc < 0:
            raise ValueError(f"defect modules occupy {occupied_uc} UC, exceeding scatter_uc_count={target_uc}")
        sequence += ["perfect"] * remaining_uc
        random.Random(seed).shuffle(sequence)
    actual_uc = sum(1 if key == "perfect" else defects[key]["uc_count"] for key in sequence)
    if actual_uc != target_uc:
        raise ValueError(f"module sequence occupies {actual_uc} UC, expected {target_uc} UC")
    actual = sum((perfect if key == "perfect" else defects[key])["length_angstrom"] for key in sequence)
    if not sequence:
        raise ValueError("scatter region cannot be empty")
    return sequence, actual


def _same_positions(reference, rotated, tolerance=0.08):
    """One-to-one species/position matching for an allowed discrete rotation."""
    import numpy as np
    if len(reference) != len(rotated):
        return False
    used = set()
    for symbol, pos in zip(rotated.get_chemical_symbols(), rotated.positions):
        options = [j for j, (other, ref) in enumerate(zip(reference.get_chemical_symbols(), reference.positions))
                   if j not in used and other == symbol and np.linalg.norm(pos - ref) <= tolerance]
        if not options:
            return False
        used.add(options[0])
    return True


def build(config, seed):
    """Build electrode | scatter | electrode, returning ASE Atoms and metadata."""
    import numpy as np
    from ase.io import read
    from ase import Atoms

    sequence, scatter_length = plan(config, seed)
    modules = config["modules"]
    source = {"perfect": modules["perfect"], **modules["defects"]}
    atoms_by_name = {}
    perfect_meta = source["perfect"]
    for name, meta in source.items():
        if module_metadata(meta["path"])["sha256"] != meta["sha256"]:
            raise ValueError(f"module changed since configuration was loaded: {name}")
        atoms = read(meta["path"], format="vasp")
        if len(atoms) != meta["atom_count"]:
            raise ValueError(f"module atom count changed: {name}")
        atoms_by_name[name] = atoms
    perfect = atoms_by_name["perfect"]
    if set(perfect.get_chemical_symbols()) != {"C"}:
        raise ValueError("perfect electrode module must contain only C")
    axis = perfect.positions[:, :2].mean(axis=0)
    angles = config["rotation_angles_deg"]
    for angle in angles:
        rotated = perfect.copy()
        rotated.rotate(angle, "z", center=(axis[0], axis[1], 0))
        if not _same_positions(perfect, rotated):
            raise ValueError(f"rotation {angle}° does not preserve perfect-module atomic registry")
    # Input modules are user-prepared. Require a common transverse cell and
    # matching pristine seam atoms at both ends; never shift or repair them.
    for name, atoms in atoms_by_name.items():
        meta = source[name]
        if not np.allclose(np.asarray(meta["cell"])[:2], np.asarray(perfect_meta["cell"])[:2], atol=1e-5):
            raise ValueError(f"module {name} has a different transverse cell")
        if name != "perfect":
            period = perfect_meta["length_angstrom"]
            seam_width = min(
                period / 2, meta["length_angstrom"] / 2,
                max(1.5, float(perfect.positions[:, 2].min()) + 0.2,
                    period - float(perfect.positions[:, 2].max()) + 0.2),
            )
            seam_tolerance = 1e-3
            for angle in angles:
                rotated = atoms.copy()
                rotated.rotate(angle, "z", center=(axis[0], axis[1], 0))
                for side in ("left", "right"):
                    if side == "left":
                        actual = rotated[rotated.positions[:, 2] <= seam_width + seam_tolerance]
                        expected = perfect[perfect.positions[:, 2] <= seam_width + seam_tolerance]
                    else:
                        actual = rotated[rotated.positions[:, 2] >= meta["length_angstrom"] - seam_width - seam_tolerance]
                        expected = perfect[perfect.positions[:, 2] >= perfect_meta["length_angstrom"] - seam_width - seam_tolerance].copy()
                        expected.positions[:, 2] += meta["length_angstrom"] - perfect_meta["length_angstrom"]
                    if not len(expected):
                        raise ValueError(f"perfect module has no {side} interface atoms to validate")
                    if not _same_positions(expected, actual, tolerance=0.2):
                        raise ValueError(f"module {name} at {angle}° has an unregistered {side} interface")
    layout = config["layout"]
    repeats = layout["side_repeats"]
    pieces = []
    placement = []
    z = 0.0
    rng = random.Random(seed)
    for name in (["perfect"] * repeats + sequence + ["perfect"] * repeats):
        meta = source[name]
        block = atoms_by_name[name].copy()
        angle = rng.choice(angles) if name != "perfect" else 0.0
        if angle:
            block.rotate(angle, "z", center=(axis[0], axis[1], 0))
        block.positions[:, 2] += z
        # Preserve axial ordering across module boundaries for lead/device ids.
        block = block[np.lexsort((block.positions[:, 0], block.positions[:, 1], block.positions[:, 2]))]
        pieces.append(block)
        placement.append({"module": name, "rotation_deg": angle,
                          "uc_count": 1 if name == "perfect" else meta["uc_count"],
                          "start_angstrom": z, "length_angstrom": meta["length_angstrom"]})
        z += meta["length_angstrom"]
    result = Atoms(cell=perfect.cell, pbc=(False, False, False))
    for block in pieces:
        result += block
    result.cell[2, 2] = z
    n_electrode = layout["electrode_atoms_per_side"]
    defect_total = sum(config["defect_counts"].values())
    return result, {"seed": seed, "module_sequence": sequence,
                    "placements": placement, "scatter_uc_count": config["scatter_uc_count"],
                    "scatter_length_angstrom": scatter_length,
                    "electrode_length_angstrom": layout["electrode_length_angstrom"],
                    "buffer_length_angstrom": layout["buffer_length_angstrom"],
                    "side_length_angstrom": repeats * perfect_meta["length_angstrom"],
                    "total_length_angstrom": z, "defect_counts": config["defect_counts"],
                    "N_defects": defect_total,
                    "actual_density_A_inv": defect_total / scatter_length,
                    "electrode_atoms_per_side": n_electrode,
                    "side_atoms_per_side": layout["side_atoms_per_side"],
                    "fixed_atoms_per_side": layout["fixed_atoms_per_side"],
                    "atoms_per_pl": layout["atoms_per_pl"],
                    "l_PL_uc": layout["l_PL_uc"],
                    "buffer_pl": layout["buffer_pl"]}
