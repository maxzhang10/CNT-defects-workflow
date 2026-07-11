import argparse
import os
from typing import List, Tuple, Optional

import numpy as np

import logkit as L

# type 映射：1->C, 2->H
TYPE_TO_SPECIES = {
    1: ("C", 1, 6),  # symbol, species_id, Z
    2: ("H", 2, 1),
}


def parse_atoms_header(line: str) -> List[str]:
    parts = line.strip().split()
    if parts[:2] != ["ITEM:", "ATOMS"]:
        raise ValueError(f"Bad ATOMS header: {line}")
    return parts[2:]


def read_box_bounds(fin) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:
    """
    Read 3 lines after 'ITEM: BOX BOUNDS ...'
    Supports:
      orthorhombic: xlo xhi
      triclinic:    xlo xhi xy   / ylo yhi xz / zlo zhi yz

    Returns:
      H (3x3) lattice matrix with columns a,b,c  (in Ang)
      tilts (xy, xz, yz) for debugging
    """
    # x line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS x line")
    xlo, xhi = float(p[0]), float(p[1])
    xy = float(p[2]) if len(p) >= 3 else 0.0

    # y line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS y line")
    ylo, yhi = float(p[0]), float(p[1])
    xz = float(p[2]) if len(p) >= 3 else 0.0

    # z line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS z line")
    zlo, zhi = float(p[0]), float(p[1])
    yz = float(p[2]) if len(p) >= 3 else 0.0

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # LAMMPS triclinic convention in dump:
    # a=(Lx,0,0), b=(xy,Ly,0), c=(xz,yz,Lz)
    a = np.array([Lx, 0.0, 0.0], dtype=float)
    b = np.array([xy, Ly, 0.0], dtype=float)
    c = np.array([xz, yz, Lz], dtype=float)

    H = np.column_stack([a, b, c])  # 3x3, columns are lattice vectors
    return H, (xy, xz, yz, Lx, Ly, Lz)


def wrap_positions_cartesian(pos: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Wrap positions into the unit cell defined by lattice matrix H (columns a,b,c).
    Uses fractional coordinates: s = H^{-1} r ; s = s mod 1 ; r = H s
    pos: (N,3)
    """
    Hinv = np.linalg.inv(H)
    frac = (Hinv @ pos.T).T            # (N,3)
    frac = frac - np.floor(frac)       # mod 1 to [0,1)
    pos_wrapped = (H @ frac.T).T
    return pos_wrapped


def write_siesta_fdf(
    out_path: str,
    H: np.ndarray,
    atoms: List[Tuple[int, int, float, float, float]],
    coord_unit: str = "Ang",
    wrap: bool = False,
    comment: str = "",
) -> None:
    # Sort by id for reproducibility
    atoms = sorted(atoms, key=lambda t: t[0])
    natoms = len(atoms)

    # positions array for optional wrapping
    pos = np.array([[x, y, z] for _, _, x, y, z in atoms], dtype=float)
    if wrap:
        pos = wrap_positions_cartesian(pos, H)

    with open(out_path, "w", encoding="utf-8") as f:
        if comment:
            f.write(f"# {comment}\n")

        f.write(f"LatticeConstant 1.0 {coord_unit}\n\n")

        f.write("%block LatticeVectors\n")
        # H columns are a,b,c -> print as rows: ax ay az etc.
        a = H[:, 0]; b = H[:, 1]; c = H[:, 2]
        f.write(f"  {a[0]:.11g}  {a[1]:.11g}  {a[2]:.11g}\n")   
        f.write(f"  {b[0]:.11g}  {b[1]:.11g}  {b[2]:.11g}\n")
        f.write(f"  {c[0]:.11g}  {c[1]:.11g}  {c[2]:.11g}\n")
        f.write("%endblock LatticeVectors\n\n")

        f.write(f"NumberOfAtoms {natoms}\n\n")

        present_types = sorted({t for _, t, *_ in atoms})
        f.write("%block ChemicalSpeciesLabel\n")
        for t in present_types:
            if t not in TYPE_TO_SPECIES:
                raise ValueError(f"Unknown atom type {t} in {out_path}")
            sym, sid, Z = TYPE_TO_SPECIES[t]
            f.write(f"  {sid}  {Z}  {sym}\n")
        f.write("%endblock ChemicalSpeciesLabel\n\n")

        f.write(f"AtomicCoordinatesFormat {coord_unit}\n")
        f.write("%block AtomicCoordinatesAndAtomicSpecies\n")
        for i, (aid, t, _x, _y, _z) in enumerate(atoms):
            sym, sid, _ = TYPE_TO_SPECIES[t]
            x, y, z = pos[i]
            f.write(f"  {x:.8f}  {y:.8f}  {z:.8f}  {sid}  # id {aid} {sym}\n")
        f.write("%endblock AtomicCoordinatesAndAtomicSpecies\n")

def find_dump_files(root: str, dump_name: str) -> List[str]:
    
   
    import fnmatch

    has_glob = any(ch in dump_name for ch in ["*", "?", "[", "]"])
    found = []

    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # 可选：跳过输出目录，避免重复扫自己生成的结果
        dirnames[:] = [
            d for d in dirnames
            if d not in {"siesta_structs_batch", "__pycache__"}
        ]

        for fn in filenames:
            if has_glob:
                matched = fnmatch.fnmatch(fn, dump_name)
            else:
                matched = (fn == dump_name)

            if matched:
                found.append(os.path.join(dirpath, fn))

    return sorted(found)


def extract_from_one_dump(
    dump_path: str,
    out_base_dir: str,
    every: int,
    coord_unit: str,
    wrap: bool,
    start: Optional[int],
    end: Optional[int],
    skip_zero: bool,
) -> int:
    written = 0

    with open(dump_path, "r", encoding="utf-8", errors="replace") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(fin.readline().strip())

            line = fin.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"{dump_path}: expected ITEM: NUMBER OF ATOMS, got {line.strip()}")
            natoms = int(fin.readline().strip())

            line = fin.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{dump_path}: expected ITEM: BOX BOUNDS, got {line.strip()}")

            H, (xy, xz, yz, Lx, Ly, Lz) = read_box_bounds(fin)

            cols = parse_atoms_header(fin.readline())
            need = ["id", "type", "x", "y", "z"]
            for k in need:
                if k not in cols:
                    raise ValueError(f"{dump_path}: ATOMS header missing '{k}', got {cols}")

            i_id = cols.index("id")
            i_type = cols.index("type")
            i_x = cols.index("x")
            i_y = cols.index("y")
            i_z = cols.index("z")

            atoms: List[Tuple[int, int, float, float, float]] = []
            for _ in range(natoms):
                parts = fin.readline().split()
                atoms.append((
                    int(parts[i_id]),
                    int(parts[i_type]),
                    float(parts[i_x]),
                    float(parts[i_y]),
                    float(parts[i_z]),
                ))

            ok = (timestep % every == 0)
            if skip_zero and timestep == 0:
                ok = False
            if start is not None and timestep < start:
                ok = False
            if end is not None and timestep > end:
                ok = False

            if ok:
                step_dir = os.path.join(out_base_dir, str(timestep))
                os.makedirs(step_dir, exist_ok=True)

                out_path = os.path.join(step_dir, "STRUCT.fdf")
                write_siesta_fdf(
                    out_path=out_path,
                    H=H,
                    atoms=atoms,
                    coord_unit=coord_unit,
                    wrap=wrap,
                    comment=(f"from {dump_path} timestep {timestep} natoms={natoms} "
                             f"tilt(xy,xz,yz)=({xy},{xz},{yz})"),
                )
                written += 1

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Recursively find dump files and write per-timestep folders (40000/80000/...) containing STRUCT.fdf. Supports orthorhombic and triclinic boxes."
    )
    ap.add_argument("--dump-name", default="traj.dump",
                    help="dump filename to search. exact (traj.dump) or pattern (*.dump). Default: traj.dump")
    ap.add_argument("--root", default=".",
                    help="root folder to scan (default: current folder)")
    ap.add_argument("--outroot", default="siesta_structs_batch",
                    help="output root folder (default: siesta_structs_batch)")
    ap.add_argument("--every", type=int, default=40000,
                    help="extract every N timesteps (default: 40000)")
    ap.add_argument("--coord-unit", default="Ang", choices=["Ang", "Bohr"],
                    help="SIESTA coordinate unit (default: Ang)")
    ap.add_argument("--wrap", action="store_true",
                    help="wrap coordinates into unit cell (works for triclinic too)")
    ap.add_argument("--start", type=int, default=None,
                    help="optional start timestep (inclusive)")
    ap.add_argument("--end", type=int, default=None,
                    help="optional end timestep (inclusive)")
    ap.add_argument("--no-skip-zero", action="store_true",
                    help="do NOT skip timestep 0 (default: skip 0)")
    args = ap.parse_args()

    dumps = find_dump_files(args.root, args.dump_name)
    if not dumps:
        raise SystemExit(f"[ERROR] 未找到匹配 '{args.dump_name}' 的 dump 文件，目录: {os.path.abspath(args.root)}")

    total_frames = 0
    L.info(f"找到 dump 文件: {len(dumps)} 个")

    root_abs = os.path.abspath(args.root)
    outroot_abs = os.path.abspath(args.outroot)

    for dump_path in dumps:
        dump_dir = os.path.dirname(os.path.abspath(dump_path))
        rel_dir = os.path.relpath(dump_dir, root_abs)
        out_base_dir = os.path.join(outroot_abs, rel_dir)

        n = extract_from_one_dump(
            dump_path=dump_path,
            out_base_dir=out_base_dir,
            every=args.every,
            coord_unit=args.coord_unit,
            wrap=args.wrap,
            start=args.start,
            end=args.end,
            skip_zero=(not args.no_skip_zero),
        )
        total_frames += n
        L.ok(f"{dump_path} -> {out_base_dir}/<timestep>/STRUCT.fdf   frames={n}")

    L.info(f"共写出帧数: {total_frames}")
    L.info(f"输出根目录: {outroot_abs}")


if __name__ == "__main__":
    main()
