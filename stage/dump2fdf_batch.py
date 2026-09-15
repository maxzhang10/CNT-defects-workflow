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
      triclinic:    xlo_bound xhi_bound xy  / ylo_bound yhi_bound xz / zlo_bound zhi_bound yz

    LAMMPS dump 的 triclinic 格式给出的是 *包围盒* bound（已含 tilt 偏移），
    不是真实盒边界。必须按 LAMMPS 文档规范反推真实 box bounds：
      xlo = xlo_bound + min(0, xy, xz, xy+xz)
      xhi = xhi_bound + max(0, xy, xz, xy+xz)
      ylo = ylo_bound + min(0, yz)
      yhi = yhi_bound + max(0, yz)
    （z 不受 tilt 影响。）

    Returns:
      H (3x3) lattice matrix with columns a,b,c  (in Ang)
      tilts (xy, xz, yz, Lx, Ly, Lz)
      origin (xlo, ylo, zlo) 也通过实例属性无法返回，故单独提供 read_box_origin。
    """
    # x line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS x line")
    xlo_bound, xhi_bound = float(p[0]), float(p[1])
    xy = float(p[2]) if len(p) >= 3 else 0.0

    # y line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS y line")
    ylo_bound, yhi_bound = float(p[0]), float(p[1])
    xz = float(p[2]) if len(p) >= 3 else 0.0

    # z line
    p = fin.readline().split()
    if len(p) < 2:
        raise ValueError("Bad BOX BOUNDS z line")
    zlo_bound, zhi_bound = float(p[0]), float(p[1])
    yz = float(p[2]) if len(p) >= 3 else 0.0

    # P2-4：从包围盒 bound 反推真实 box bounds。
    has_tilt = (xy != 0.0 or xz != 0.0 or yz != 0.0)
    if has_tilt:
        xlo = xlo_bound + min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound + max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound + min(0.0, yz)
        yhi = yhi_bound + max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
    else:
        xlo, xhi = xlo_bound, xhi_bound
        ylo, yhi = ylo_bound, yhi_bound
        zlo, zhi = zlo_bound, zhi_bound

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # LAMMPS triclinic convention in dump:
    # a=(Lx,0,0), b=(xy,Ly,0), c=(xz,yz,Lz)
    a = np.array([Lx, 0.0, 0.0], dtype=float)
    b = np.array([xy, Ly, 0.0], dtype=float)
    c = np.array([xz, yz, Lz], dtype=float)

    H = np.column_stack([a, b, c])  # 3x3, columns are lattice vectors
    # 把 origin 存到 H 的属性上，供 wrap 使用。
    H_origin = np.array([xlo, ylo, zlo], dtype=float)
    return H, (xy, xz, yz, Lx, Ly, Lz), H_origin


def wrap_positions_cartesian(
    pos: np.ndarray, H: np.ndarray, origin: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Wrap positions into the unit cell defined by lattice matrix H (columns a,b,c).
    Uses fractional coordinates: s = H^{-1} (r - origin) ; s = s mod 1 ; r = origin + H s

    P2-4：wrap 前必须减去 box origin（xlo/ylo/zlo），否则当 dump lower
    bounds 非零时 wrap 结果会发生错误平移。
    pos: (N,3)
    """
    Hinv = np.linalg.inv(H)
    if origin is not None:
        shifted = pos - origin
    else:
        shifted = pos
    frac = (Hinv @ shifted.T).T            # (N,3)
    frac = frac - np.floor(frac)           # mod 1 to [0,1)
    pos_wrapped = (H @ frac.T).T
    if origin is not None:
        pos_wrapped = pos_wrapped + origin
    return pos_wrapped


def write_siesta_fdf(
    out_path: str,
    H: np.ndarray,
    atoms: List[Tuple[int, int, float, float, float]],
    coord_unit: str = "Ang",
    wrap: bool = False,
    comment: str = "",
    origin: Optional[np.ndarray] = None,
) -> None:
    # Sort by id for reproducibility
    atoms = sorted(atoms, key=lambda t: t[0])
    natoms = len(atoms)

    # P2-5：LAMMPS dump 用 units metal（Å）。若输出单位选 Bohr，必须把
    # 晶格和坐标从 Å 换算到 Bohr（÷ ase.units.Bohr = ÷ 0.529177...），
    # 不能只改标签，否则数值被标成 Bohr 但仍是 Å，后续 fdf2xyz 又乘 Bohr，
    # 结构缩小约 0.529 倍。
    scale = 1.0
    if coord_unit == "Bohr":
        try:
            from ase.units import Bohr as ASE_BOHR  # Å per Bohr
        except ImportError:
            # 1 Bohr = 0.529177210903 Å (CODATA 2018)
            ASE_BOHR = 0.529177210903
        scale = 1.0 / ASE_BOHR

    H_out = H * scale

    # positions array for optional wrapping
    pos = np.array([[x, y, z] for _, _, x, y, z in atoms], dtype=float)
    if wrap:
        pos = wrap_positions_cartesian(pos, H, origin=origin)
    pos_out = pos * scale

    with open(out_path, "w", encoding="utf-8") as f:
        if comment:
            f.write(f"# {comment}\n")

        f.write(f"LatticeConstant 1.0 {coord_unit}\n\n")

        f.write("%block LatticeVectors\n")
        # H columns are a,b,c -> print as rows: ax ay az etc.
        a = H_out[:, 0]; b = H_out[:, 1]; c = H_out[:, 2]
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
            x, y, z = pos_out[i]
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
    samples: Optional[int] = None,
) -> int:
    """
    从 dump 文件中提取并写入 FDF 文件。

    Args:
        samples: 如果指定，只保留最后 N 个满足条件的帧（按时间步排序）。
                用于从 MD 轨迹末尾采样多个构型。
    """
    # 第一轮：收集所有满足条件的帧信息
    frames: List[Tuple[int, np.ndarray, Tuple[float, ...], List[Tuple]]] = []

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

            H, tilts, origin = read_box_bounds(fin)
            xy, xz, yz, Lx, Ly, Lz = tilts

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
                frames.append((timestep, H, tilts, atoms, origin))

    # 如果指定了 samples，只保留最后 N 个帧
    if samples is not None and len(frames) > samples:
        frames = frames[-samples:]

    # 第二轮：写入 FDF 文件
    written = 0
    for timestep, H, tilts, atoms, origin in frames:
        xy, xz, yz, Lx, Ly, Lz = tilts
        step_dir = os.path.join(out_base_dir, str(timestep))

        # 已完成的 DPNEGF leaf：不覆盖 STRUCT.fdf。
        # done flag 由 sub_dpnegf.py 写在同一层 leaf(即该 timestep 目录)下。
        if os.path.exists(os.path.join(step_dir, "dpnegf_done.flag")):
            L.skip(f"DPNEGF 已完成，跳过覆盖 STRUCT.fdf: {step_dir}")
            continue

        os.makedirs(step_dir, exist_ok=True)

        out_path = os.path.join(step_dir, "STRUCT.fdf")
        write_siesta_fdf(
            out_path=out_path,
            H=H,
            atoms=atoms,
            coord_unit=coord_unit,
            wrap=wrap,
            comment=(f"from {dump_path} timestep {timestep} natoms={len(atoms)} "
                     f"tilt(xy,xz,yz)=({xy},{xz},{yz})"),
            origin=origin,
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
    ap.add_argument("--samples", type=int, default=None,
                    help="只保留最后 N 个满足条件的帧，用于从 MD 轨迹多次采样 (default: None, 保留所有)")
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
            samples=args.samples,
        )
        total_frames += n
        L.ok(f"{dump_path} -> {out_base_dir}/<timestep>/STRUCT.fdf   frames={n}")

    L.info(f"共写出帧数: {total_frames}")
    L.info(f"输出根目录: {outroot_abs}")


if __name__ == "__main__":
    main()
