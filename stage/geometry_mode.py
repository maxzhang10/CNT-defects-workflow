"""Geometry selection shared by batch, replica and transfer entry points."""


def geometry_mode(config, phase=None):
    mode = config.get("geometry_mode", "with_electrodes")
    if mode not in ("with_electrodes", "scattering_only"):
        raise ValueError(f"未知 geometry_mode: {mode!r}")
    if mode == "scattering_only":
        if config.get("structure_mode", "defects") != "defects":
            raise ValueError("scattering_only 只支持缺陷生成模式，不能与 assemble 组合")
        if phase is not None and phase != "lammps":
            raise ValueError("geometry_mode=scattering_only 仅支持 --phase lammps")
    return mode
