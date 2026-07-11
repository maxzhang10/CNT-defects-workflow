# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an automated simulation pipeline for computing electronic transport through defective carbon nanotubes (CNTs). It chains four external tools end-to-end:

1. **CNT + defect geometry generation** (ASE) → writes `POSCAR`, `data.lmp`, and LAMMPS inputs
2. **LAMMPS** MD annealing → produces `dump` trajectory frames at finite temperature
3. **dump → SIESTA `.fdf` → `.xyz`** conversion → per-timestep atomic structures
4. **DPNEGF** (DeePTB non-equilibrium Green's function) transport calculation → `output/negf.out.pth`

All steps run *locally* on the current node (no SLURM scheduler is used by the drivers; the `#SBATCH` header in `input_files/dpnegf/run.sh` is inert since `sub_dpnegf.py` invokes it via `bash run.sh`).

## Commands

Everything is driven by two orchestrators at the repo root. There is no build step, no test suite, and no linter — this is a plain Python + external-binary pipeline.

```bash
# Single/double-defect pipeline (uses config.json + stage/ele_defects_ele.py)
python run.py

# Multi-defect pipeline (uses config_multi.json + stage/ele_multi_defects_ele.py)
python run_multi.py

# Preview commands without executing
python run.py --dry-run

# Skip geometry+LAMMPS, only do dump→fdf→xyz→copy (LAMMPS dumps must already exist)
python run.py --skip-ele

# Generate DPNEGF inputs but do not run DPNEGF
python run.py --skip-dpnegf

# Override the data root or config
python run.py --root /path/to/data_root --config /path/to/config.json

# Inspect pipeline progress across all cases (read-only, no side effects)
python workflow_status.py --config config.json
```

`run.py` and `run_multi.py` are near-identical; they differ only in which `ele_*` script and default config they use (`run_multi.py` also cleans DPNEGF caches against `root` rather than each `workdir`). Keep changes to shared helpers in sync between the two.

### Running a single stage script directly

Stage scripts under `stage/` are standalone CLIs and can be run in isolation for debugging. Note they `import cnt_geometry`, `defects`, etc. as top-level modules, so **run them from inside `stage/`** (or with `stage/` on `PYTHONPATH`):

```bash
cd stage
python dump2fdf_batch.py --root <dump_dir> --outroot <dpnegf_dir> --every 40000
python fdf2xyz.py --root <dpnegf_dir> --outroot <dpnegf_dir> --config ../config.json
python copy_input_dpnegf.py --root <dpnegf_dir> --input-dir ../input_files/dpnegf --config ../config.json
```

## Configuration

`config.json` / `config_multi.json` are the single source of truth. Key fields consumed across stages:

- `temperature`, `chirality` (`[m, n]`), `r_max`, `data_root` — geometry + output location
- `structures` — which defect structures to build (subset of the keys in `ele_defects_ele.py`'s `all_structures`: `P`, `MVH`, `DV`, `5775`, `MVH_MVH`, `MVH_DV`, `MVH_5775`, `DV_DV`, `DV_5775`, `5775_5775`)
- `l_def` — number of unit cells between/around defects (feeds `cnt_geometry.geo_info`)
- `N_defects` (multi only) — defect count for `ele_multi_defects_ele.py`

The config path propagates to every stage via the `CNT_CONFIG` environment variable, which `run.py` sets before spawning subprocesses. Stage scripts resolve config in this priority order: `--config` flag → `CNT_CONFIG` env → `work_flow_test/config.json` default. When editing a stage script's config loading, preserve this precedence.

## Data layout (the implicit contract)

The entire pipeline coordinates through a directory convention rooted at `data_root`. This layout is *not* configurable — it is hard-coded in path-building logic across `ele_defects_ele.py`, `run.py`'s directory-discovery functions, and `workflow_status.py`:

```
<data_root>/<temperature>K/<m>_<n>/<STRUCTURE>/
    ├── lammps/                 # LAMMPS workdir: POSCAR, data.lmp, in.lammps, CH.airebo-m, run.sh, dump, *_done.flag
    └── dpnegf/<timestep>/       # one DPNEGF workdir per extracted MD frame (e.g. 40000, 80000)
            ├── STRUCT.fdf       # from dump2fdf_batch.py
            ├── <m>_<n>.xyz      # from fdf2xyz.py
            ├── input.json, run.py, run.sh, nnenv*.pth   # from copy_input_dpnegf.py
            └── output/negf.out.pth
```

`run.py` discovers work by walking this tree with heuristic predicates (`find_structure_dirs`, `find_lammps_workdirs`, `find_dpnegf_workdirs_from_structures`). A directory "counts" as a given stage's workdir only if specific files exist — e.g. a DPNEGF leaf requires `input.json` + `run.py` + `run.sh` + `*.xyz` + `nnenv*.pth` all present. If you change any output filename, update these predicates or discovery silently skips the directory.

## Idempotency & re-runs (flag files)

Stages are resumable via sentinel flag files rather than a state DB:

- `lammps_started/submitted/done/failed.flag`, `dpnegf_started/submitted/done/failed.flag`
- `*_done.flag` → the stage is **skipped** on re-run
- `*_failed.flag` → the stage is **not** auto-retried; you must delete the flag to force a rerun
- A dangling `*_submitted.flag` without a `done` flag → treated as an interrupted run and skipped with a warning

After each DPNEGF run, `clean_dpnegf_output_cache()` deletes large regenerable caches (`output/self_energy/`, `output/HS_*.h5`) to save disk. `workflow_status.py` reads these same flags/outputs to classify each stage as DONE / FAILED / PARTIAL / SUSPICIOUS / MISSING (SUSPICIOUS = done-flag present but expected output missing).

## Stage module responsibilities (`stage/`)

- `ele_defects_ele.py` / `ele_multi_defects_ele.py` — top-level geometry builders (structured as `# %%` notebook cells). Build clean CNT, insert defects, write `POSCAR` + `data.lmp` + LAMMPS inputs. **These execute geometry at import/module scope**, not behind `main()`.
- `cnt_geometry.py` — CNT construction, cylindrical-coordinate helpers, `geo_info(m, n, r_max, l_def) -> (T, N_uc, l_PL, length)` which defines principal-layer (PL) sizing used everywhere for lead/device partitioning.
- `defects.py` — defect generators: `CNT_MVH` (monovacancy+H), `CNT_DV` (divacancy), `CNT_5775` (Stone-Wales-like 5-7-7-5).
- `exporters.py` — `write_poscar`, `write_lammps`, `reposition_hydrogens` (moves H atoms out of the lead PLs).
- `lammps_io.py` — templates `in.lammps` per structure: sets `nfix` (frozen lead atoms = `4*l_PL*N_uc`), thermostat temperature, and adds `mass 2` / `C H` pair_coeff for H-containing structures.
- `dump2fdf_batch.py` — parses LAMMPS dumps (orthorhombic + triclinic boxes), writes one `STRUCT.fdf` per selected timestep (`--every`, default 40000; timestep 0 skipped by default).
- `fdf2xyz.py` — parses `STRUCT.fdf` → ASE → `<m>_<n>.xyz` with a NanoTCAD-ViDES-style `Lattice="..."` header. Optionally swaps the first two left PLs' atom ordering (`swap_left_two_pl_order`) so lead atoms are contiguous for DPNEGF.
- `copy_input_dpnegf.py` — copies DPNEGF templates into each leaf, **symlinks** `.pth` models (never copies), and rewrites `input.json` `stru_options` lead_L/device/lead_R atom-index ranges + `run.py`'s `model_path`/`structure` based on computed electrode sizes.
- `sub_lmps.py` / `sub_dpnegf.py` — thin local runners: validate required files, `bash run.sh` in the workdir, capture stdout/stderr, write flag files.

## Gotchas

- **Chirality is read from `config.json`, not parsed from file paths or FDF headers** in the current code path (`fdf2xyz.py`). Older helpers like `get_chirality_from_fdf` still exist but the main flow passes chirality explicitly — do not reintroduce path-based `\d+_\d+` regex parsing, which would misparse structure names like `5775_5775`.
- The DPNEGF model is expected as **exactly one** `*.pth` in `input_files/dpnegf/`; multiple `.pth` files raise an error unless `--model-file` is given.
- `input_files/dpnegf/run.py` ships with hard-coded example paths (`model_path`, `structure`); `copy_input_dpnegf.py` rewrites these per-leaf, so edits to those literals in the template are overwritten downstream.
- Nearly all inline comments and error messages are in Chinese; match that convention when editing existing files.
- `data/`, `data_*/`, `*.pth`, `*.xyz`, `*.dump`, and all NEGF/DFT outputs are gitignored — generated artifacts are not tracked.
