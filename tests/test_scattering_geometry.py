import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage"))
sys.path.insert(0, str(ROOT))
import batch_config
import batch_generate
import band_gap
import ele_multi_defects_ele as generator
import run_multi
import split_workflow
from dump2fdf_batch import write_siesta_fdf
from fdf2xyz import parse_struct_fdf


def test_scattering_replica_generation_and_handoff(tmp_path, monkeypatch):
    batch_path = tmp_path / "batch.py"
    batch_path.write_text(
        'geometry_mode = "scattering_only"\ntemperatures = [300]\n'
        'lammps_device = "cpu"\ntemplate = {"md_steps": 0}\n'
        'configs = [dict(m=5, n=5, l_def=6, N_defects=1, structures=["MVH"], replicas=1)]\n'
    )
    batch = batch_config.load(batch_path)
    task, = batch_generate.build_tasks(batch, tmp_path)
    config = batch_generate.make_task_config(task, batch)
    run = task["run_root"]
    run.mkdir(parents=True)
    config_path = run / "workflow_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("CNT_CONFIG", str(config_path))
    monkeypatch.setenv("CNT_STRUCTURE_ROOT", str(run))
    generator.main()

    atoms = read(run / "lammps/POSCAR")
    assert config["geometry_mode"] == "scattering_only"
    assert task["length"] == 6
    assert atoms.cell[2, 2] == pytest.approx(6 * task["T"])
    assert atoms.get_chemical_symbols() == ["C"] * (6 * task["N_uc"] - 1) + ["H"] * 3
    log = json.loads((run / "defects.json").read_text(encoding="utf-8"))
    assert 1.42 <= log["defect_log"][0]["z"] <= atoms.cell[2, 2] - 1.42

    # Supply only the external POSCAR converter. Run the real XYZ export,
    # including the orchestration that normally swaps the two left PLs.
    original_run = run_multi.run_cmd
    def run_cmd(cmd, **kwargs):
        if cmd[0] == "sgeom":
            rows = [(i, 1 if atom.symbol == "C" else 2, *atom.position)
                    for i, atom in enumerate(atoms, 1)]
            write_siesta_fdf(str(kwargs["cwd"] / "STRUCT.fdf"), atoms.cell.array, rows)
        else:
            original_run(cmd, **kwargs)
    monkeypatch.setattr(run_multi, "run_cmd", run_cmd)
    monkeypatch.setattr(run_multi.shutil, "which", lambda name: name)
    monkeypatch.setattr(sys, "argv", ["run_multi.py", "--config", str(config_path),
                                      "--root", str(run), "--phase", "lammps"])
    run_multi.main()
    fdf = parse_struct_fdf(run / "dpnegf/STRUCT.fdf")
    xyz = read(run / "dpnegf/5_5.xyz")
    assert xyz.pbc.tolist() == [False, False, True]
    header = (run / "dpnegf/5_5.xyz").read_text().splitlines()[1]
    assert header.split('"', 2)[2] == ' Properties=species:S:1:pos:R:3 pbc="F F T"'
    np.testing.assert_array_equal(xyz.numbers, fdf.numbers)
    np.testing.assert_allclose(xyz.positions, fdf.positions, atol=1e-7)
    split_workflow.export_manifest(tmp_path, [task])
    with pytest.raises(ValueError, match="--phase lammps"):
        split_workflow.load_manifest(tmp_path, imported=True)

    # Run the B batch entry twice with a stand-in for the CPU Slurm job.
    templates = tmp_path / "software/input_files/cpu_cluster/band_gap"
    templates.mkdir(parents=True)
    template = ROOT / "input_files/cpu_cluster/band_gap"
    for name in ("run.sh", "run.py", "input.json"):
        (templates / name).write_bytes((template / name).read_bytes())
    (templates / "electronic.pth").write_bytes(b"test checkpoint")
    jobs = []
    def submit(leaf, **kwargs):
        jobs.append(leaf)
        assert (leaf / "model.pth").read_bytes() == b"test checkpoint"
        assert (leaf / "structure.xyz").read_bytes() == (run / "dpnegf/5_5.xyz").read_bytes()
        expected = json.loads((template / "input.json").read_bytes())
        expected["task_options"]["occupation_options"]["nel_atom"]["H"] = 1.0
        assert json.loads((leaf / "input.json").read_bytes()) == expected
        (leaf / "output/results").mkdir(parents=True)
        (leaf / "output/results/gap.json").write_text('{"band_gap_eV": 0.0}')
        (leaf / "band_gap_done.flag").touch()
        return "123"
    monkeypatch.setattr(band_gap, "sbatch_submit_with_retry", submit)
    monkeypatch.setattr(band_gap, "wait_for_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["batch_generate.py", "--config", str(batch_path),
                                      "--workflow-dir", str(tmp_path / "software"),
                                      "--phase", "band_gap", "--scheduler", "slurm"])
    assert batch_generate.main() == batch_generate.main() == 0
    assert jobs == [run / "band_gap/final"]
    (templates / "electronic.pth").write_bytes(b"changed checkpoint")
    assert batch_generate.main() == 1


def test_band_gap_pure_carbon_keeps_template(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    options = {"task_options": {"occupation_options": {"nel_atom": {"C": 4.0}, "charge": 2}}}
    (templates / "input.json").write_text(json.dumps(options))
    (templates / "run.py").write_bytes((ROOT / "input_files/cpu_cluster/band_gap/run.py").read_bytes())
    (templates / "run.sh").touch()
    (templates / "test.pth").touch()
    xyz = tmp_path / "5_5.xyz"
    xyz.write_text('1\nProperties=species:S:1:pos:R:3\nC 0 0 0\n')
    leaf = tmp_path / "final"
    band_gap.prepare_inputs(leaf, xyz, templates)
    assert json.loads((leaf / "input.json").read_bytes()) == options
    assert "NEL_ATOM = {'C': 4.0}\n" in (leaf / "run.py").read_text()


def test_band_gap_python_api_receives_replica_options(tmp_path, monkeypatch, capsys):
    template = ROOT / "input_files/cpu_cluster/band_gap"
    for name in ("run.py", "run.sh", "input.json"):
        (tmp_path / name).write_bytes((template / name).read_bytes())
    (tmp_path / "electronic.pth").touch()
    xyz = tmp_path / "source.xyz"
    xyz.write_text("2\n\nC 0 0 0\nH 0 0 1\n")
    leaf = tmp_path / "replica"
    band_gap.prepare_inputs(leaf, xyz, tmp_path)
    assert "python -u run.py" in (leaf / "run.sh").read_text()
    assert "NEL_ATOM = {'C': 4.0, 'H': 1.0}\n" in (leaf / "run.py").read_text()

    def calculator(**kwargs):
        assert Path(kwargs["checkpoint"]) == leaf / "model.pth"
        assert Path(kwargs["structure"]) == leaf / "structure.xyz"
        assert Path(kwargs["results_path"]) == leaf / "output/results"
        expected = json.loads((leaf / "input.json").read_bytes())["task_options"]
        expected.pop("task")
        expected["cache_options"]["path"] = str(leaf / "output/results/hs_blocks.npz")
        assert kwargs["task_options"] == expected
        result = dict(gap_status="positive_gap", signed_gap_eV=1, band_gap_eV=1,
                      minimum_direct_gap_eV=1, vbm_eV=0, cbm_eV=1, vbm_k_fractional=[0],
                      cbm_k_fractional=[0], converged=True, requested_resolution_eV=0.01,
                      solver="shift-invert", cache=dict(hit=False, spectrum_hits=0, spectrum_misses=0),
                      k_convergence=[])
        return SimpleNamespace(compute=lambda: result)

    monkeypatch.setitem(sys.modules, "dpnegf.runner.cal_bandgap",
                        SimpleNamespace(BandGapCalculator=calculator))
    runpy.run_path(str(leaf / "run.py"))
    assert "signed gap       : 1.0000 eV" in capsys.readouterr().out


@pytest.mark.parametrize("phase", ["all", "dpnegf"])
def test_incompatible_phase_rejected_before_creating_root(tmp_path, monkeypatch, phase):
    config = tmp_path / "config.json"
    config.write_text('{"geometry_mode":"scattering_only"}', encoding="utf-8")
    root = tmp_path / "outputs"
    monkeypatch.setattr(sys, "argv", ["run_multi.py", "--config", str(config),
                                      "--root", str(root), "--phase", phase])
    with pytest.raises(ValueError, match="--phase lammps"):
        run_multi.main()
    assert not root.exists()
