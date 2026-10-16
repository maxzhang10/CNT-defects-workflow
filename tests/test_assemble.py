"""Small local checks for module-cell length, planning and batch loading."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1] / "stage"
sys.path.insert(0, str(STAGE))
import assemble
import batch_config


def poscar(path, length, z_values, xs=None):
    xs = xs or [0.0] * len(z_values)
    lines = ["synthetic module", "1.0", "10 0 0", "0 10 0", f"0 0 {length}",
             "C", str(len(z_values)), "Cartesian"]
    lines += [f"{x} 0 {z}" for x, z in zip(xs, z_values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=STAGE.parent)
        self.root = Path(self.temp.name)
        poscar(self.root / "perfect", 2.0, [0.2, 1.8], [0, 1])
        poscar(self.root / "defect", 4.0, [0.2, 1.8, 2.0, 2.2, 3.8], [0, 1, 0.5, 0, 1])
        self.cfg = {
            "modules": {"perfect": assemble.module_metadata(self.root / "perfect"),
                        "defects": {"X": assemble.module_metadata(self.root / "defect")}},
            "defect_counts": {"X": 1}, "scatter_uc_count": 3,
            "layout": {"side_repeats": 2, "electrode_atoms_per_side": 4,
                       "side_atoms_per_side": 4, "fixed_atoms_per_side": 2,
                       "atoms_per_pl": 2, "l_PL_uc": 1, "buffer_pl": 0,
                       "electrode_length_angstrom": 4.0,
                       "buffer_length_angstrom": 0.0},
            "rotation_angles_deg": [0], "module_sequence": None,
        }
        self.cfg["modules"]["defects"]["X"]["uc_count"] = 2

    def tearDown(self):
        self.temp.cleanup()

    def test_cell_length_not_coordinate_extent(self):
        self.assertEqual(self.cfg["modules"]["defects"]["X"]["length_angstrom"], 4.0)
        self.assertEqual(assemble.plan(self.cfg, 3)[1], 6.0)

    def test_reproducible_plan_and_manual_order(self):
        self.assertEqual(assemble.plan(self.cfg, 7), assemble.plan(self.cfg, 7))
        self.cfg["module_sequence"] = ["perfect", "X"]
        self.assertEqual(assemble.plan(self.cfg, 8)[0], ["perfect", "X"])

    def test_impossible_length_and_wrong_counts_fail(self):
        self.cfg["scatter_uc_count"] = 1
        with self.assertRaisesRegex(ValueError, "exceeding"):
            assemble.plan(self.cfg, 1)
        self.cfg["scatter_uc_count"] = 3
        self.cfg["module_sequence"] = ["perfect", "perfect", "perfect"]
        with self.assertRaisesRegex(ValueError, "defect counts"):
            assemble.plan(self.cfg, 1)

    def test_batch_entry_normalization(self):
        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        period = assemble.module_metadata(source)["length_angstrom"]
        raw = {
            "structure_mode": "assemble", "temperatures": [100], "data_root": ".",
            "template": {"r_max": 6.5, "md_steps": 0},
            "configs": [{"chirality": [8, 2], "perfect_module": str(source),
                         "defect_modules": {"X": [str(source), 1]}, "defect_counts": {"X": 1},
                         "scatter_uc_count": 3, "buffer_pl": 2,
                         "rotation_angles_deg": [0], "replicas": 2}],
        }
        path = self.root / "batch.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        batch = batch_config.load(path)
        self.assertEqual(batch["configs"][0]["layout"]["l_PL_uc"], 1)
        self.assertEqual(batch["configs"][0]["layout"]["side_repeats"], 4)
        self.assertEqual(batch["structure_mode"], "assemble")
        raw["configs"][0]["defect_modules"]["X"] = [str(source), 2]
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cell length"):
            batch_config.load(path)

    def test_perfect_module_must_be_one_uc(self):
        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        perfect = assemble.module_metadata(source)
        doubled = {**perfect, "length_angstrom": 2 * perfect["length_angstrom"],
                   "atom_count": 2 * perfect["atom_count"]}
        with self.assertRaisesRegex(ValueError, "exactly 56"):
            assemble.lead_layout(doubled, 8, 2, 6.5, 2)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is unavailable locally")
    def test_build_electrode_partition(self):
        atoms, meta = assemble.build(self.cfg, 4)
        self.assertEqual(len(atoms), 4 + 7 + 4)
        self.assertEqual(meta["electrode_atoms_per_side"], 4)
        self.assertEqual(meta["atoms_per_pl"], 2)
        self.assertEqual(meta["actual_density_A_inv"], 1 / 6)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is unavailable locally")
    def test_interface_layer_exactly_at_half_uc_is_included(self):
        perfect_path = self.root / "half_layer_perfect"
        defect_path = self.root / "half_layer_defect"
        poscar(perfect_path, 2.0, [0.0, 1.0])
        poscar(defect_path, 6.0, [0.0, 1.0, 2.5, 3.5, 5.0])
        perfect = assemble.module_metadata(perfect_path)
        defect = assemble.module_metadata(defect_path)
        defect["uc_count"] = 3
        self.cfg["modules"] = {"perfect": perfect, "defects": {"X": defect}}
        self.cfg["scatter_uc_count"] = 4
        atoms, meta = assemble.build(self.cfg, 4)
        self.assertEqual(len(atoms), 4 + 7 + 4)
        self.assertEqual(meta["scatter_uc_count"], 4)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is unavailable locally")
    def test_real_8_2_poscar_generation(self):
        """Use the checked-in perfect UC as a placeholder defect module."""
        import numpy as np
        from ase.io import read
        import batch_generate
        from copy_input_dpnegf import update_input_json_for_leaf
        from ele_multi_defects_ele import main_assemble
        from fdf2xyz import swap_left_two_pl_order

        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        if not source.is_file():
            self.skipTest("checked-in (8,2) POSCAR is unavailable")
        meta = assemble.module_metadata(source)
        period = meta["length_angstrom"]
        raw = {
            "structure_mode": "assemble", "temperatures": [100],
            "data_root": str(self.root / "data"),
            "base_structure_seed": 123, "base_lammps_seed": 456,
            "lammps_mode": "md", "template": {"r_max": 6.5, "md_steps": 0},
            "configs": [{
                "chirality": [8, 2], "perfect_module": str(source),
                "defect_modules": {"test_placeholder": [str(source), 1]},
                "defect_counts": {"test_placeholder": 1},
                "scatter_uc_count": 3, "buffer_pl": 2,
                "rotation_angles_deg": [0, 180], "replicas": 1,
            }],
        }
        batch_file = self.root / "real_batch.json"
        batch_file.write_text(json.dumps(raw), encoding="utf-8")
        batch = batch_config.load(batch_file)
        task = batch_generate.build_tasks(batch, batch["data_root"])[0]
        config = batch_generate.make_task_config(task, batch)
        atoms_a, info_a = assemble.build(config["assembly"], config["seed"])
        atoms_b, info_b = assemble.build(config["assembly"], config["seed"])
        self.assertEqual(info_a, info_b)
        self.assertTrue(np.array_equal(atoms_a.positions, atoms_b.positions))
        self.assertEqual(len(atoms_a), 11 * 56)
        self.assertEqual(info_a["electrode_atoms_per_side"], 2 * 56)
        self.assertEqual(info_a["side_atoms_per_side"], 4 * 56)
        self.assertEqual(info_a["fixed_atoms_per_side"], 3 * 56)
        self.assertEqual(info_a["atoms_per_pl"], 56)
        self.assertAlmostEqual(info_a["actual_density_A_inv"], 1 / (3 * period))

        main_assemble(config)
        result_dir = Path(config["structure_root"])
        reread = read(result_dir / "lammps" / "POSCAR", format="vasp")
        self.assertEqual(len(reread), 11 * 56)
        self.assertAlmostEqual(reread.cell[2, 2], 11 * period, places=5)
        swapped = swap_left_two_pl_order(reread, info_a["atoms_per_pl"])
        self.assertTrue(np.array_equal(swapped.positions[:56], reread.positions[56:112]))
        self.assertTrue((result_dir / "lammps" / "data.lmp").is_file())
        self.assertFalse((result_dir / "lammps" / "in.lammps").exists())
        saved = json.loads((result_dir / "assembly.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["N_defects"], 1)
        self.assertEqual(saved["scatter_uc_count"], 3)
        self.assertEqual(len(saved["placements"]), 11)
        leaf = result_dir / "partition_check"
        leaf.mkdir()
        fdf_lines = ["%block AtomicCoordinatesAndAtomicSpecies"]
        fdf_lines += ["0 0 0 1"] * len(reread)
        fdf_lines += ["%endblock AtomicCoordinatesAndAtomicSpecies"]
        (leaf / "STRUCT.fdf").write_text("\n".join(fdf_lines), encoding="utf-8")
        template_dir = STAGE.parent / "input_files" / "dpnegf"
        for filename in ("input.json", "run.py"):
            (leaf / filename).write_bytes((template_dir / filename).read_bytes())
        update_input_json_for_leaf(
            leaf, "model.pth", (8, 2),
            {"use_saved": False, "save_path": "./self_energy/"}, ["Ef"],
            r_max=6.5, assembly=config["assembly"],
        )
        regions = json.loads((leaf / "input.json").read_text(encoding="utf-8"))["task_options"]["stru_options"]
        self.assertEqual(regions["lead_L"]["id"], "0-112")
        self.assertEqual(regions["device"]["id"], "112-504")
        self.assertEqual(regions["lead_R"]["id"], "504-616")
        with self.assertRaises(FileExistsError):
            main_assemble(config)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is unavailable locally")
    def test_multi_uc_defect_poscar_uses_declared_uc_count(self):
        from ase.io import read, write

        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        period = assemble.module_metadata(source)["length_angstrom"]
        doubled_path = self.root / "defect_2uc_POSCAR"
        write(doubled_path, read(source, format="vasp").repeat((1, 1, 2)), format="vasp")
        perfect = assemble.module_metadata(source)
        defect = assemble.module_metadata(doubled_path)
        defect["uc_count"] = 2
        layout = assemble.lead_layout(perfect, 8, 2, 6.5, 2)
        config = {
            "modules": {"perfect": perfect, "defects": {"X": defect}},
            "defect_counts": {"X": 1}, "scatter_uc_count": 3,
            "layout": layout, "rotation_angles_deg": [0], "module_sequence": None,
        }
        atoms, info = assemble.build(config, 13)
        self.assertEqual(layout["l_PL_uc"], 1)
        self.assertEqual(layout["side_repeats"], 4)
        self.assertEqual(info["electrode_atoms_per_side"], 112)
        self.assertEqual(len(atoms), 616)


if __name__ == "__main__":
    unittest.main()
