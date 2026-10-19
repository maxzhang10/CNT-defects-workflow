"""Small local checks for module-cell length, planning and batch loading."""

import importlib.util
import json
import random
from collections import Counter
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

    def test_seed_streams_depend_on_length_replica_and_purpose(self):
        first = assemble.derive_seeds(20260705, 6, 1)
        self.assertEqual(first, assemble.derive_seeds(20260705, 6, 1))
        variants = [assemble.derive_seeds(20260705, length, replica)
                    for length in (6, 8) for replica in (1, 2)]
        values = [item[key] for item in variants
                  for key in ("placement_seed", "rotation_seed")]
        self.assertEqual(len(set(values)), 8)

    def test_minimal_joint_sampling_has_no_position_angle_lock(self):
        self.cfg["modules"]["defects"]["X"]["uc_count"] = 1
        self.cfg["scatter_uc_count"] = 2
        joint = Counter()
        for replica in range(1, 2001):
            seeds = assemble.derive_seeds(20260705, 2, replica)
            sequence, _ = assemble.plan(
                self.cfg, 0, placement_seed=seeds["placement_seed"])
            angle = random.Random(seeds["rotation_seed"]).choice([0, 180])
            joint[(sequence.index("X"), angle)] += 1
        self.assertEqual(len(joint), 4)
        for count in joint.values():
            self.assertTrue(350 < count < 650, joint)

    def test_defect_dictionary_order_does_not_change_plan(self):
        self.cfg["modules"]["defects"]["Y"] = self.cfg["modules"]["defects"]["X"]
        self.cfg["scatter_uc_count"] = 6
        self.cfg["defect_counts"] = {"X": 1, "Y": 1}
        expected = assemble.plan(self.cfg, 42)
        self.cfg["defect_counts"] = {"Y": 1, "X": 1}
        self.assertEqual(expected, assemble.plan(self.cfg, 42))

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
        import batch_generate
        import negf_provenance
        tasks = batch_generate.build_tasks(batch, batch["data_root"])
        config = batch_generate.make_task_config(tasks[0], batch)
        seeds = assemble.derive_seeds(batch["base_structure_seed"], 3, 1)
        for key, value in seeds.items():
            self.assertEqual(config[key], value)
        self.assertEqual(tasks[0]["sequence"], assemble.plan(
            config["assembly"], config["seed"],
            placement_seed=config["placement_seed"])[0])
        self.assertNotEqual(tasks[0]["rotation_seed"], tasks[1]["rotation_seed"])
        old_hash = negf_provenance.compute_negf_config_hash(config)
        config["rotation_seed"] += 1
        self.assertNotEqual(old_hash, negf_provenance.compute_negf_config_hash(config))
        raw["configs"][0]["defect_modules"]["X"] = [str(source), 2]
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cell length"):
            batch_config.load(path)

    def test_balanced_mixture_tasks_and_prefix_stability(self):
        import batch_generate
        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        entry = {"chirality": [8, 2], "perfect_module": str(source),
                 "defect_modules": {"MVH2": [str(source), 1], "MVH1": [str(source), 1]},
                 "defect_mix_mode": "balanced_count", "total_defects": 5,
                 "scatter_uc_count": 6, "rotation_angles_deg": [0], "replicas": 3}
        raw = {"structure_mode": "assemble", "temperatures": [100],
               "configs": [entry]}
        path = self.root / "batch.json"

        def tasks():
            path.write_text(json.dumps(raw), encoding="utf-8")
            batch = batch_config.load(path)
            return batch, batch_generate.build_tasks(batch, self.root)

        batch, first = tasks()
        expected = [{"MVH1": 3, "MVH2": 2}, {"MVH1": 2, "MVH2": 3},
                    {"MVH1": 3, "MVH2": 2}]
        self.assertEqual([t["assembly"]["defect_counts"] for t in first], expected)
        self.assertEqual(batch["configs"][0]["defect_counts"], expected[0])
        for task, counts in zip(first, expected):
            self.assertEqual(task["N_defects"], 5)
            self.assertEqual(Counter(task["sequence"]), Counter({**counts, "perfect": 1}))
            config = batch_generate.make_task_config(task, batch)
            self.assertEqual(config["assembly"]["defect_counts"], counts)
            _, metadata = assemble.build(
                config["assembly"], config["seed"],
                placement_seed=config["placement_seed"], rotation_seed=config["rotation_seed"])
            self.assertEqual(metadata["defect_counts"], counts)
            self.assertEqual(metadata["N_defects"], 5)
        entry["replicas"] = 6
        entry["defect_modules"] = dict(reversed(list(entry["defect_modules"].items())))
        _, extended = tasks()
        for before, after in zip(first, extended):
            for key in ("sequence", "run_root", "placement_seed", "rotation_seed", "assembly"):
                if key == "assembly":
                    self.assertEqual(before[key]["defect_counts"], after[key]["defect_counts"])
                else:
                    self.assertEqual(before[key], after[key])
        self.assertEqual(sum(t["assembly"]["defect_counts"]["MVH1"] for t in extended), 15)
        for total in (0, 1, 4):
            entry["total_defects"] = total
            _, generated = tasks()
            for task in generated:
                counts = task["assembly"]["defect_counts"]
                self.assertEqual(sum(counts.values()), total)
                self.assertLessEqual(abs(counts["MVH1"] - counts["MVH2"]), 1)
        entry["total_defects"] = 5
        entry["defect_counts"] = {"MVH1": 3, "MVH2": 2}
        with self.assertRaisesRegex(ValueError, "cannot specify defect_counts"):
            tasks()
        del entry["defect_counts"]
        entry["module_sequence"] = []
        with self.assertRaisesRegex(ValueError, "cannot specify module_sequence"):
            tasks()
        del entry["module_sequence"]
        entry["total_defects"] = 7
        with self.assertRaisesRegex(ValueError, "exceeding"):
            tasks()
        entry["total_defects"] = 5
        doubled = self.root / "two_uc"
        poscar(doubled, 2 * assemble.module_metadata(source)["length_angstrom"], [0.2])
        entry["defect_modules"]["MVH2"] = [str(doubled), 2]
        with self.assertRaisesRegex(ValueError, "equal module UC counts"):
            tasks()
        del entry["defect_modules"]["MVH2"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            tasks()

    def test_multi_module_balanced_mixture(self):
        import batch_generate
        source = STAGE.parent / "input_files" / "uc_POSCAR" / "8_2" / "POSCAR"
        for size in (3, 4, 5):
            names = [f"MVH{i}" for i in range(size)]
            for total in (0, 1, size - 1, size, size + 1, 2 * size - 1):
                entry = {"chirality": [8, 2], "perfect_module": str(source),
                         "defect_modules": {name: [str(source), 1] for name in names},
                         "defect_mix_mode": "balanced_count", "total_defects": total,
                         "scatter_uc_count": 2 * size, "rotation_angles_deg": [0],
                         "replicas": size + 1}
                raw = {"structure_mode": "assemble", "temperatures": [100], "configs": [entry]}
                path = self.root / "batch.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                batch = batch_config.load(path)
                first = batch_generate.build_tasks(batch, self.root)
                cumulative = Counter(dict.fromkeys(names, 0))
                for task in first:
                    counts = task["assembly"]["defect_counts"]
                    self.assertEqual(sum(counts.values()), total)
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                    cumulative.update(counts)
                    self.assertLessEqual(max(cumulative.values()) - min(cumulative.values()), 1)
                    self.assertEqual(Counter(task["sequence"]),
                                     Counter({**counts, "perfect": 2 * size - total}))
                    config = batch_generate.make_task_config(task, batch)
                    self.assertEqual(config["assembly"]["defect_counts"], counts)
                entry["replicas"] += size
                entry["defect_modules"] = dict(reversed(list(entry["defect_modules"].items())))
                path.write_text(json.dumps(raw), encoding="utf-8")
                extended = batch_generate.build_tasks(batch_config.load(path), self.root)
                for before, after in zip(first, extended):
                    for key in ("sequence", "run_root", "placement_seed", "rotation_seed"):
                        self.assertEqual(before[key], after[key])
                    self.assertEqual(before["assembly"]["defect_counts"], after["assembly"]["defect_counts"])
                if size == 4 and total == 5:
                    config = batch_generate.make_task_config(first[0], batch)
                    _, metadata = assemble.build(config["assembly"], config["seed"],
                        placement_seed=config["placement_seed"], rotation_seed=config["rotation_seed"])
                    self.assertEqual(metadata["defect_counts"], dict(zip(names, (2, 1, 1, 1))))
                    self.assertEqual(metadata["N_defects"], 5)

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
    def test_build_uses_separate_explicit_streams(self):
        self.cfg["rotation_angles_deg"] = [0, 360]
        atoms, info = assemble.build(self.cfg, 4, placement_seed=11, rotation_seed=22)
        again, repeated = assemble.build(self.cfg, 4, placement_seed=11, rotation_seed=22)
        self.assertEqual(info, repeated)
        self.assertTrue((atoms.positions == again.positions).all())
        _, changed_rotation = assemble.build(self.cfg, 4, placement_seed=11, rotation_seed=23)
        self.assertEqual(info["module_sequence"], changed_rotation["module_sequence"])
        _, changed_placement = assemble.build(self.cfg, 4, placement_seed=12, rotation_seed=22)
        defect_angles = lambda meta: [p["rotation_deg"] for p in meta["placements"]
                                     if p["module"] != "perfect"]
        self.assertEqual(defect_angles(info), defect_angles(changed_placement))
        self.assertEqual(defect_angles(info), [random.Random(22).choice([0, 360])])
        self.assertEqual(info["placement_seed"], 11)
        self.assertEqual(info["rotation_seed"], 22)

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
        streams = {key: config[key] for key in ("placement_seed", "rotation_seed")}
        atoms_a, info_a = assemble.build(config["assembly"], config["seed"], **streams)
        atoms_b, info_b = assemble.build(config["assembly"], config["seed"], **streams)
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
        self.assertEqual(saved, info_a)
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
