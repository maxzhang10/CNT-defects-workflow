"""Cluster-specific template dispatch and copied LAMMPS inputs."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage'))
import cluster_inputs
import copy_input_dpnegf
import lammps_io
import split_workflow
from ase import Atoms


class ClusterInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_lammps_copies_matching_templates_and_potential(self):
        for device, model in (('cpu', 'CH.airebo-m'), ('gpu', 'test_model.pth')):
            with self.subTest(device=device):
                template = cluster_inputs.input_directory(device, 'lammps', self.root)
                template.mkdir(parents=True)
                for name in ('in.lammps', 'run.sh'):
                    shutil.copy2(cluster_inputs.input_directory(device, 'lammps') / name, template / name)
                (template / model).write_text(device)
                run = self.root / device
                with patch.object(cluster_inputs, '__file__', str(self.root / 'stage/cluster_inputs.py')), \
                        patch.object(lammps_io, 'symlink_force', side_effect=shutil.copy2):
                    lammps_io.prepare_lammps_inputs(
                        temperature=300, structures={'sample': Atoms('C4')}, chirality=(5, 5),
                        l_PL=1, N_uc=4, structure_root=run, lammps_seed=123,
                        md_steps=40, lammps_device=device, fixed_atoms_per_side=1)
                self.assertEqual((run / 'lammps/run.sh').read_bytes(), (template / 'run.sh').read_bytes())
                self.assertEqual((run / 'lammps' / model).read_text(), device)
                text = (run / 'lammps/in.lammps').read_text(encoding='utf-8')
                self.assertIn('airebo/morse' if device == 'cpu' else 'test_model.pth', text)

    def test_dpnegf_cli_default_uses_configured_cluster(self):
        for device in ('cpu', 'gpu'):
            config = self.root / 'config.json'
            config.write_text(json.dumps({'lammps_device': device, 'chirality': [5, 5], 'r_max': 6.5,
                'self_energy_cache': {'use_saved': False, 'save_path': './self_energy'},
                'conductance_options': {'mu': ['Ef']}}))
            argv = ['copy_input_dpnegf.py', '--config', str(config), '--root', str(self.root)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(copy_input_dpnegf, 'copy_inputs_to_leaf_dirs', return_value=0) as copy:
                copy_input_dpnegf.main()
                self.assertEqual(copy.call_args.kwargs['input_dir'], cluster_inputs.input_directory(device, 'dpnegf'))

    def test_b_cluster_selection_preserves_a_configuration(self):
        run = self.root / 'replica_001'
        run.mkdir()
        original = {'lammps_device': 'gpu', 'self_energy_cache': {'save_path': '/B/cache', 'use_saved': True}}
        split_workflow.write_json(run / 'workflow_config.json', original)
        entry = {'replica': 'replica_001', 'configuration': '.'}
        target = split_workflow.freeze_config(self.root, entry, 'cpu')
        saved = json.loads(target.read_text())
        self.assertEqual(saved['lammps_device'], 'cpu')
        self.assertEqual(saved['self_energy_cache'], original['self_energy_cache'])
        self.assertEqual(json.loads((run / 'workflow_config.json').read_text()), original)
        split_workflow.freeze_config(self.root, entry, 'cpu')
        with self.assertRaises(ValueError):
            split_workflow.freeze_config(self.root, entry, 'gpu')

    def test_invalid_cluster_cannot_fall_back_to_another_template(self):
        with self.assertRaises(ValueError):
            cluster_inputs.input_directory('cuda', 'dpnegf')

    def test_preflight_checks_selected_cluster_before_creating_outputs(self):
        cpu = cluster_inputs.input_directory('cpu', 'lammps', self.root)
        gpu = cluster_inputs.input_directory('gpu', 'lammps', self.root)
        cpu.mkdir(parents=True)
        gpu.mkdir(parents=True)
        for name in ('in.lammps', 'run.sh', 'CH.airebo-m'):
            (cpu / name).touch()
        (gpu / 'opt.lammps').touch()
        with self.assertRaisesRegex(FileNotFoundError, 'cpu_cluster'):
            cluster_inputs.validate_templates('cpu', 'lammps', self.root, 'opt')
        self.assertEqual(cluster_inputs.validate_templates('cpu', 'lammps', self.root), cpu)


if __name__ == '__main__':
    unittest.main()
