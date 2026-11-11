"""Cross-cluster handoff and retry integrity regression tests."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage'))
sys.path.insert(0, str(ROOT))
import split_workflow as split
from dump2fdf_batch import write_siesta_fdf
from fdf2xyz import FDF_to_xyz
import numpy as np


class SplitWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / '300K/5_5/test/replica_001'
        self.leaf = self.run / 'dpnegf/40000'
        self.leaf.mkdir(parents=True)
        self.config = {'chirality': [5, 5], 'md_steps': 40000, 'lammps_mode': 'md',
                       'self_energy_cache': {'use_saved': True, 'save_path': '/cluster-b/cache'},
                       'temperature': 300, 'lammps_repeats': 1}
        split.write_json(self.run / 'workflow_config.json', self.config)
        (self.run / 'lammps').mkdir()
        (self.run / 'lammps/lammps_done.flag').touch()
        write_siesta_fdf(str(self.leaf / 'STRUCT.fdf'), np.eye(3) * 10,
                        [(1, 1, 1., 2., 3.), (2, 1, 2., 3., 4.)])
        FDF_to_xyz(str(self.leaf / 'STRUCT.fdf'), chirality=(5, 5), swap_left_pl=False)
        self.tasks = [{'run_root': self.run, 'configuration_root': self.run.parent}]

    def export(self):
        split.export_manifest(self.root, self.tasks)
        return split.load_manifest(self.root)

    def test_only_portable_inputs_are_exported(self):
        manifest = self.export()
        self.assertEqual(len(manifest['files']), 3)
        self.assertFalse(any('dump' in name or 'flag' in name for name in manifest['files']))
        other = self.root / 'cluster-b'
        other.mkdir()
        for name in [*manifest['files'], 'transfer_manifest.json']:
            dest = other / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / name, dest)
        imported = split.load_manifest(other)
        path = split.freeze_config(other, imported['replicas'][0])
        config = json.loads(path.read_text())
        self.assertEqual(config['self_energy_cache'], self.config['self_energy_cache'])
        self.assertEqual(config['structure_root'], str(path.parent))

    def test_missing_or_changed_transfer_is_rejected(self):
        self.export()
        (self.leaf / '5_5.xyz').write_text('corrupted')
        with self.assertRaises(ValueError):
            split.load_manifest(self.root)

    def test_wrong_final_frame_is_rejected(self):
        self.leaf.rename(self.leaf.with_name('20000'))
        with self.assertRaises(ValueError):
            split.export_manifest(self.root, self.tasks)

    def test_dump_conversion_rejects_missing_final_frame(self):
        import dump2fdf_batch
        dump = self.run / 'lammps/traj.dump'
        dump.write_text('ITEM: TIMESTEP\n20000\nITEM: NUMBER OF ATOMS\n1\n'
                        'ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n'
                        'ITEM: ATOMS id type x y z\n1 1 1 2 3\n')
        argv = ['dump2fdf_batch.py', '--root', str(dump.parent), '--outroot', str(self.run / 'dpnegf'),
                '--every', '40000', '--start', '40000', '--end', '40000', '--require-frames']
        with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
            dump2fdf_batch.main()

    def test_incomplete_batch_has_no_manifest(self):
        (self.run / 'lammps/lammps_done.flag').unlink()
        with self.assertRaises(ValueError):
            split.export_manifest(self.root, self.tasks)
        self.assertFalse((self.root / 'transfer_manifest.json').exists())

    def test_source_cache_is_preserved_and_changes_are_rejected(self):
        entry = self.export()['replicas'][0]
        target = split.freeze_config(self.root, entry)
        before = target.read_bytes()
        split.freeze_config(self.root, entry)
        self.assertEqual(before, target.read_bytes())
        self.assertEqual(json.loads(before)['self_energy_cache'], self.config['self_energy_cache'])
        saved = json.loads(before)
        saved['self_energy_cache']['save_path'] = '/different/cache'
        split.write_json(target, saved)
        with self.assertRaises(ValueError):
            split.freeze_config(self.root, entry)

    def test_model_change_cannot_reuse_done_result(self):
        for name in ('input.json', 'run.py', 'expected_negf_config_hash.txt', 'model.pth'):
            (self.leaf / name).write_text('original')
        split.lock_inputs(self.leaf)
        (self.leaf / 'dpnegf_done.flag').touch()
        split.lock_inputs(self.leaf)
        (self.leaf / 'model.pth').write_text('different model')
        with self.assertRaises(ValueError):
            split.lock_inputs(self.leaf)

    def test_import_preserves_replica_configuration(self):
        self.export()
        args = SimpleNamespace(scheduler='slurm', max_parallel=1, skip_conductance=True)
        with patch.object(split.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                patch('cluster_inputs.validate_templates'):
            self.assertEqual(split.run_imported_batch(args, ROOT, self.root), 0)
        command = run.call_args.args[0]
        self.assertIn('dpnegf', command)
        self.assertIn(str(self.run / 'workflow_config.dpnegf.json'), command)
        self.assertEqual(json.loads((self.run / 'workflow_config.json').read_text()), self.config)

    def test_all_transfer_checks_precede_submission(self):
        self.export()
        (self.leaf / 'STRUCT.fdf').unlink()
        args = SimpleNamespace()
        with patch.object(split.subprocess, 'run') as run:
            with self.assertRaises(FileNotFoundError):
                split.run_imported_batch(args, ROOT, self.root)
            run.assert_not_called()

    def test_extra_old_replica_is_rejected_before_submission(self):
        self.export()
        (self.run.parent / 'replica_002').mkdir()
        with self.assertRaisesRegex(ValueError, 'differ from transfer manifest'):
            split.load_manifest(self.root)

    def test_missing_b_templates_do_not_freeze_configuration(self):
        self.export()
        with self.assertRaises(FileNotFoundError):
            split.run_imported_batch(SimpleNamespace(), self.root / 'missing-software', self.root)
        self.assertFalse((self.run / 'workflow_config.dpnegf.json').exists())

    def test_b_local_pipeline_runs_once_and_resumes_completed_result(self):
        import os
        software = self.root / 'software'
        software.mkdir()
        shutil.copy2(ROOT / 'run_multi.py', software / 'run_multi.py')
        shutil.copytree(ROOT / 'stage', software / 'stage', ignore=shutil.ignore_patterns('__pycache__'))
        if os.name == 'nt':
            # Windows without developer/admin privileges cannot create symlinks.
            # Substitute only model distribution in this disposable test copy.
            copier = software / 'stage/copy_input_dpnegf.py'
            source = copier.read_text(encoding='utf-8')
            copier.write_text(source.replace('dst.symlink_to(src)', 'shutil.copy2(src, dst)'), encoding='utf-8')
        template = software / 'input_files/cpu_cluster/dpnegf'
        template.mkdir(parents=True)
        shutil.copy2(ROOT / 'input_files/cpu_cluster/dpnegf/input.json', template / 'input.json')
        (template / 'run.sh').write_text('#!/bin/bash\nexit 1\n')  # Local mode must not use Slurm script.
        (template / 'model.pth').write_text('fake model')
        (template / 'run.py').write_text(
            'from pathlib import Path\nmodel_path = "model.pth"\nstructure = "5_5.xyz"\n'
            'counter = Path("solver_count.txt")\n'
            'counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else "1")\n'
            'Path("output").mkdir(exist_ok=True)\n'
            'try:\n import torch\nexcept ImportError:\n'
            ' Path("output/negf.out.pth").write_bytes(b"fake result")\n'
            'else:\n torch.save({"fake": True}, "output/negf.out.pth")\n')
        self.config.update(lammps_device='gpu', r_max=6.5, l_def=1, structure_mode='assemble',
            conductance_options={'mu': ['Ef']}, save_self_energy=True,
            assembly={'scatter_uc_count': 4, 'buffer_pl': 2, 'defect_counts': {},
                      'layout': {'l_PL_uc': 1, 'atoms_per_pl': 1, 'atoms_per_uc': 1, 'uc_length_angstrom': 2.46}})
        split.write_json(self.run / 'workflow_config.json', self.config)
        write_siesta_fdf(str(self.leaf / 'STRUCT.fdf'), np.eye(3) * 20,
                        [(i + 1, 1, 1., 2., float(i)) for i in range(8)])
        FDF_to_xyz(str(self.leaf / 'STRUCT.fdf'), chirality=(5, 5), atoms_per_pl=1)
        self.export()
        shutil.rmtree(self.run / 'lammps')  # B receives no LAMMPS inputs/dump.
        args = SimpleNamespace(scheduler='local', max_parallel=1, skip_conductance=True)
        # Resolve PYTHONPATH before the solver changes cwd into its leaf directory.
        paths = [str(Path(p).resolve()) for p in sys.path if p]
        with patch.dict(os.environ, {'PYTHONPATH': os.pathsep.join(paths), 'PYTHONIOENCODING': 'utf-8'}):
            for _ in range(2):
                rc = split.run_imported_batch(args, software, self.root, 'cpu')
                self.assertEqual(rc, 0, (self.run / 'workflow.dpnegf.log').read_text(encoding='utf-8'))
        self.assertEqual((self.leaf / 'solver_count.txt').read_text(), '1')
        self.assertTrue((self.leaf / 'dpnegf_done.flag').exists())
        self.assertEqual((self.leaf / 'expected_negf_config_hash.txt').read_text(),
                         (self.leaf / 'output/negf_config_hash.txt').read_text())

    def test_b_reads_data_root_from_batch_py_and_json(self):
        import batch_generate
        import batch_config
        for suffix in ('py', 'json'):
            with self.subTest(suffix=suffix):
                path = self.root / ('batch.' + suffix)
                raw = {'data_root': 'imported data', 'max_parallel': 3, 'lammps_device': 'cpu',
                       'structure_mode': 'assemble', 'configs': [{'module': '/A/not-copied/POSCAR'}]}
                if suffix == 'json':
                    split.write_json(path, raw)
                else:
                    path.write_text('\n'.join(f'{key} = {value!r}' for key, value in raw.items()))
                argv = ['batch_generate.py', '--phase', 'dpnegf', '--config', str(path)]
                with patch.object(sys, 'argv', argv), patch.object(batch_generate, 'build_tasks') as build, \
                        patch.object(batch_config, '_validate_batch') as validate, \
                        patch.object(split, 'run_imported_batch', return_value=0) as imported:
                    self.assertEqual(batch_generate.main(), 0)
                    self.assertEqual(imported.call_args.args[2], (self.root / 'imported data').resolve())
                    self.assertEqual(imported.call_args.args[0].max_parallel, 3)
                    self.assertEqual(imported.call_args.args[3], 'cpu')
                    build.assert_not_called()
                    validate.assert_not_called()
                argv += ['--max-parallel', '7']
                with patch.object(sys, 'argv', argv), patch.object(split, 'run_imported_batch', return_value=0) as imported:
                    self.assertEqual(batch_generate.main(), 0)
                    self.assertEqual(imported.call_args.args[0].max_parallel, 7)

    def test_regular_batch_loaders_still_validate(self):
        import batch_config
        for suffix in ('py', 'json'):
            path = self.root / ('batch.' + suffix)
            if suffix == 'py':
                path.write_text('data_root = "."\n')
            else:
                split.write_json(path, {'data_root': '.'})
            with patch.object(batch_config, '_validate_batch', return_value='validated') as validate:
                self.assertEqual(batch_config.load(path), 'validated')
                self.assertEqual(validate.call_args.args, ({'data_root': '.'}, path.resolve()))

    def test_phase_entrypoints_select_only_the_requested_work(self):
        import run_multi
        from negf_provenance import compute_negf_config_hash
        config = dict(self.config, structure_mode='assemble', assembly={
            'scatter_uc_count': 1, 'layout': {'l_PL_uc': 1, 'atoms_per_pl': 1},
            'buffer_pl': 2, 'defect_counts': {}}, save_self_energy=True)
        path = self.run / 'workflow_config.json'
        split.write_json(path, config)
        lammps = self.run / 'lammps'
        for name in ('data.lmp', 'in.lammps', 'run.sh', 'potential.pth', 'traj.dump'):
            (lammps / name).touch()
        argv = ['run_multi.py', '--root', str(self.run), '--config', str(path),
                '--scheduler', 'local', '--skip-conductance', '--phase', 'lammps']
        with patch.object(sys, 'argv', argv), patch.object(run_multi, 'run_cmd') as commands:
            run_multi.main()
        text = ' '.join(str(call.args[0]) for call in commands.call_args_list)
        self.assertIn('sub_lmps.py', text)
        self.assertIn('dump2fdf_batch.py', text)
        self.assertIn('fdf2xyz.py', text)
        self.assertNotIn('copy_input_dpnegf.py', text)
        self.assertNotIn('sub_dpnegf.py', text)
        # B has only transferred structures and its own model/template directory.
        shutil.rmtree(lammps)
        software = self.root / 'software'
        template = software / 'input_files/gpu_cluster/dpnegf'
        template.mkdir(parents=True)
        (template / 'model.pth').write_text('model')
        def prepare(cmd, **kwargs):
            if 'copy_input_dpnegf.py' in str(cmd):
                for name in ('run.py', 'run.sh', 'input.json', 'model.pth'):
                    (self.leaf / name).write_text('input')
                (self.leaf / 'expected_negf_config_hash.txt').write_text(compute_negf_config_hash(config))
        argv[-1] = 'dpnegf'
        argv += ['--stage', str(ROOT / 'stage')]
        with patch.object(sys, 'argv', argv), patch.object(run_multi, '__file__', str(software / 'run_multi.py')), \
                patch.object(run_multi, 'run_cmd', side_effect=prepare) as commands:
            run_multi.main()
        text = ' '.join(str(call.args[0]) for call in commands.call_args_list)
        self.assertIn('copy_input_dpnegf.py', text)
        self.assertIn('sub_dpnegf.py', text)
        for forbidden in ('sub_lmps.py', 'dump2fdf_batch.py', 'fdf2xyz.py', 'ele_multi_defects_ele.py'):
            self.assertNotIn(forbidden, text)
        self.assertTrue((self.leaf / 'dpnegf_input_fingerprint.json').is_file())

    def test_external_uc_is_materialized_and_relocated(self):
        uc = self.root / 'original_POSCAR'
        uc.write_text('external unit cell')
        self.config.update(uc_file=str(uc), uc_sha256=split.digest(uc))
        split.write_json(self.run / 'workflow_config.json', self.config)
        entry = self.export()['replicas'][0]
        uc.unlink()
        path = split.freeze_config(self.root, entry)
        relocated = Path(json.loads(path.read_text())['uc_file'])
        self.assertEqual(relocated.read_text(), 'external unit cell')

    def test_outside_manifest_path_is_rejected(self):
        manifest = self.export()
        manifest['files']['../outside'] = 'invalid'
        split.write_json(self.root / 'transfer_manifest.json', manifest)
        with self.assertRaises(ValueError):
            split.load_manifest(self.root)


if __name__ == '__main__':
    unittest.main()
