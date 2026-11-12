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

    def second_replica(self):
        run = self.run.with_name('replica_002')
        shutil.copytree(self.run, run)
        return {'run_root': run, 'configuration_root': run.parent}

    def test_manifest_grows_without_changing_published_entries(self):
        first = self.export()
        second = self.second_replica()
        split.export_manifest(self.root, [second])
        manifest = split.load_manifest(self.root)
        self.assertEqual(len(manifest['replicas']), 2)
        self.assertEqual(manifest['replicas'][0], first['replicas'][0])
        for name, sha in first['files'].items():
            self.assertEqual(manifest['files'][name], sha)
        before = (self.root / 'transfer_manifest.json').read_bytes()
        self.config['temperature'] = 400
        split.write_json(self.run / 'workflow_config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'Published input contents changed'):
            split.export_manifest(self.root, self.tasks)
        self.assertEqual((self.root / 'transfer_manifest.json').read_bytes(), before)

    def test_a_publishes_before_batch_end_and_preserves_manifest_on_restart(self):
        import batch_generate as batch
        second = self.second_replica()
        tasks = [dict(task, task_id=f'task{i}', config_name='case')
                 for i, task in enumerate([*self.tasks, second])]
        raw = {'data_root': self.root, 'max_parallel': 1, 'configs': []}
        argv = ['batch_generate.py', '--config', 'unused.json', '--phase', 'lammps',
                '--workflow-dir', str(ROOT), '--scheduler', 'local']
        def ordered_completed(futures):
            for i, future in enumerate(futures):
                if i == 1:
                    manifest = json.loads((self.root / 'transfer_manifest.json').read_text())
                    self.assertEqual(len(manifest['replicas']), 1)
                yield future
        with patch.object(sys, 'argv', argv), patch.object(batch.bc, 'load', return_value=raw), \
                patch.object(batch.bc, 'format_resolution_table', return_value=None), \
                patch.object(batch, 'build_tasks', return_value=tasks), \
                patch.object(batch, 'make_task_config', return_value=self.config), \
                patch.object(batch, 'as_completed', side_effect=ordered_completed), \
                patch.object(batch, 'run_single_task', side_effect=lambda task, *args:
                             (task['task_id'], 0 if task['task_id'] == 'task0' else 1, 'TEST')) as run:
            self.assertEqual(batch.main(), 1)
            before = (self.root / 'transfer_manifest.json').read_bytes()
            run.reset_mock()
            self.assertEqual(batch.main(), 1)
            self.assertEqual([call.args[0]['task_id'] for call in run.call_args_list], ['task1'])
            self.assertEqual((self.root / 'transfer_manifest.json').read_bytes(), before)
            run.side_effect = lambda task, *args: (task['task_id'], 0, 'OK')
            self.assertEqual(batch.main(), 0)
        self.assertEqual(len(split.load_manifest(self.root)['replicas']), 2)

    def test_partial_delivery_defers_collection_until_all_replicas_arrive(self):
        self.config['lammps_repeats'] = 2
        split.write_json(self.run / 'workflow_config.json', self.config)
        self.export()
        args = SimpleNamespace(scheduler='slurm', max_parallel=1, skip_conductance=False)
        with patch.object(split.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                patch('cluster_inputs.validate_templates'):
            self.assertEqual(split.run_imported_batch(args, ROOT, self.root), 0)
            self.assertEqual(run.call_count, 1)
            self.assertIn('run_multi.py', run.call_args.args[0][1])
            second = self.second_replica()
            # A source has no B frozen configuration. The new task freezes its own paths.
            (second['run_root'] / 'workflow_config.dpnegf.json').unlink()
            split.export_manifest(self.root, [second])
            run.reset_mock()
            self.assertEqual(split.run_imported_batch(args, ROOT, self.root), 0)
            self.assertEqual(run.call_count, 3)
            self.assertIn('collect_md_conductance.py', run.call_args.args[0][1])

    def test_collection_requires_consistent_counts_and_exact_replica_numbers(self):
        manifest = self.export()
        first = manifest['replicas'][0]
        entry = dict(manifest['replicas'][0])
        entry['replica'] = entry['replica'].replace('001', '002')
        second = self.second_replica()
        manifest['replicas'] = [entry]
        with self.assertRaisesRegex(ValueError, 'Unexpected replica numbers'):
            split.complete_configurations(self.root, manifest)
        split.write_json(second['run_root'] / 'workflow_config.json', dict(self.config, lammps_repeats=2))
        manifest['replicas'].append(first)
        with self.assertRaisesRegex(ValueError, 'Inconsistent expected replica counts'):
            split.complete_configurations(self.root, manifest)

    def test_transfer_uses_fixed_manifest_and_checks_copied_bytes(self):
        self.export()
        script = (ROOT / 'bscc_put_dpnegf.sh').read_text(encoding='utf-8')
        code = script.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        manifest_path = self.root / 'transfer_manifest.json'
        snapshot = manifest_path.read_bytes()
        copy = shutil.copy2
        for corrupt in (False, True):
            manifest_path.write_bytes(snapshot)
            staging = self.root / ('staging-bad' if corrupt else 'staging-good')
            staging.mkdir()
            def during_copy(source, destination):
                result = copy(source, destination)
                manifest_path.write_text('{"version": 999}')
                if corrupt:
                    Path(destination).write_text('changed during copy')
                return result
            with patch.object(sys, 'argv', ['-', str(self.root), str(staging)]), \
                    patch.object(shutil, 'copy2', side_effect=during_copy):
                if corrupt:
                    with self.assertRaisesRegex(SystemExit, 'SHA-256'):
                        exec(compile(code, 'transfer-staging', 'exec'), {})
                    self.assertFalse((staging / 'transfer_manifest.json').exists())
                else:
                    exec(compile(code, 'transfer-staging', 'exec'), {})
                    self.assertEqual((staging / 'transfer_manifest.json').read_bytes(), snapshot)
                    split.load_manifest(staging)

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
        self.config.update(lammps_device='gpu', r_max=6.5, l_def=1, structure_mode='assemble', lammps_repeats=2,
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
        frozen = (self.run / 'workflow_config.dpnegf.json').read_bytes()
        fingerprint = (self.leaf / 'dpnegf_input_fingerprint.json').read_bytes()
        # Deliver another replica into the same B root while preserving the old result.
        second = self.run.with_name('replica_002')
        second_leaf = second / 'dpnegf/40000'
        second_leaf.mkdir(parents=True)
        shutil.copy2(self.run / 'workflow_config.json', second / 'workflow_config.json')
        for name in ('STRUCT.fdf', '5_5.xyz'):
            shutil.copy2(self.leaf / name, second_leaf / name)
        (second / 'lammps').mkdir()
        (second / 'lammps/lammps_done.flag').touch()
        split.export_manifest(self.root, [{'run_root': second, 'configuration_root': second.parent}])
        shutil.rmtree(second / 'lammps')
        with patch.dict(os.environ, {'PYTHONPATH': os.pathsep.join(paths), 'PYTHONIOENCODING': 'utf-8'}):
            rc = split.run_imported_batch(args, software, self.root, 'cpu')
            self.assertEqual(rc, 0, (second / 'workflow.dpnegf.log').read_text(encoding='utf-8'))
        self.assertEqual((self.leaf / 'solver_count.txt').read_text(), '1')
        self.assertEqual((second_leaf / 'solver_count.txt').read_text(), '1')
        self.assertEqual((self.run / 'workflow_config.dpnegf.json').read_bytes(), frozen)
        self.assertEqual((self.leaf / 'dpnegf_input_fingerprint.json').read_bytes(), fingerprint)

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
