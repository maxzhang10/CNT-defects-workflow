"""Portable structure handoff and independent DPNEGF batch execution."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside transfer root: {relative}')
    return path


def validate_structures(root, config):
    from fdf2xyz import parse_struct_fdf
    from ase.io import read
    import numpy as np
    leaves = sorted((root / 'dpnegf').rglob('STRUCT.fdf'))
    if len(leaves) != 1:
        raise ValueError(f'Expected one final structure per replica, found {len(leaves)}: {root}')
    fdf = leaves[0]
    xyz = fdf.with_name('%s_%s.xyz' % tuple(config['chirality']))
    a, b = parse_struct_fdf(fdf), read(xyz, format='extxyz')
    if not len(a) or len(a) != len(b) or not np.isfinite(b.positions).all():
        raise ValueError(f'Invalid converted structure: {fdf}')
    if not np.isfinite(a.positions).all() or not np.allclose(a.cell, b.cell):
        raise ValueError(f'Invalid structure cell: {fdf}')
    # XYZ intentionally swaps the first two lead PLs, so compare unordered atoms.
    def ordered(atoms):
        rows = np.column_stack((atoms.numbers, atoms.positions))
        return rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))]
    if not np.allclose(ordered(a), ordered(b), rtol=0, atol=1e-7):
        raise ValueError(f'FDF and XYZ describe different structures: {fdf}')
    if config.get('lammps_mode', 'md') == 'md' and int(config.get('md_steps', 40000)) > 0:
        if fdf.parent.name != str(config.get('md_steps', 40000)):
            raise ValueError(f'Missing expected final timestep: {fdf}')
    return [fdf, xyz]


def export_manifest(root, tasks):
    root = Path(root).resolve()
    files, entries = {}, []
    for task in tasks:
        run = Path(task['run_root']).resolve()
        config_path = run / 'workflow_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        required = [config_path, *validate_structures(run, config)]
        if config.get('lammps_mode', 'md') == 'opt' or int(config.get('md_steps', 40000)) > 0:
            if not (run / 'lammps/lammps_done.flag').is_file() or (run / 'lammps/lammps_failed.flag').exists():
                raise ValueError(f'LAMMPS not successfully completed: {run}')
        entry = {'replica': run.relative_to(root).as_posix(),
                 'configuration': Path(task['configuration_root']).resolve().relative_to(root).as_posix()}
        if config.get('uc_file'):
            source = Path(config['uc_file']).resolve()
            sha = digest(source)
            if config.get('uc_sha256') and config['uc_sha256'] != sha:
                raise ValueError(f'UC hash mismatch: {source}')
            asset = root / '_transfer_assets' / sha / source.name
            asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, asset)
            entry['uc_file'] = asset.relative_to(root).as_posix()
            required.append(asset)
        for path in required:
            files[path.relative_to(root).as_posix()] = digest(path)
        entries.append(entry)
    write_json(root / 'transfer_manifest.json', {'version': 1, 'replicas': entries, 'files': files})
    print(f'[OK] Transfer ready: {len(entries)} replicas; {root / "transfer_manifest.json"}')


def load_manifest(root):
    manifest = json.loads((root / 'transfer_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or not manifest.get('replicas'):
        raise ValueError('Invalid or empty transfer manifest')
    for name, sha in manifest['files'].items():
        if digest(inside(root, name)) != sha:
            raise ValueError(f'Transfer checksum mismatch: {name}')
    seen = set()
    configurations = {}
    for entry in manifest['replicas']:
        run = inside(root, entry['replica'])
        if run in seen:
            raise ValueError(f'Duplicate replica: {run}')
        seen.add(run)
        configuration = inside(root, entry['configuration'])
        if run.parent != configuration:
            raise ValueError(f'Replica is outside its declared configuration: {run}')
        configurations.setdefault(configuration, set()).add(run)
        config_path = run / 'workflow_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        required = [config_path, *validate_structures(run, config)]
        if config.get('uc_file'):
            required.append(inside(root, entry['uc_file']))
        if any(p.relative_to(root).as_posix() not in manifest['files'] for p in required):
            raise ValueError(f'Manifest omits required inputs: {run}')
    for configuration, expected in configurations.items():
        actual = {path.resolve() for path in configuration.iterdir()
                  if path.is_dir() and re.fullmatch(r'replica_\d+', path.name)}
        if actual != expected:
            raise ValueError(f'Replica directories differ from transfer manifest: {configuration}; '
                             f'extra={sorted(map(str, actual - expected))}; missing={sorted(map(str, expected - actual))}')
    return manifest


def freeze_config(root, entry, lammps_device=None):
    run = inside(root, entry['replica'])
    target = run / 'workflow_config.dpnegf.json'
    config = json.loads((run / 'workflow_config.json').read_text(encoding='utf-8'))
    config.update(data_root=str(root), structure_root=str(run),
                  configuration_root=str(inside(root, entry['configuration'])))
    if lammps_device is not None:
        from cluster_inputs import input_directory
        input_directory(lammps_device, 'dpnegf')  # Validate B's cluster selection.
        config['lammps_device'] = lammps_device
    if entry.get('uc_file'):
        config['uc_file'] = str(inside(root, entry['uc_file']))
    if target.exists():
        saved = json.loads(target.read_text(encoding='utf-8'))
        if saved != config:
            raise ValueError(f'Frozen B configuration changed; use a new calculation directory: {target}')
    else:
        if list(run.rglob('dpnegf_done.flag')) or list(run.rglob('dpnegf_submitted.flag')):
            raise ValueError(f'Existing DPNEGF state without frozen B configuration: {run}')
        write_json(target, config)
    return target


def lock_inputs(leaf):
    """Bind retries to the actual structures, model, config and solver inputs."""
    names = ['STRUCT.fdf', 'input.json', 'run.py', 'expected_negf_config_hash.txt']
    names += [p.name for p in sorted(leaf.glob('*.xyz'))]
    names += [p.name for p in sorted(leaf.glob('*.pth'))]
    fingerprint = {name: digest(leaf / name) for name in names}
    lock = leaf / 'dpnegf_input_fingerprint.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8')) != fingerprint:
            raise ValueError(f'DPNEGF input contents changed; use a new calculation directory: {leaf}')
    else:
        if any((leaf / name).exists() for name in ('dpnegf_done.flag', 'dpnegf_submitted.flag', 'dpnegf_started.flag')):
            raise ValueError(f'Existing result/job without input fingerprint: {leaf}')
        write_json(lock, fingerprint)


def run_imported_batch(args, workflow_dir, root, lammps_device=None):
    root = Path(root).resolve()
    manifest = load_manifest(root)  # Check the entire batch before submitting anything.
    from cluster_inputs import validate_templates
    device = lammps_device
    if device is None:
        source = inside(root, manifest['replicas'][0]['replica']) / 'workflow_config.json'
        device = json.loads(source.read_text(encoding='utf-8')).get('lammps_device', 'gpu')
    validate_templates(device, 'dpnegf', workflow_dir)
    configs = [freeze_config(root, entry, lammps_device) for entry in manifest['replicas']]
    def run_one(config):
        with (config.parent / 'workflow.dpnegf.log').open('a', encoding='utf-8') as log:
            return subprocess.run([sys.executable, str(workflow_dir / 'run_multi.py'),
                '--phase', 'dpnegf', '--root', str(config.parent), '--config', str(config),
                '--scheduler', args.scheduler, '--skip-conductance'],
                stdout=log, stderr=subprocess.STDOUT).returncode
    results = []
    print(f'[START] DPNEGF: {len(configs)} replicas, root={root}', flush=True)
    with ThreadPoolExecutor(max_workers=args.max_parallel or 16) as pool:
        pending = {pool.submit(run_one, config): config for config in configs}
        for future in as_completed(pending):
            config = pending[future]
            try:
                rc = future.result()
            except Exception as exc:
                print(f'[FAIL] {config.parent}: {exc}', flush=True)
                rc = 1
            results.append(rc)
            message = f'[{"OK" if rc == 0 else "FAIL"}] {len(results)}/{len(configs)} {config.parent} exit={rc}'
            print(message, flush=True)
            with (root / 'record.dpnegf.log').open('a', encoding='utf-8') as log:
                log.write(message + '\n')
    if any(results):
        return 1
    if not args.skip_conductance:
        cmd = [sys.executable, str(workflow_dir / 'stage/collect_md_conductance.py'), '--root', str(root)]
        for name in sorted({entry['configuration'] for entry in manifest['replicas']}):
            cmd += ['--configuration-dir', str(inside(root, name))]
        return subprocess.run(cmd).returncode
    return 0
