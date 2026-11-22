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
    """Publish completed replicas cumulatively; existing entries are immutable.

    Callers must serialize updates: the batch coordinator calls serially;
    GPU pool workers hold the shared pool lock across this call.
    """
    root = Path(root).resolve()
    manifest_path = root / 'transfer_manifest.json'
    manifest = (json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest_path.exists() else {'version': 1, 'replicas': [], 'files': {}})
    if manifest.get('version') != 1:
        raise ValueError('Unsupported transfer manifest version')
    files = dict(manifest['files'])
    entries = {entry['replica']: entry for entry in manifest['replicas']}
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
            if asset.exists():
                if digest(asset) != sha:
                    raise ValueError(f'Published UC contents changed: {asset}')
            else:
                shutil.copy2(source, asset)
            entry['uc_file'] = asset.relative_to(root).as_posix()
            required.append(asset)
        for path in required:
            name, sha = path.relative_to(root).as_posix(), digest(path)
            if name in files and files[name] != sha:
                raise ValueError(f'Published input contents changed: {path}')
            files[name] = sha
        previous = entries.get(entry['replica'])
        if previous is not None and previous != entry:
            raise ValueError(f'Published replica changed: {run}')
        entries[entry['replica']] = entry
    write_json(manifest_path, {'version': 1,
                              'replicas': [entries[name] for name in sorted(entries)], 'files': files})
    print(f'[OK] Transfer ready: {len(entries)} replicas; {root / "transfer_manifest.json"}', flush=True)


def load_manifest(root, *, check_replica_directories=True, manifest=None, imported=False):
    # A supplied snapshot can be validated before an atomic recovery write.
    if manifest is None:
        manifest = json.loads((root / 'transfer_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or not manifest.get('replicas'):
        raise ValueError('Invalid or empty transfer manifest')
    for name, sha in manifest['files'].items():
        path = inside(root, name)
        if not imported and digest(path) != sha:
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
        if imported and imported_terminal_leaf(run) is not None:
            continue
        config_path = run / 'workflow_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        required = [config_path, *validate_structures(run, config)]
        if config.get('uc_file'):
            required.append(inside(root, entry['uc_file']))
        if any(p.relative_to(root).as_posix() not in manifest['files'] for p in required):
            raise ValueError(f'Manifest omits required inputs: {run}')
    for configuration, expected in configurations.items() if check_replica_directories else []:
        actual = {path.resolve() for path in configuration.iterdir()
                  if path.is_dir() and re.fullmatch(r'replica_\d+', path.name)}
        if actual != expected:
            raise ValueError(f'Replica directories differ from transfer manifest: {configuration}; '
                             f'extra={sorted(map(str, actual - expected))}; missing={sorted(map(str, expected - actual))}')
    return manifest


def complete_configurations(root, manifest):
    """Select only configurations with every planned replica delivered."""
    groups = {}
    for entry in manifest['replicas']:
        groups.setdefault(entry['configuration'], []).append(entry)
    complete = []
    for name, entries in sorted(groups.items()):
        counts = set()
        for entry in entries:
            config = json.loads((inside(root, entry['replica']) / 'workflow_config.json').read_text(encoding='utf-8'))
            expected = config.get('lammps_repeats')
            if type(expected) is not int or expected < 1:
                raise ValueError(f'Invalid expected replica count: {entry["replica"]}')
            counts.add(expected)
        if len(counts) != 1:
            raise ValueError(f'Inconsistent expected replica counts: {name}')
        expected = counts.pop()
        planned = {f'replica_{i:03d}' for i in range(1, expected + 1)}
        received = {Path(entry['replica']).name for entry in entries}
        if not received <= planned:
            raise ValueError(f'Unexpected replica numbers: {name}')
        if received == planned:
            complete.append(name)
        else:
            print(f'[WAIT] Conductance deferred: {name}; received {len(received)}/{expected} replicas', flush=True)
    return complete


def imported_terminal_leaf(run):
    """A split-phase replica has one final leaf. Locate its terminal marker."""
    directory = Path(run) / 'dpnegf'
    leaves = {flag.parent for name in ('dpnegf_failed.flag', 'dpnegf_submit_failed.flag', 'dpnegf_done.flag')
              for flag in directory.rglob(name)}
    if len(leaves) > 1:
        raise ValueError(f'Multiple terminal leaves in imported replica: {run}')
    return next(iter(leaves), None)


def imported_status(run):
    from negf_provenance import result_is_readable
    leaf = imported_terminal_leaf(run)
    if leaf is None:
        return 'pending'
    if any((leaf / name).is_file() for name in ('dpnegf_failed.flag', 'dpnegf_submit_failed.flag')):
        return 'failed'
    if result_is_readable(leaf, require_torch=True):
        return 'done'
    print(f'[FAIL] Done flag with unreadable DPNEGF result: {leaf}', flush=True)
    return 'failed'


def prepare_imported_config(root, entry, lammps_device=None, overrides=None):
    run = inside(root, entry['replica'])
    target = run / 'workflow_config.dpnegf.json'
    if target.is_file() and any(flag.with_name('job_id.txt').is_file()
                                for flag in (run / 'dpnegf').rglob('dpnegf_submitted.flag')):
        # A resumed Slurm job still uses the inputs prepared at submission.
        return target
    config = json.loads((run / 'workflow_config.json').read_text(encoding='utf-8'))
    config.update(data_root=str(root), structure_root=str(run),
                  configuration_root=str(inside(root, entry['configuration'])))
    if lammps_device is not None:
        from cluster_inputs import input_directory
        input_directory(lammps_device, 'dpnegf')  # Validate B's cluster selection.
        config['lammps_device'] = lammps_device
    if entry.get('uc_file'):
        config['uc_file'] = str(inside(root, entry['uc_file']))
    overrides = overrides or {}
    if overrides.get('by_chirality'):
        m, n = config['chirality']
        overrides = {**overrides, **overrides['by_chirality'].get(f'{m}_{n}', {})}
    if 'e_fermi' in overrides:
        config['e_fermi'] = overrides['e_fermi']
    if 'self_energy_cache' in overrides:
        from batch_config import resolve_self_energy_cache
        m, n = config['chirality']
        uc_sha = config.get('uc_sha256')
        if config.get('structure_mode') == 'assemble':
            perfect = config['assembly'].get('modules', {}).get('perfect', {})
            uc_sha = perfect.get('sha256', uc_sha)
        config['self_energy_cache'] = resolve_self_energy_cache(
            overrides['self_energy_cache'], m=m, n=n, data_root=root,
            uc_sha256=uc_sha, ele_T=config.get('ele_T'))
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


def run_imported_batch(args, workflow_dir, root, lammps_device=None, overrides=None):
    root = Path(root).resolve()
    manifest = load_manifest(root, imported=True)
    pending_entries = []
    prior_failures = 0
    for entry in manifest['replicas']:
        run = inside(root, entry['replica'])
        status = imported_status(run)
        if status == 'pending':
            pending_entries.append(entry)
        else:
            print(f'[SKIP] {status}: {run}', flush=True)
            prior_failures += status == 'failed'
    if not pending_entries:
        return int(bool(prior_failures))
    from cluster_inputs import validate_templates
    device = lammps_device
    if device is None:
        source = inside(root, pending_entries[0]['replica']) / 'workflow_config.json'
        device = json.loads(source.read_text(encoding='utf-8')).get('lammps_device', 'gpu')
    validate_templates(device, 'dpnegf', workflow_dir)
    configs = [prepare_imported_config(root, entry, lammps_device, overrides) for entry in pending_entries]
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
    return int(bool(prior_failures or any(results)))
