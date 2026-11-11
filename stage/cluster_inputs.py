"""Resolve solver templates using the configured cluster device."""
from pathlib import Path


def input_directory(device, solver, project_root=None):
    if device not in ('cpu', 'gpu'):
        raise ValueError(f'lammps_device 必须为 cpu / gpu，当前值: {device!r}')
    if solver not in ('lammps', 'dpnegf'):
        raise ValueError(f'Unknown solver: {solver}')
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return root / 'input_files' / f'{device}_cluster' / solver


def validate_templates(device, solver, project_root=None, mode='md'):
    """Fail before generating structures or freezing B config if assets are missing."""
    directory = input_directory(device, solver, project_root)
    if solver == 'lammps':
        required = ['opt.lammps' if mode == 'opt' else 'in.lammps', 'run.sh']
        pattern = 'CH.airebo-m' if device == 'cpu' else '*.pth'
    else:
        required = ['input.json', 'run.py', 'run.sh']
        pattern = '*.pth'
    missing = [str(directory / name) for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing cluster templates:\n' + '\n'.join(missing))
    models = [path for path in directory.glob(pattern) if path.is_file()]
    if len(models) != 1:
        raise ValueError(f'Expected exactly one {pattern} model/potential in {directory}; found {len(models)}')
    return directory
