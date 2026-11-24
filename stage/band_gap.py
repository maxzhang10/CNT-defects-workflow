"""Run the CPU band-gap solver on a transferred scattering supercell."""
import hashlib
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.io import read

from geometry_mode import geometry_mode
from slurm_utils import sbatch_submit_with_retry, wait_for_jobs
from split_workflow import digest, inside, load_manifest, validate_structures


def prepare_inputs(leaf, xyz, templates):
    model, = templates.glob("*.pth")
    options = json.loads((templates / "input.json").read_text(encoding="utf-8"))
    atoms = read(xyz, format="extxyz")
    if "H" in atoms.get_chemical_symbols():
        options["task_options"]["occupation_options"]["nel_atom"]["H"] = 1.0
    config = json.dumps(options, ensure_ascii=False, indent=2).encode("utf-8")
    files = {"structure.xyz": xyz, "model.pth": model, "run.sh": templates / "run.sh"}
    fingerprint = {name: digest(src) for name, src in files.items()}
    fingerprint["input.json"] = hashlib.sha256(config).hexdigest()
    if leaf.exists():
        previous = json.loads((leaf / "band_gap_inputs.json").read_text(encoding="utf-8"))
        if previous != fingerprint or any(digest(leaf / name) != sha for name, sha in fingerprint.items()):
            raise ValueError(f"band_gap inputs changed: {leaf}")
        return
    # Publish the complete input set together, including after an interrupted copy.
    with TemporaryDirectory(dir=leaf.parent) as tmp:
        staged = Path(tmp) / "inputs"
        staged.mkdir()
        for name, src in files.items():
            shutil.copyfile(src, staged / name)
        (staged / "input.json").write_bytes(config)
        (staged / "band_gap_inputs.json").write_text(json.dumps(fingerprint), encoding="utf-8")
        staged.rename(leaf)


def run_replica(run, templates, scheduler, poll_interval=30):
    config = json.loads((run / "workflow_config.json").read_text(encoding="utf-8"))
    geometry_mode(config, "band_gap")
    _, xyz = validate_structures(run, config)
    leaf = run / "band_gap" / ("final" if xyz.parent == run / "dpnegf" else xyz.parent.name)
    leaf.parent.mkdir(exist_ok=True)
    with (leaf.parent / ".scheduler.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        prepare_inputs(leaf, xyz, templates)
        done = leaf / "band_gap_done.flag"
        failed = leaf / "band_gap_failed.flag"
        job = leaf / "job_id.txt"
        if not done.exists() and not failed.exists():
            if job.exists() or scheduler == "slurm":
                if job.exists():
                    job_id = job.read_text().strip()
                    if not job_id.isdecimal():
                        raise ValueError(f"Invalid Slurm job ID: {job}")
                else:
                    job_id = sbatch_submit_with_retry(leaf, label="band_gap")
                    job.write_text(job_id)
                wait_for_jobs([job_id], poll_interval=poll_interval, label="band_gap")
            else:
                with (leaf / "band_gap.log").open("a") as log:
                    subprocess.run(["bash", "run.sh"], cwd=leaf, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
        if failed.exists() or not done.exists():
            raise RuntimeError(f"band_gap did not complete: {leaf}")
        result_file = leaf / "output/results/gap.json"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        gap = result["band_gap_eV"]
        if type(gap) not in (int, float) or not math.isfinite(gap):
            raise ValueError(f"Invalid band_gap_eV: {result_file}")
    return leaf


def run_batch(root, templates, scheduler, max_parallel):
    manifest = load_manifest(root)
    runs = []
    for entry in manifest["replicas"]:
        geometry_mode(entry, "band_gap")
        runs.append(inside(root, entry["replica"]))
    failed = False
    with ThreadPoolExecutor(max_workers=max_parallel) as workers:
        tasks = {workers.submit(run_replica, run, templates, scheduler): run for run in runs}
        for task in as_completed(tasks):
            try:
                print(f"[OK] band_gap: {task.result()}", flush=True)
            except Exception as exc:
                print(f"[FAIL] band_gap: {tasks[task]}: {exc}", flush=True)
                failed = True
    return int(failed)
