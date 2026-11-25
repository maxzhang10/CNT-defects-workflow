"""Run the CPU band-gap solver on a transferred scattering supercell."""
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.io import read

from copy_input_dpnegf import collect_input_files, copy_or_link_file
from geometry_mode import geometry_mode
from slurm_utils import sbatch_submit_with_retry, wait_for_jobs
from split_workflow import digest, inside, load_manifest, validate_structures


def prepare_inputs(leaf, xyz, templates):
    _, model_filename = collect_input_files(templates, files_to_copy=())
    model = templates / model_filename
    options = json.loads((templates / "input.json").read_text(encoding="utf-8"))
    atoms = read(xyz, format="extxyz")
    nel_atom = options["task_options"]["occupation_options"]["nel_atom"]
    if "H" in atoms.get_chemical_symbols():
        nel_atom["H"] = 1.0
    script = (templates / "run.py").read_text(encoding="utf-8")
    script, count = re.subn(r"^NEL_ATOM[ \t]*=.*$", lambda _: f"NEL_ATOM = {nel_atom!r}",
                            script, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one NEL_ATOM assignment in {templates / 'run.py'}")
    config = json.dumps(options, ensure_ascii=False, indent=2).encode("utf-8")
    launcher = (templates / "run.sh").read_text(encoding="utf-8")
    if launcher.count("__BAND_GAP_INPUTS__") != 1:
        raise ValueError(f"Expected one __BAND_GAP_INPUTS__ placeholder in {templates / 'run.sh'}")
    arguments = shlex.join(["--model", "./" + model.name, "--structure", "./" + xyz.name])
    launcher = launcher.replace("__BAND_GAP_INPUTS__", arguments)
    fingerprint = {xyz.name: digest(xyz), model.name: digest(model)}
    generated = {"input.json": config, "run.py": script.encode("utf-8"), "run.sh": launcher.encode("utf-8")}
    fingerprint.update({name: hashlib.sha256(data).hexdigest() for name, data in generated.items()})
    if leaf.exists():
        previous = json.loads((leaf / "band_gap_inputs.json").read_text(encoding="utf-8"))
        if previous != fingerprint or any(digest(leaf / name) != sha for name, sha in fingerprint.items()):
            raise ValueError(f"band_gap inputs changed: {leaf}")
        link = leaf / model.name
        if not link.is_symlink() or not link.samefile(model):
            raise ValueError(f"Expected a model symlink to {model}: {link}")
        return
    # Publish the complete input set together, including after an interrupted copy.
    with TemporaryDirectory(dir=leaf.parent) as tmp:
        staged = Path(tmp) / "inputs"
        staged.mkdir()
        shutil.copyfile(xyz, staged / xyz.name)
        copy_or_link_file(model, staged / model.name)
        for name, data in generated.items():
            (staged / name).write_bytes(data)
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
