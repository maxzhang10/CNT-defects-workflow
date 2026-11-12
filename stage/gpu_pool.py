"""One-GPU Slurm array workers sharing a durable replica queue.

The state file lives on the same shared filesystem as the batch output. All
read/modify/write transactions are protected by POSIX flock; this mode requires
a filesystem whose flock is visible across compute nodes.
"""

import contextlib
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def _lock_file(file):
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(file, fcntl.LOCK_EX)


def _unlock_file(file):
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file, fcntl.LOCK_UN)


def _paths(root):
    directory = Path(root) / ".cnt_gpu_pool"
    return directory, directory / "state.json", directory / "state.lock"


@contextlib.contextmanager
def _locked(root):
    directory, state_path, lock_path = _paths(root)
    directory.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        _lock_file(lock)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            yield state
            if state is not None:
                fd, name = tempfile.mkstemp(prefix="state-", suffix=".json", dir=directory)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as output:
                        json.dump(state, output, indent=2, ensure_ascii=False)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(name, state_path)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
        finally:
            _unlock_file(lock)


def _config_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_active(job_id):
    if not job_id:
        return False
    result = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%i"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def _published_tasks(root, tasks, batch):
    """Verify immutable handoff entries before skipping already published replicas."""
    if not (root / "transfer_manifest.json").exists():
        return set()
    import batch_generate as bg
    from split_workflow import load_manifest
    manifest = load_manifest(root, check_replica_directories=False)
    by_root = {Path(task["run_root"]).resolve(): task for task in tasks}
    published = set()
    for entry in manifest["replicas"]:
        run = (root / entry["replica"]).resolve()
        if run not in by_root:
            raise ValueError(f"Published replica absent from batch: {run}")
        saved = json.loads((run / "workflow_config.json").read_text(encoding="utf-8"))
        if saved != bg.make_task_config(by_root[run], batch):
            raise ValueError(f"Published LAMMPS configuration changed: {run}")
        published.add(by_root[run]["task_id"])
    return published


def submit(args, batch, tasks, workflow_dir, config_path):
    if args.phase != "lammps":
        raise ValueError("GPU pool only supports phase=lammps")
    root = Path(batch["data_root"]).resolve()
    config_path = Path(config_path).resolve()
    directory, _, _ = _paths(root)
    directory.mkdir(parents=True, exist_ok=True)
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in GPU pool")
    initial = {
        "phase": "lammps",
        "config_hash": _config_hash(config_path), "task_ids": ids,
        "tasks": {task_id: {"status": "pending", "owner": None} for task_id in ids},
        "array_job_id": None, "generation": uuid.uuid4().hex,
        "finalized": False,
    }
    _initialize(root, initial)
    with _locked(root) as state:
        if state.get("phase") != "lammps":
            raise ValueError("Existing pool state is not LAMMPS-only; use a new data_root")
        if state.get("submitting"):
            raise RuntimeError("Another GPU pool submission is in progress")
        if state is not None:
            if state["config_hash"] != _config_hash(config_path) or state["task_ids"] != ids:
                raise ValueError("GPU pool configuration changed; use a new data_root")
            if _array_active(state.get("array_job_id")):
                raise RuntimeError(f"GPU pool already active: {state['array_job_id']}")
            published = _published_tasks(root, tasks, batch)
            for task_id, entry in state["tasks"].items():
                if task_id in published:
                    entry.update(status="done", owner=None)
                elif entry["status"] == "done":
                    raise ValueError(f"Completed replica missing from transfer manifest: {task_id}")
            if not args.pool_retry_failed and any(
                entry["status"] == "failed" for entry in state["tasks"].values()
            ):
                raise RuntimeError("GPU pool has failed tasks; repair them and use --pool-retry-failed")
            for entry in state["tasks"].values():
                if entry["status"] == "running":
                    entry.update(status="pending", owner=None)
            if args.pool_retry_failed:
                for task in tasks:
                    entry = state["tasks"][task["task_id"]]
                    if entry["status"] != "failed":
                        continue
                    flags = list((Path(task["run_root"]) / "lammps").glob("*_failed.flag"))
                    if flags:
                        raise RuntimeError(
                            f"Repair failed leaf and clear its failure flags before retry: {flags[0]}")
                    entry.update(status="pending", owner=None)
            state["finalized"] = False
            state.pop("summary", None)
            state["array_job_id"] = None
            state["generation"] = uuid.uuid4().hex
        state["submitting"] = True
    with _locked(root) as state:
        if state["config_hash"] != _config_hash(config_path):
            raise ValueError("GPU pool configuration changed during submission")
        generation = state["generation"]
    try:
        script = directory / "worker.sh"
        setup = f"source {shlex.quote(str(Path(args.pool_setup).resolve()))}\n" if args.pool_setup else ""
        script.write_text(
            "#!/bin/bash\nset -Eeuo pipefail\n"
            + setup
            + 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"\n'
            + 'export MKL_NUM_THREADS="$OMP_NUM_THREADS"\n'
            + 'export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"\n'
            + 'export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"\n'
            + f"cd {shlex.quote(str(workflow_dir))}\n"
            + f"exec {shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
              f"worker --config {shlex.quote(str(config_path))} "
              f"--generation {shlex.quote(generation)} "
              f"--workflow-dir {shlex.quote(str(workflow_dir))}\n",
            encoding="utf-8",
        )
        cmd = ["sbatch", "--parsable", f"--array=0-{args.para - 1}%{args.para}",
               "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.pool_cpus}",
               "--gpus=1", f"--time={args.worker_time}",
               f"--output={directory / 'worker-%A_%a.out'}",
               f"--error={directory / 'worker-%A_%a.err'}"]
        if args.pool_partition:
            cmd.append(f"--partition={args.pool_partition}")
        if args.pool_mem:
            cmd.append(f"--mem={args.pool_mem}")
        cmd.append(str(script))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Invalid sbatch job ID: {result.stdout!r}")
    except Exception:
        with _locked(root) as state:
            state["submitting"] = False
        raise
    with _locked(root) as state:
        state["array_job_id"] = job_id
        state["submitting"] = False
    print(f"GPU pool submitted: array={job_id}, target GPUs={args.para}, tasks={len(ids)}")
    print(f"State: {_paths(root)[1]}")
    return 0


def _initialize(root, initial):
    directory, state_path, lock_path = _paths(root)
    directory.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        _lock_file(lock)
        try:
            if not state_path.exists():
                state_path.write_text(json.dumps(initial, indent=2, ensure_ascii=False), encoding="utf-8")
        finally:
            _unlock_file(lock)


def _claim(root, generation, owner):
    with _locked(root) as state:
        if state["generation"] != generation:
            return None
        for task_id in state["task_ids"]:
            entry = state["tasks"][task_id]
            if entry["status"] == "pending":
                entry.update(status="running", owner=owner, started=time.time())
                return task_id
    return None


def _finish(root, generation, task_id, owner, result):
    with _locked(root) as state:
        if state["generation"] != generation:
            raise RuntimeError("GPU pool generation changed during calculation")
        entry = state["tasks"][task_id]
        if entry["status"] != "running" or entry["owner"] != owner:
            raise RuntimeError(f"Lost claim for {task_id}")
        entry.update(status="done" if result == 0 else "failed", exit_code=result,
                     ended=time.time())


def _validate_result(task):
    from split_workflow import validate_structures
    run = Path(task["run_root"])
    config_path = run / "workflow_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing LAMMPS workflow configuration: {run}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("lammps_mode", "md") == "opt" or int(config.get("md_steps", 40000)) > 0:
        if not (run / "lammps/lammps_done.flag").is_file() or (run / "lammps/lammps_failed.flag").exists():
            raise RuntimeError(f"LAMMPS not successfully completed: {run}")
    validate_structures(run, config)


def _publish_result(root, generation, task):
    from split_workflow import export_manifest
    # Serialize cumulative manifest updates across all workers. Publishing precedes
    # marking DONE; restart reconciles an entry published just before interruption.
    with _locked(root) as state:
        if state["generation"] != generation:
            raise RuntimeError("GPU pool generation changed before handoff")
        export_manifest(root, [task])


def _finalize(root, generation, tasks, workflow_dir):
    with _locked(root) as state:
        if state["generation"] != generation or state["finalized"]:
            return
        statuses = [entry["status"] for entry in state["tasks"].values()]
        if "pending" in statuses or "running" in statuses:
            return
        state["finalized"] = True
        failed = statuses.count("failed")
        array_job_id = state.get("array_job_id")
    with _locked(root) as state:
        state["summary"] = {"phase": "lammps", "failed": failed,
                            "completed_at": time.time()}
    if array_job_id:
        # Cancel only pending siblings. A running worker is allowed to exit itself.
        subprocess.run(["scancel", "--state=PENDING", str(array_job_id)], check=False)
    print(f"LAMMPS GPU pool finished: failed={failed}", flush=True)
    return 1 if failed else 0


def worker(config_path, generation, workflow_dir):
    import batch_generate as bg
    if not os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError("GPU pool worker must run inside its Slurm job array")
    config_path = Path(config_path).resolve()
    workflow_dir = Path(workflow_dir).resolve()
    batch = bg.bc.load(config_path)
    root = Path(batch["data_root"]).resolve()
    tasks = bg.build_tasks(batch, root)
    by_id = {task["task_id"]: task for task in tasks}
    with _locked(root) as state:
        if state.get("phase") != "lammps":
            raise ValueError("GPU pool worker requires LAMMPS-only state")
        if state["config_hash"] != _config_hash(config_path) or state["task_ids"] != list(by_id):
            raise ValueError("GPU pool task list or configuration changed")
    owner = f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', os.getpid())}"
    print(f"GPU pool worker {owner} on {platform.node()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    with _locked(root) as state:
        if state["generation"] == generation and not state.get("array_job_id"):
            state["array_job_id"] = os.environ.get("SLURM_ARRAY_JOB_ID")
    while True:
        end_time = os.environ.get("SLURM_JOB_END_TIME")
        if end_time and int(end_time) - time.time() < 300:
            print(f"[{owner}] less than five minutes remain; stop claiming new tasks", flush=True)
            break
        task_id = _claim(root, generation, owner)
        if task_id is None:
            break
        print(f"[{owner}] claimed {task_id}", flush=True)
        try:
            result = bg.run_single_task(by_id[task_id], batch, "local", workflow_dir, phase="lammps")
            rc = result[1]
            if rc == 0:
                _validate_result(by_id[task_id])
                _publish_result(root, generation, by_id[task_id])
        except Exception as exc:
            print(f"[{owner}] {task_id}: {exc}", file=sys.stderr, flush=True)
            rc = 1
        _finish(root, generation, task_id, owner, rc)
    return _finalize(root, generation, tasks, workflow_dir) or 0


def wait(root, poll_interval=30):
    """Monitor LAMMPS calculation and structure handoff completion."""
    previous = None
    while True:
        with _locked(root) as state:
            if state is None or not state.get("array_job_id"):
                raise RuntimeError("No submitted GPU pool to wait for")
            job_id = state["array_job_id"]
            counts = {name: sum(e["status"] == name for e in state["tasks"].values())
                      for name in ("pending", "running", "done", "failed")}
            summary = state.get("summary")
        if counts != previous:
            print(f"GPU pool {job_id}: {counts}", flush=True)
            previous = counts
        if summary is not None:
            return 1 if summary["failed"] else 0
        if not _array_active(job_id):
            # Read again: the last worker may have finalized since our snapshot.
            with _locked(root) as state:
                summary = state.get("summary")
            if summary is not None:
                return 1 if summary["failed"] else 0
            print("GPU pool workers exited before completion; inspect logs and resubmit.", flush=True)
            return 1
        time.sleep(poll_interval)


def status(root):
    with _locked(root) as state:
        if state is None:
            raise FileNotFoundError("No GPU pool state in data_root")
        counts = {name: sum(e["status"] == name for e in state["tasks"].values())
                  for name in ("pending", "running", "done", "failed")}
        print(json.dumps({"array_job_id": state.get("array_job_id"),
                          "counts": counts,
                          "failed_tasks": [task_id for task_id, entry in state["tasks"].items()
                                           if entry["status"] == "failed"],
                          "summary": state.get("summary")}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("worker", "status"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation")
    parser.add_argument("--workflow-dir")
    options = parser.parse_args()
    if options.action == "worker":
        sys.exit(worker(options.config, options.generation, options.workflow_dir))
    import batch_config as bc
    status(bc.load(options.config)["data_root"])
