"""Resource-pool queue and Slurm submission regression checks."""

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage"))
import gpu_pool as gp  # noqa: E402


class GpuPoolTests(unittest.TestCase):
 def test_claims_are_unique_across_workers(self):
    with tempfile.TemporaryDirectory() as directory:
        self._claims_are_unique_across_workers(Path(directory))

 def _claims_are_unique_across_workers(self, tmp_path):
    state = {
        "generation": "g", "task_ids": [str(i) for i in range(40)],
        "tasks": {str(i): {"status": "pending", "owner": None} for i in range(40)},
        "array_job_id": None, "finalized": False, "skip_conductance": True,
    }
    gp._initialize(tmp_path, state)

    def drain(owner):
        claimed = []
        while (task_id := gp._claim(tmp_path, "g", owner)) is not None:
            claimed.append(task_id)
            gp._finish(tmp_path, "g", task_id, owner, 0)
        return claimed

    with ThreadPoolExecutor(max_workers=4) as executor:
        groups = list(executor.map(drain, ["a", "b", "c", "d"]))
    claimed = [task_id for group in groups for task_id in group]
    assert len(claimed) == len(set(claimed)) == 40
    assert all(entry["status"] == "done" for entry in
               json.loads((tmp_path / ".cnt_gpu_pool" / "state.json").read_text())["tasks"].values())


 def test_submit_creates_one_array_of_single_gpu_jobs(self):
    with tempfile.TemporaryDirectory() as directory:
        self._submit_creates_one_array_of_single_gpu_jobs(Path(directory))

 def _submit_creates_one_array_of_single_gpu_jobs(self, tmp_path):
    config = tmp_path / "batch.json"
    config.write_text("{}", encoding="utf-8")
    tasks = [{"task_id": "a", "run_root": tmp_path / "a"},
             {"task_id": "b", "run_root": tmp_path / "b"}]
    args = SimpleNamespace(phase="lammps", para=3, pool_cpus=6, worker_time="08:00:00",
                           pool_partition=None, pool_mem="32G",
                           pool_setup=None, skip_conductance=True,
                           pool_retry_failed=False)

    def fake_run(cmd, **_):
        assert cmd[0] == "sbatch"
        assert "--array=0-2%3" in cmd
        assert "--gpus=1" in cmd
        assert "--cpus-per-task=6" in cmd
        partitions = [option for option in cmd if option.startswith("--partition=")]
        assert partitions == ([f"--partition={args.pool_partition}"] if args.pool_partition else [])
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    with patch.object(gp.subprocess, "run", side_effect=fake_run):
        gp.submit(args, {"data_root": tmp_path}, tasks, ROOT, config)
    state = json.loads((tmp_path / ".cnt_gpu_pool" / "state.json").read_text())
    assert state["array_job_id"] == "12345"
    assert len(state["tasks"]) == 2
    args.pool_partition = "gpu_real"
    with patch.object(gp, "_array_active", return_value=False), \
         patch.object(gp.subprocess, "run", side_effect=fake_run):
        gp.submit(args, {"data_root": tmp_path}, tasks, ROOT, config)

 def test_failed_tasks_require_explicit_retry(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "batch.json"
        config.write_text("{}", encoding="utf-8")
        task = {"task_id": "a", "run_root": root / "a"}
        initial = {"phase": "lammps", "config_hash": gp._config_hash(config), "task_ids": ["a"],
                   "tasks": {"a": {"status": "failed", "owner": "old"}},
                   "array_job_id": "123", "generation": "old",
                   "finalized": True, "skip_conductance": True}
        gp._initialize(root, initial)
        args = SimpleNamespace(phase="lammps", para=1, pool_cpus=6, worker_time="01:00:00",
                               pool_partition="gpu_4090", pool_mem=None,
                               pool_setup=None, skip_conductance=True,
                               pool_retry_failed=False)
        with patch.object(gp, "_array_active", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "--pool-retry-failed"):
                gp.submit(args, {"data_root": root}, [task], ROOT, config)

 def test_only_one_worker_finalizes(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gp._initialize(root, {
            "generation": "g", "task_ids": ["a"],
            "tasks": {"a": {"status": "done", "owner": "w"}},
            "array_job_id": "123", "finalized": False,
            "skip_conductance": False,
        })
        task = {"task_id": "a", "configuration_root": root,
                "temperature": 100, "conductance_mode": "default"}
        with patch.object(gp.subprocess, "run") as cancel:
            self.assertEqual(gp._finalize(root, "g", [task], ROOT), 0)
            self.assertIsNone(gp._finalize(root, "g", [task], ROOT))
        cancel.assert_called_once_with(["scancel", "--state=PENDING", "123"],
                                       check=False)

 def test_worker_drains_tasks_and_marks_completion(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "batch.json"
        config.write_text("{}", encoding="utf-8")
        tasks = [{"task_id": str(i), "run_root": root / str(i),
                  "configuration_root": root, "temperature": 100,
                  "conductance_mode": "default"} for i in range(3)]
        gp._initialize(root, {
            "phase": "lammps",
            "config_hash": gp._config_hash(config),
            "generation": "g", "task_ids": ["0", "1", "2"],
            "tasks": {str(i): {"status": "pending", "owner": None} for i in range(3)},
            "array_job_id": None, "finalized": False, "skip_conductance": True,
        })
        phases = []
        def run_task(task, *_, phase):
            phases.append(phase)
            return task["task_id"], 0, "OK"
        fake_batch = SimpleNamespace(
            bc=SimpleNamespace(load=lambda _: {"data_root": root}),
            build_tasks=lambda *_: tasks,
            run_single_task=run_task,
        )
        with patch.dict(sys.modules, {"batch_generate": fake_batch}), \
             patch.dict(gp.os.environ, {"SLURM_ARRAY_JOB_ID": "123",
                                        "SLURM_ARRAY_TASK_ID": "0"}), \
             patch.object(gp.platform, "node", return_value="test-node"), \
             patch.object(gp, "_validate_result"), \
             patch.object(gp, "_publish_result") as publish, \
             patch.object(gp.subprocess, "run"):
            self.assertEqual(gp.worker(config, "g", ROOT), 0)
        self.assertEqual(phases, ["lammps"] * 3)
        self.assertEqual(publish.call_count, 3)
        state = json.loads((root / ".cnt_gpu_pool" / "state.json").read_text())
        self.assertTrue(state["finalized"])
        self.assertTrue(all(entry["status"] == "done" for entry in state["tasks"].values()))

 def test_missing_result_is_not_marked_successful(self):
    with tempfile.TemporaryDirectory() as directory:
        with self.assertRaisesRegex(RuntimeError, "Missing LAMMPS workflow configuration"):
            gp._validate_result({"run_root": Path(directory)})

 def test_monitor_waits_for_handoff(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gp._initialize(root, {"array_job_id": "123", "tasks": {
            "a": {"status": "done"}}})

        def finish(_):
            with gp._locked(root) as state:
                state["summary"] = {"phase": "lammps", "failed": 0}


        with patch.object(gp, "_array_active", return_value=True) as active, \
             patch.object(gp.time, "sleep", side_effect=finish):
            self.assertEqual(gp.wait(root, poll_interval=0), 0)
        active.assert_called_once_with("123")

 def test_monitor_reports_incomplete_array(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gp._initialize(root, {"array_job_id": "123", "tasks": {
            "a": {"status": "running"}}})
        with patch.object(gp, "_array_active", return_value=False):
            self.assertEqual(gp.wait(root, poll_interval=0), 1)

 def test_pool_rejects_other_phases_before_submission(self):
    for phase in ("all", "dpnegf"):
        with self.assertRaisesRegex(ValueError, "phase=lammps"):
            gp.submit(SimpleNamespace(phase=phase), {}, [], ROOT, "unused")

 def test_parallel_publication_keeps_every_replica(self):
    import split_workflow
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tasks = []
        for i in range(4):
            run = root / "config" / f"replica_{i:03d}"
            run.mkdir(parents=True)
            (run / "workflow_config.json").write_text('{"md_steps": 0}')
            (run / "STRUCT.fdf").write_text("structure")
            (run / "structure.xyz").write_text("xyz")
            tasks.append({"task_id": str(i), "run_root": run,
                          "configuration_root": run.parent})
        gp._initialize(root, {"generation": "g"})
        with patch.object(split_workflow, "validate_structures",
                          side_effect=lambda run, _: [run / "STRUCT.fdf", run / "structure.xyz"]):
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda task: gp._publish_result(root, "g", task), tasks))
        manifest = json.loads((root / "transfer_manifest.json").read_text())
        self.assertEqual(len(manifest["replicas"]), 4)
        self.assertEqual(len(manifest["files"]), 12)

 def test_failed_lammps_cannot_pass_result_validation(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workflow_config.json").write_text('{"md_steps": 100}')
        (root / "lammps").mkdir()
        (root / "lammps/lammps_done.flag").touch()
        (root / "lammps/lammps_failed.flag").touch()
        with self.assertRaisesRegex(RuntimeError, "not successfully completed"):
            gp._validate_result({"run_root": root})


class ManifestRecoveryTests(unittest.TestCase):
    def prepare(self, root):
        config_path = root / "batch.json"
        config_path.write_text("{}", encoding="utf-8")
        tasks = [{"task_id": str(i), "run_root": root / "config" / f"replica_{i:03d}",
                  "configuration_root": root / "config"} for i in range(3)]
        saved = {"md_steps": 100, "lammps_mode": "md", "chirality": [18, 0]}
        for task in tasks[:2]:
            run = task["run_root"]
            (run / "lammps").mkdir(parents=True)
            (run / "lammps/lammps_done.flag").touch()
            (run / "workflow_config.json").write_text(json.dumps(saved), encoding="utf-8")
            leaf = run / "dpnegf/100"
            leaf.mkdir(parents=True)
            (leaf / "STRUCT.fdf").write_text("fdf")
            (leaf / "18_0.xyz").write_text("xyz")
        gp._initialize(root, {
            "phase": "lammps", "config_hash": gp._config_hash(config_path),
            "task_ids": ["0", "1", "2"], "array_job_id": None,
            "generation": "old", "finalized": False,
            "tasks": {str(i): {"status": "done" if i < 2 else "pending", "owner": None}
                      for i in range(3)},
        })
        args = SimpleNamespace(phase="lammps", para=2, pool_cpus=1, worker_time="08:00:00",
                               pool_partition=None, pool_mem=None, pool_setup=None,
                               pool_retry_failed=False)
        return args, {"data_root": root}, tasks, config_path, saved

    @staticmethod
    def validate_fixture(run, config):
        # Exercise real manifest merging/hashing with small fixture payloads;
        # structural parsing is covered by the existing split-workflow tests.
        paths = [run / "dpnegf/100/STRUCT.fdf", run / "dpnegf/100/18_0.xyz"]
        if any(not path.is_file() or not path.read_text() for path in paths):
            raise ValueError(f"Invalid converted structure: {run}")
        return paths

    def test_submit_restores_deleted_or_partial_manifest_without_recomputing(self):
        import split_workflow as sw
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, batch, tasks, config, saved = self.prepare(root)
                fake_bg = SimpleNamespace(make_task_config=lambda *_: saved)
                with patch.dict(sys.modules, {"batch_generate": fake_bg}), \
                     patch.object(sw, "validate_structures", side_effect=self.validate_fixture), \
                     patch.object(gp.subprocess, "run", return_value=SimpleNamespace(
                         returncode=0, stdout="12345\n", stderr="")) as sbatch:
                    sw.export_manifest(root, tasks[:1] if partial else tasks[:2])
                    if not partial:
                        (root / "transfer_manifest.json").unlink()
                    before = {p: p.read_bytes() for t in tasks[:2]
                              for p in t["run_root"].rglob("*") if p.is_file()}
                    self.assertEqual(gp.submit(args, batch, tasks, ROOT, config), 0)
                    manifest = sw.load_manifest(root, check_replica_directories=False)
                self.assertEqual(len(manifest["replicas"]), 2)
                self.assertEqual(len(manifest["files"]), 6)
                sbatch.assert_called_once()
                self.assertEqual(sbatch.call_args.args[0][0], "sbatch")
                self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
                state = json.loads((root / ".cnt_gpu_pool/state.json").read_text())
                self.assertEqual(gp._claim(root, state["generation"], "worker"), "2")
                self.assertEqual([state["tasks"][str(i)]["status"] for i in range(3)],
                                 ["done", "done", "pending"])

    def test_invalid_completed_outputs_block_recovery_and_submission(self):
        import split_workflow as sw
        for fault in ("config", "structure", "flag"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, batch, tasks, config, saved = self.prepare(root)
                run = tasks[1]["run_root"]
                if fault == "config":
                    (run / "workflow_config.json").write_text('{"md_steps": 200}')
                elif fault == "structure":
                    (run / "dpnegf/100/18_0.xyz").unlink()
                else:
                    (run / "lammps/lammps_done.flag").unlink()
                state_path = root / ".cnt_gpu_pool/state.json"
                before = state_path.read_bytes()
                with patch.dict(sys.modules, {"batch_generate": SimpleNamespace(make_task_config=lambda *_: saved)}), \
                     patch.object(sw, "validate_structures", side_effect=self.validate_fixture), \
                     patch.object(gp.subprocess, "run") as sbatch:
                    with self.assertRaises(ValueError):
                        gp.submit(args, batch, tasks, ROOT, config)
                sbatch.assert_not_called()
                self.assertFalse((root / "transfer_manifest.json").exists())
                self.assertEqual(state_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
