import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stage")]
import batch_config
import batch_generate
import cnt_geometry
import ele_multi_defects_ele as ele
from defect_angles import normalize_angle_range
from negf_provenance import compute_negf_config_hash


class DefectAngleTests(unittest.TestCase):
    def sample(self, angles=None):
        T, _, pl, length = cnt_geometry.geo_info(5, 5, 6.5, 12, 2)
        tube = cnt_geometry.build_unit_cnt(5, 5) * (1, 1, length)
        cnt_geometry.set_reference_cyl(tube)
        kwargs = {} if angles is None else {"defect_angle_range_deg": angles}
        return ele.generate_defect_coords(tube, 4, T, pl, 12, 5, 5,
                                         seed=42, buffer_pl=2, **kwargs)

    def test_actual_anchor_range_and_reproducibility(self):
        for bounds in ([0, 90], [330, 390]):
            coords, indices = self.sample(bounds)
            self.assertEqual((coords, indices), self.sample(bounds))
            self.assertEqual(len(set(indices)), 4)
            for theta, _ in coords:
                self.assertLessEqual((np.rad2deg(theta) - bounds[0]) % 360,
                                     bounds[1] - bounds[0])

    def test_full_circle_compatibility(self):
        self.assertEqual(self.sample(), self.sample([0, 360]))
        self.assertEqual(self.sample(), self.sample([-180, 180]))

    def test_validation(self):
        for value in ([0, 0], [90, 0], [0, 361], [0, float('inf')],
                      [False, 90], None, [0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_angle_range(value)
        self.assertEqual(normalize_angle_range([-30, 30]), [330, 390])

    def test_hash_compatibility(self):
        base = {"seed": 42}
        self.assertEqual(compute_negf_config_hash(base), compute_negf_config_hash(
            {**base, "defect_angle_range_deg": [0, 360]}))
        self.assertNotEqual(compute_negf_config_hash(base), compute_negf_config_hash(
            {**base, "defect_angle_range_deg": [0, 90]}))

    def test_batch_override_reaches_workflow(self):
        raw = {"temperatures": [100], "template": {"defect_angle_range_deg": [0, 180]},
               "configs": [{"m": 5, "n": 5, "l_def": 12, "N_defects": 4,
                            "structures": ["MVH"], "replicas": 1}]}
        for override, expected in ((None, [0, 180]), ([0, 90], [0, 90])):
            data = copy.deepcopy(raw)
            if override is not None:
                data["configs"][0]["defect_angle_range_deg"] = override
            batch = batch_config._validate_batch(data, ROOT / "batch.json")
            task = batch_generate.build_tasks(batch, ROOT / "data")[0]
            config = batch_generate.make_task_config(task, batch)
            self.assertEqual(config["defect_angle_range_deg"], expected)


if __name__ == "__main__":
    unittest.main()
