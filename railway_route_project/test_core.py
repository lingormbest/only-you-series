"""回归测试；人工构造数据不代表实际工程调查结果。"""

import math
import unittest

from economics import (
    evaluate_cost, evaluate_soft, interpolate,
    read_inventory, validate_structures,
)
from main import (
    Point, cnn_prior, coarse_check, grade_sections,
    positive_linear_integral, turn_angle, uct_score,
)

class FlatDEM:
    def sample(self, x, y):
        return 0.0

class CoreTests(unittest.TestCase):
    def test_fss(self):
        cfg = {"lambda_max": 1.2, "max_grade_permille": 20}
        start, end = Point(0, 0, 0), Point(1000, 0, 0)
        self.assertTrue(
            coarse_check(start, end, start, Point(500, 0, 0), 0, cfg)["passed"]
        )
        self.assertFalse(
            coarse_check(start, end, start, Point(500, 0, 100), 0, cfg)["passed"]
        )

    def test_straight_and_reverse(self):
        a, b = Point(0, 0, 0), Point(100, 0, 0)
        self.assertAlmostEqual(turn_angle(a, b, Point(200, 0, 0)), 0)
        self.assertAlmostEqual(turn_angle(a, b, a), math.pi)

    def test_positive_area_cross_zero(self):
        self.assertAlmostEqual(
            positive_linear_integral(-10, 10, 100), 250
        )
        self.assertAlmostEqual(
            positive_linear_integral(10, 10, 100), 1000
        )

    def test_interpolation_no_extrapolation(self):
        self.assertEqual(interpolate(5, [[0, 10], [10, 20]]), 15)
        self.assertIsNone(interpolate(11, [[0, 10], [10, 20]]))

    def test_prior_uses_required_channel(self):
        result = cnn_prior(
            Point(0, 0, 0), Point(100, 0, 0),
            FlatDEM(),
            lambda x, y: [0.9, 0.09, 0.01],
            lambda x, y, z, ground: "bridge",
        )
        self.assertAlmostEqual(result["value"], 0.09)

    def test_zero_prior_has_floor(self):
        result = cnn_prior(
            Point(0, 0, 0), Point(100, 0, 0),
            FlatDEM(),
            lambda x, y: [1.0, 0.0, 0.0],
            lambda x, y, z, ground: "bridge",
        )
        self.assertAlmostEqual(result["value"], 0.01)

    def test_unvisited_prior_affects_ranking(self):
        high = uct_score(None, 1, 0, 1, 0.9, 0)
        low = uct_score(None, 1, 0, 1, 0.1, 0)
        self.assertGreater(high, low)

    def test_overlapping_structures_rejected(self):
        records = [
            {"id": "T1", "type": "tunnel", "start_m": 0, "end_m": 600},
            {"id": "B1", "type": "bridge", "start_m": 500, "end_m": 800},
        ]
        with self.assertRaises(ValueError):
            validate_structures(records, 1000)

    def test_unknown_inventory_is_not_empty(self):
        engineering = {
            "structures_status": "unknown",
            "structures": None,
            "quantities": {},
        }
        result = evaluate_cost(5000, engineering, {"track_yuan_per_m": 5200})
        self.assertEqual(result["structures_status"], "unknown")
        self.assertIsNone(result["structure_inventory"])
        self.assertIsNone(result["structure_cost_details"])
        self.assertIsNone(result["components_yuan"]["bridge"])
        self.assertIsNone(result["components_yuan"]["tunnel"])
        self.assertEqual(result["known_components_sum_yuan"], 26000000)

    def test_empty_list_cannot_silently_mean_absent(self):
        with self.assertRaises(ValueError):
            read_inventory(
                {"structures_status": "unknown", "structures": []},
                5000,
            )

    def test_grade_sections_merge_samples(self):
        points = [
            Point(0, 0, 0),
            Point(100, 0, 0.72),
            Point(200, 0, 1.44),
        ]
        sections = grade_sections(points)
        self.assertEqual(len(sections), 1)
        self.assertAlmostEqual(sections[0]["length_m"], 200)

    def test_ideal_grade_is_12(self):
        config = {
            "ideal_grade_permille": 12,
            "environment": {"enabled": False},
        }
        below = evaluate_soft(
            {}, config, [{"grade_permille": -7.2, "length_m": 5000}]
        )
        above = evaluate_soft(
            {}, config, [{"grade_permille": -15, "length_m": 1000}]
        )
        self.assertEqual(
            below["items"]["grade"]["max_excess_permille"], 0
        )
        self.assertEqual(
            above["items"]["grade"]["max_excess_permille"], 3
        )
        self.assertEqual(
            above["items"]["grade"]["excess_integral_permille_m"], 3000
        )
        self.assertEqual(above["items"]["grade"]["penalty_yuan"], 0)

    def test_missing_environment_is_not_zero(self):
        config = {
            "ideal_grade_permille": 12,
            "environment": {
                "enabled": True,
                "threshold_m": 0,
                "alpha": 0.1,
                "base_price_yuan_per_m": None,
            },
        }
        self.assertIsNone(
            evaluate_soft({}, config)["soft_penalty_yuan"]
        )

if __name__ == "__main__":
    unittest.main()