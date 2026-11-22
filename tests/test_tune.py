import unittest

from q4tune import beam_search, q5_anchors


class TestTensorOptimizer(unittest.TestCase):
    def test_pair_under_budget(self):
        baselines = {"wiki": {"kld": 0.1}, "code": {"kld": 0.05}}
        trials = [
            {"name": "a", "type": "Q3_K", "delta_bytes": -100, "relative_kld": {"wiki": 0.2, "code": 0.1}},
            {"name": "b", "type": "Q5_K", "delta_bytes": 100, "relative_kld": {"wiki": -0.4, "code": -0.3}},
        ]
        ranked = beam_search(trials, baselines, 1000, 1000, beam_width=100, bucket_width=10, max_edits=2)
        self.assertTrue(ranked)
        self.assertEqual({tuple(x) for x in ranked[0]["edits"]}, {("a", "Q3_K"), ("b", "Q5_K")})
        self.assertAlmostEqual(ranked[0]["predicted_worst_ratio"], 0.8)
        self.assertEqual(ranked[0]["predicted_bytes"], 1000)

    def test_only_one_type_per_tensor(self):
        baselines = {"wiki": {"kld": 0.1}, "code": {"kld": 0.05}}
        trials = [
            {"name": "a", "type": "Q3_K", "delta_bytes": -100, "relative_kld": {"wiki": 0.1, "code": 0.1}},
            {"name": "a", "type": "Q5_K", "delta_bytes": 100, "relative_kld": {"wiki": -0.4, "code": -0.3}},
            {"name": "b", "type": "Q3_K", "delta_bytes": -100, "relative_kld": {"wiki": 0.01, "code": 0.01}},
        ]
        ranked = beam_search(trials, baselines, 1000, 1000, beam_width=100, bucket_width=10)
        for candidate in ranked:
            names = [n for n, _ in candidate["edits"]]
            self.assertEqual(len(set(names)), len(names))

    def test_q5_anchor_selection(self):
        bases = {"wiki": {"kld": 0.1}, "code": {"kld": 0.05}}
        rows = [
            {"name": "a", "type": "Q5_K", "delta_bytes": 100, "relative_kld": {"wiki": -0.03, "code": -0.08}},
            {"name": "b", "type": "Q5_K", "delta_bytes": 100, "relative_kld": {"wiki": -0.1, "code": -0.09}},
            {"name": "c", "type": "Q3_K", "delta_bytes": -100, "relative_kld": {"wiki": 0.01, "code": 0.01}},
        ]
        anchors = q5_anchors(rows, bases, 1000, 1200, limit=2)
        self.assertEqual({tuple(x) for x in anchors[0]["edits"]}, {("a", "Q5_K"), ("b", "Q5_K")})
        self.assertEqual(anchors[0]["predicted_bytes"], 1200)
        self.assertEqual(len(anchors), 2)

    def test_budget_and_edit_limit(self):
        baselines = {"wiki": {"kld": 0.1}, "code": {"kld": 0.05}}
        trials = [
            {"name": str(i), "type": "Q5_K", "delta_bytes": 100, "relative_kld": {"wiki": -0.1, "code": -0.1}}
            for i in range(4)
        ]
        ranked = beam_search(trials, baselines, 1000, 1120, beam_width=100, bucket_width=10, max_edits=2)
        self.assertTrue(ranked)
        self.assertEqual(len(ranked[0]["edits"]), 1)
        self.assertTrue(all(x["predicted_bytes"] <= 1120 for x in ranked))


if __name__ == "__main__":
    unittest.main()
