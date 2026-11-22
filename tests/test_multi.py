import unittest
from q4multi import pattern_for, select_recipe, passes_tuning
import re

class TestMulti(unittest.TestCase):
    def test_layer_group(self):
        p = pattern_for("ffn_down@12:17")
        self.assertTrue(re.search(p, "blk.12.ffn_down.weight"))
        self.assertTrue(re.search(p, "blk.17.ffn_down.weight"))
        self.assertFalse(re.search(p, "blk.18.ffn_down.weight"))
        self.assertFalse(re.search(p, "blk.12.ffn_up.weight"))

    def test_bad_layer(self):
        with self.assertRaises(ValueError):
            pattern_for("ffn_gate@18:12")

    def test_minimax(self):
        bases = {"text": 0.1, "code": 0.05}
        trials = [
            {"group": "ffn_down@0:17", "type": "Q5_K", "delta_bytes": 80, "deltas": {"text": -0.03, "code": 0.02}, "scores": {}},
            {"group": "ffn_gate@0:17", "type": "Q3_K", "delta_bytes": -90, "deltas": {"text": -0.01, "code": -0.01}, "scores": {}},
        ]
        winner = select_recipe(["ffn_down@0:17", "ffn_gate@0:17"], trials, bases, 1000, 1000)
        self.assertEqual([t["type"] for t in winner[1]], ["Q4_K", "Q3_K"])
        self.assertAlmostEqual(winner[0][0], .9)

    def test_acceptance(self):
        observed = {"wiki": {"kld_ratio": 0.95, "same_top_percent": 88.0}, "code": {"kld_ratio": 0.98, "same_top_percent": 92.4}}
        bases = {"wiki": 88.2, "code": 92.6}
        self.assertTrue(passes_tuning(observed, bases, True))
        self.assertFalse(passes_tuning(observed, bases, False))
        observed["code"]["same_top_percent"] = 91.6
        self.assertFalse(passes_tuning(observed, bases, True))
        observed["code"]["same_top_percent"] = 92.5
        observed["code"]["kld_ratio"] = 1.01
        self.assertFalse(passes_tuning(observed, bases, True))

if __name__ == "__main__":
    unittest.main()
