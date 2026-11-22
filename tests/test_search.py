import unittest

from q4search import choose

class TestSearch(unittest.TestCase):
    def test_budgeted_trade(self):
        trials = [
            dict(group="ffn_down", typ="Q5_K", delta_bytes=100, delta_kld=-0.020, kld=0.06),
            dict(group="ffn_gate", typ="Q3_K", delta_bytes=-110, delta_kld=0.010, kld=0.09),
        ]
        selected = choose(["ffn_down", "ffn_gate"], trials, 1000, 1000)
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected[0][0], -0.010)
        self.assertLessEqual(selected[0][1], 1000)

    def test_no_feasible_choice(self):
        self.assertIsNone(choose(["ffn_down"], [], 1000, 900))
