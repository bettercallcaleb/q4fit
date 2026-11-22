import json
import tempfile
import unittest
from pathlib import Path

import q4fit

class TestQ4Fit(unittest.TestCase):
    def setUp(self):
        self.ts = [
            dict(name="blk.0.attn_v.weight", shape=[256, 1024], n=262144, bytes=524288, source="BF16", eligible=True),
            dict(name="blk.1.attn_v.weight", shape=[256, 1024], n=262144, bytes=524288, source="BF16", eligible=True),
            dict(name="blk.0.attn_norm.weight", shape=[256], n=256, bytes=1024, source="F32", eligible=False),
        ]

    def test_size(self):
        self.assertEqual(q4fit.quant_bytes(self.ts[0], "Q4_K"), 147456)
        self.assertEqual(q4fit.quant_bytes(self.ts[0], "Q5_K"), 180224)
        self.assertEqual(q4fit.quant_bytes(self.ts[2], "Q4_K"), 1024)

    def test_rules(self):
        o = q4fit.override_map(self.ts, [r"^blk\.0\.attn_v\.weight$=Q6_K", r"blk\.1\.attn_v=Q3_K"])
        self.assertEqual(o["blk.0.attn_v.weight"], "Q6_K")
        self.assertEqual(o["blk.1.attn_v.weight"], "Q3_K")

    def test_unmatched(self):
        with self.assertRaises(ValueError):
            q4fit.override_map(self.ts, ["unmatched=Q5_K"])

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            q4fit.override_map(self.ts, ["attn_v=BAD"])

    def test_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            file, total = q4fit.recipe(self.ts, "Q4_K", {"blk.0.attn_v.weight": "Q5_K"}, Path(d))
            self.assertIn("blk.0.attn_v.weight=q5_k", file.read_text())
            self.assertEqual(json.loads((Path(d) / "recipe.json").read_text())["estimated_bytes"], total)

    def test_metric(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.log"
            f.write_text("mean KLD: 0.0198")
            self.assertEqual(q4fit.get_kl(f), 0.0198)
            f.write_text("Mean    KLD:   0.085821\nMaximum KLD: 0.907946\n99.0% KLD: 0.722835\nMinimum KLD: 0.000431\nSame top p: 90.551")
            self.assertAlmostEqual(q4fit.get_kl(f), 0.085821)
            self.assertAlmostEqual(q4fit.read_metric(f, r"^\s*99\.0%\s+KLD:\s*([0-9.]+)"), 0.722835)
            f.write_text("no metric")
            self.assertIsNone(q4fit.get_kl(f))

if __name__ == "__main__":
    unittest.main()
