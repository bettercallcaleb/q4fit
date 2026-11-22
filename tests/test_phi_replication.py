import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def load(name):
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPhiReplication(unittest.TestCase):
    def test_disjoint_source_rows(self):
        phi = load("PHI4MINI_TASK_MANIFEST.json")
        q5 = load("QWEN3_4B_TASK_MANIFEST.json")
        q6 = load("QWEN3_4B_THINKING_MANIFEST.json")
        old_math = {x["source_index"] for manifest in (q5, q6) for x in manifest["items"] if x["domain"] == "mgsm_en"}
        old_code = {x["source_task_id"] for manifest in (q5, q6) for x in manifest["items"] if x["domain"] == "mbpp"}
        math_en = {x["source_index"] for x in phi["items"] if x["domain"] == "mgsm_en"}
        math_zh = {x["source_index"] for x in phi["items"] if x["domain"] == "mgsm_zh"}
        code = {x["source_task_id"] for x in phi["items"] if x["domain"] == "mbpp"}
        self.assertFalse(old_math & math_en)
        self.assertFalse(old_code & code)
        self.assertEqual(math_en, math_zh)
        self.assertEqual((len(math_en), len(code), len(phi["items"])), (24, 16, 64))

    def test_preselected_recipe(self):
        plan = load("PHI4MINI_SEARCH_PLAN.json")
        selected = load("PHI4MINI_SELECTED.json")
        recipe = ROOT / selected["recipe_file"]
        self.assertEqual(selected["search_plan_sha256"], sha(BENCH / "PHI4MINI_SEARCH_PLAN.json"))
        self.assertEqual(selected["recipe_sha256"], sha(recipe))
        self.assertEqual(selected["source_gguf_sha256"], plan["source_gguf_bf16_sha256"])
        self.assertEqual(selected["frozen_task_suite_sha256"], load("PHI4MINI_TASK_MANIFEST.json")["frozen_tasks_sha256"])
        self.assertEqual(selected["trial_count"], len(plan["probes"]))
        self.assertEqual(len(selected["selected"]["edits"]), 6)
        self.assertLessEqual(selected["selected"]["actual_bytes"], plan["budget_bytes"])

    def test_ablation_labeled_exploratory(self):
        a = load("PHI4MINI_ABLATION.json")
        self.assertIn("post-hoc", a["status"])
        self.assertGreater(a["share_of_full_q4ks_kld_drop_explained_by_embedding_alone"]["wiki_holdout"], 0.8)
        self.assertGreater(a["share_of_full_q4ks_kld_drop_explained_by_embedding_alone"]["code_holdout"], 0.8)
        amounts = {k: v["bytes"] for k, v in a["samples"].items()}
        self.assertEqual(amounts["standard_q4ks"] - amounts["pure_q4"], (amounts["embedding_q6_only"] - amounts["pure_q4"]) + (amounts["eight_q5_only"] - amounts["pure_q4"]))

    def test_published_scores(self):
        result = load("PHI4MINI_RESULTS.json")
        q4, fit = result["holdout"]["q4"], result["holdout"]["q4fit"]
        self.assertEqual(result["selected_sha256"], load("PHI4MINI_SELECTED.json")["selected"]["gguf_sha256"])
        self.assertTrue(result["direct_quant_byte_identical"])
        self.assertLess(fit["wiki_holdout"]["kld"], q4["wiki_holdout"]["kld"])
        self.assertLess(fit["code_holdout"]["kld"], q4["code_holdout"]["kld"])
        self.assertEqual(result["task_scores"]["suite_sha256"], load("PHI4MINI_TASK_MANIFEST.json")["frozen_tasks_sha256"])
        self.assertEqual(result["task_scores"]["models"]["q4fit"]["mbpp"]["correct"], 12)


if __name__ == "__main__":
    unittest.main()
