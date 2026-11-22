import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestThinkingManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = json.loads((ROOT / "benchmarks/QWEN3_4B_TASK_MANIFEST.json").read_text(encoding="utf-8"))
        cls.new = json.loads((ROOT / "benchmarks/QWEN3_4B_THINKING_MANIFEST.json").read_text(encoding="utf-8"))

    def test_disjoint_samples(self):
        old_math = {x["source_index"] for x in self.old["items"] if x["domain"] == "mgsm_en"}
        new_math = {x["source_index"] for x in self.new["items"] if x["domain"] == "mgsm_en"}
        old_code = {x["source_task_id"] for x in self.old["items"] if x["domain"] == "mbpp"}
        new_code = {x["source_task_id"] for x in self.new["items"] if x["domain"] == "mbpp"}
        self.assertFalse(old_math & new_math)
        self.assertFalse(old_code & new_code)
        self.assertEqual(len(new_math), 48)
        self.assertEqual(len(new_code), 24)

    def test_language_alignment(self):
        en = {x["source_index"] for x in self.new["items"] if x["domain"] == "mgsm_en"}
        zh = {x["source_index"] for x in self.new["items"] if x["domain"] == "mgsm_zh"}
        self.assertEqual(en, zh)
        self.assertEqual(len(self.new["items"]), 120)

    def test_thinking_protocol(self):
        self.assertEqual(self.new["excluded_previous_suite_sha256"], self.old["frozen_tasks_sha256"])
        self.assertEqual(self.new["thinking_mode"]["reasoning_budget"], 256)
        self.assertEqual(self.new["decode"]["math_max_tokens"], 1024)
        self.assertEqual(self.new["decode"]["code_max_tokens"], 1024)
        self.assertTrue(self.new["thinking_mode"]["enabled"])
        self.assertEqual(self.old["sources"]["mgsm_en"]["full_source_jsonl_sha256"], self.new["sources"]["mgsm_en"]["full_source_jsonl_sha256"])


if __name__ == "__main__":
    unittest.main()
