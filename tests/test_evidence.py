import json
import tempfile
import unittest
from pathlib import Path

from q4evidence import build, main


class EvidenceTests(unittest.TestCase):
    def fixture(self, folder):
        a, b, c, d = "a" * 64, "b" * 64, "c" * 64, "d" * 64
        receipt = {
            "format": "q4fit-depth-v1",
            "benchmark": "conceptladder-bench/0.3.1-ladder",
            "suite": "ladder",
            "dataset_sha256": a,
            "items": 12,
            "variants": 1,
            "seed": 17,
            "answer_seed": 63,
            "inference_seed": 5,
            "context": 4096,
            "gpu_layers": 99,
            "temperature": 0,
            "allow_thinking": False,
            "server_sha256": d,
            "server": "/private/work/llama-server",
            "reference": {"path": "/private/base.gguf", "sha256": b, "bytes": 128},
            "candidate": {"path": "/private/quant.gguf", "sha256": c, "bytes": 64},
        }
        comparison = {
            "dataset_sha256": a,
            "seed": 17,
            "answer_seed": 63,
            "reference_model": "reference",
            "candidate_model": "candidate",
            "reference_accuracy": 0.75,
            "candidate_accuracy": 0.5,
            "accuracy_delta": -0.25,
            "reference_mean_depth": 3.0,
            "candidate_mean_depth": 2.0,
            "depth_delta": -1.0,
            "reference_correct": 9,
            "retained_when_reference_correct": 6,
            "knowledge_loss_rate": 0.333333,
            "lost": ["example-1", "example-2", "example-3"],
            "gained": [],
        }
        for name, value in (
            ("receipt.json", receipt),
            ("comparison.json", comparison),
            ("reference.json", {"model": "reference", "dataset_sha256": a, "seed": 17, "answer_seed": 63}),
            ("candidate.json", {"model": "candidate", "dataset_sha256": a, "seed": 17, "answer_seed": 63}),
        ):
            (folder / name).write_text(json.dumps(value), encoding="utf-8")
        return receipt, comparison

    def test_export_redacts_paths_and_keeps_qualifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            record = build(folder)
            self.assertEqual(record["schema"], "ramgpt-experiment-v1")
            self.assertEqual(record["status"], "exploratory")
            self.assertEqual(record["results"]["lost_count"], 3)
            self.assertEqual(len(record["evidence_sha256"]), 4)
            text = json.dumps(record)
            self.assertNotIn("/private/", text)
            self.assertNotIn("verified", record["status"])
            output = folder / "evidence.json"
            self.assertEqual(main(["--depth-dir", str(folder), "--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), record)
            with self.assertRaises(SystemExit):
                main(["--depth-dir", str(folder), "--out", str(output)])

    def test_rejects_inconsistent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            path = folder / "comparison.json"
            value = json.loads(path.read_text())
            value["answer_seed"] = 99
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                build(folder)

    def test_rejects_missing_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Missing completed"):
                build(folder)
            self.fixture(folder)
            path = folder / "receipt.json"
            value = json.loads(path.read_text())
            value["reference"]["sha256"] = "not-a-hash"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build(folder)


if __name__ == "__main__":
    unittest.main()
