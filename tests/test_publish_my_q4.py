import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'scripts' / 'publish_my_q4.py'
spec = importlib.util.spec_from_file_location('publish_my_q4', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
lock_source_version = module.lock_source_version


class TestBeginnerSourceLock(unittest.TestCase):
    def test_reuse_same_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            lock_source_version(folder, "Qwen/Qwen3-0.6B", "abc123")
            lock_source_version(folder, "Qwen/Qwen3-0.6B", "abc123")
            self.assertEqual(json.loads((folder / "source-version.json").read_text())["revision"], "abc123")

    def test_changed_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            lock_source_version(folder, "Qwen/Qwen3-0.6B", "abc123")
            with self.assertRaisesRegex(RuntimeError, "version changed"):
                lock_source_version(folder, "Qwen/Qwen3-0.6B", "def456")

    def test_existing_untracked_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "original-f16.gguf").write_bytes(b"GGUF")
            with self.assertRaisesRegex(RuntimeError, "no version record"):
                lock_source_version(folder, "Qwen/Qwen3-0.6B", "abc123")

    def test_existing_recorded_original_can_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "original-f16.gguf").write_bytes(b"GGUF")
            (folder / "publish-receipt.json").write_text(json.dumps({
                "source_model": "Qwen/Qwen3-0.6B",
                "source_revision": "abc123",
            }))
            lock_source_version(folder, "Qwen/Qwen3-0.6B", "abc123")
            self.assertTrue((folder / "source-version.json").is_file())


if __name__ == "__main__":
    unittest.main()
