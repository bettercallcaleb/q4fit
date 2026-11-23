import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from conceptladder.cli import fingerprint, load_packs, materialize, parse_letter
from q4depth import build_comparison, main, render_report


SERVER = r'''#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["messages"][-1]["role"] == "user"
        body = json.dumps({"choices": [{"message": {"content": "A"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
'''


class DepthTests(unittest.TestCase):
    def test_packs_match_independent_release(self):
        items = load_packs()
        self.assertEqual(len(items), 36)
        self.assertEqual(fingerprint(items), "158500c2f341adb653914ead56867ee83a60fe6b830b9c25cb81fb64db609a8a")
        for pack in ("tcp", "sqlite", "tls13"):
            self.assertEqual(len(load_packs(pack)), 12)

    def test_labeled_answers(self):
        self.assertEqual(parse_letter("B. Readers and one writer can proceed concurrently"), "B")
        self.assertEqual(parse_letter("Answer: A. No; there is still only one writer at a time"), "A")

    def test_hard_suites(self):
        from conceptladder.cli import load_suite
        items = load_suite("hard", "all", 17, 4)
        self.assertEqual(len(items), 240)
        self.assertEqual(len(load_suite("hard", "security-hard", 17, 2)), 24)

    def test_comparison_and_report(self):
        items = load_packs("tcp")
        before = {i["id"]: True for i in items}
        after = dict(before)
        after[items[0]["id"]] = False
        a = {"accuracy": 1.0, "correct": 12, "mean_contiguous_depth": 4.0, "depths": {f"tcp/{x}": 4 for x in ("handshake", "flow-control", "fin")}, "pass": before}
        b = {"accuracy": 11 / 12, "correct": 11, "mean_contiguous_depth": 8 / 3, "depths": {f"tcp/{x}": (0 if x == "handshake" else 4) for x in ("handshake", "flow-control", "fin")}, "pass": after}
        reference = {"model": "original", "seed": 17, "dataset_sha256": fingerprint(items)}
        candidate = {"model": "q4", "seed": 17, "dataset_sha256": fingerprint(items)}
        comparison = build_comparison(reference, candidate, a, b)
        self.assertEqual(comparison["lost"], [items[0]["id"]])
        self.assertEqual(comparison["gained"], [])
        self.assertAlmostEqual(comparison["knowledge_loss_rate"], 1 / 12, places=5)
        report = render_report(comparison, "original.gguf", "q4.gguf", {"reference": 1, "candidate": 2})
        self.assertIn("handshake", report)
        self.assertIn("Reference-correct questions lost: **1**", report)

    def test_two_models_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original = folder / "original.gguf"
            q4 = folder / "q4.gguf"
            original.write_bytes(b"mock-original")
            q4.write_bytes(b"mock-quant")
            server = folder / "fake-server"
            server.write_text(SERVER, encoding="utf-8")
            server.chmod(server.stat().st_mode | stat.S_IXUSR)
            output = folder / "report"
            result = main([
                "--reference", str(original),
                "--candidate", str(q4),
                "--llama-server", str(server),
                "--out", str(output),
                "--suite", "classic",
                "--packs", "tcp",
                "--port", str(self._unused_port()),
                "--startup-timeout", "15",
                "--request-timeout", "5",
            ])
            self.assertEqual(result, 0)
            comparison = json.loads((output / "comparison.json").read_text())
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["items"], 12)
            self.assertEqual(receipt["benchmark"], "conceptladder-bench/0.1.0-classic")
            self.assertEqual(len(comparison["reference_depths"]), 3)
            self.assertEqual(len(comparison["lost"]), 0)
            self.assertEqual(comparison["reference_accuracy"], comparison["candidate_accuracy"])
            self.assertTrue((output / "report.md").is_file())

    def test_hard_two_model_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original = folder / "original.gguf"
            quant = folder / "quant.gguf"
            original.write_bytes(b"original-hard-test")
            quant.write_bytes(b"quantized-hard-test")
            server = folder / "fake-server"
            server.write_text(SERVER, encoding="utf-8")
            server.chmod(server.stat().st_mode | stat.S_IXUSR)
            output = folder / "hard-results"
            self.assertEqual(main([
                "--reference", str(original),
                "--candidate", str(quant),
                "--llama-server", str(server),
                "--out", str(output),
                "--suite", "hard",
                "--variants", "1",
                "--packs", "db-hard",
                "--port", str(self._unused_port()),
                "--startup-timeout", "15",
                "--request-timeout", "5",
            ]), 0)
            receipt = json.loads((output / "receipt.json").read_text())
            reference = json.loads((output / "reference.json").read_text())
            summary = json.loads((output / "comparison.json").read_text())
            self.assertEqual(receipt["items"], 12)
            self.assertEqual(receipt["suite"], "hard")
            self.assertEqual(receipt["benchmark"], "conceptladder-bench/0.2.0-hard")
            self.assertEqual(reference["suite"], "hard")
            self.assertEqual(reference["variants"], 1)
            self.assertEqual(len(summary["reference_depths"]), 3)
            self.assertIn("12-question", (output / "report.md").read_text())

    @staticmethod
    def _unused_port():
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
