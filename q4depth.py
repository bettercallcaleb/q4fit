import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from conceptladder.cli import create_run, fingerprint, grade, load_suite, write_json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_server(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    found = shutil.which(value)
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(f"llama-server not found: {value}")


def ensure_port_available(port):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use; choose another --port")


def server_command(args, gguf, server, label):
    command = [
        str(server), "-m", str(gguf), "--alias", label,
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", str(args.context), "-np", "1",
        "-ngl", str(args.ngl), "-t", str(args.threads),
        "--seed", str(args.seed), "--jinja", "--no-webui",
    ]
    if not args.allow_thinking:
        command += ["--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}']
    return command


def ready(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
            status = json.load(response)
            return response.status == 200 and status.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def evaluate(args, gguf, label, items, server, folder):
    ensure_port_available(args.port)
    log_path = folder / f"{label}-server.log"
    command = server_command(args, gguf, server, label)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited for {label}; see {log_path}")
                if ready(url):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(f"llama-server startup timed out for {label}; see {log_path}")

            print(f"Evaluating {label}: {gguf}", flush=True)
            request_args = SimpleNamespace(
                url=url + "/v1",
                model=label,
                api_key="",
                timeout=args.request_timeout,
                max_tokens=args.max_tokens,
                seed=args.seed,
                delay=0,
                suite=args.suite,
                packs=args.packs,
                variants=args.variants,
            )
            run = create_run(request_args, items)
            output = folder / f"{label}.json"
            write_json(str(output), run)
            errors = [row for row in run["responses"] if row["error"] not in (None, "unparseable")]
            if errors:
                raise RuntimeError(f"{len(errors)} API request errors during {label}; inspect {output}")
            stats = grade(run, items)
            print(f"{label}: {stats['correct']}/{stats['total']} correct; depth {stats['mean_contiguous_depth']:.2f}/4", flush=True)
            return run, stats, round(time.monotonic() - started, 2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def build_comparison(reference, candidate, a, b):
    lost = sorted(k for k in a["pass"] if a["pass"][k] and not b["pass"][k])
    gained = sorted(k for k in a["pass"] if not a["pass"][k] and b["pass"][k])
    return {
        "reference_model": reference["model"],
        "candidate_model": candidate["model"],
        "dataset_sha256": reference["dataset_sha256"],
        "seed": reference["seed"],
        "reference_accuracy": a["accuracy"],
        "candidate_accuracy": b["accuracy"],
        "accuracy_delta": round(b["accuracy"] - a["accuracy"], 6),
        "reference_mean_depth": a["mean_contiguous_depth"],
        "candidate_mean_depth": b["mean_contiguous_depth"],
        "depth_delta": round(b["mean_contiguous_depth"] - a["mean_contiguous_depth"], 6),
        "retained_when_reference_correct": a["correct"] - len(lost),
        "reference_correct": a["correct"],
        "knowledge_loss_rate": round(len(lost) / a["correct"], 6) if a["correct"] else None,
        "lost": lost,
        "gained": gained,
        "reference_depths": a["depths"],
        "candidate_depths": b["depths"],
    }


def render_report(comparison, reference, candidate, elapsed):
    lines = [
        "# Q4Fit — ConceptLadder knowledge-depth report",
        "",
        f"- Benchmark dataset SHA256: `{comparison['dataset_sha256']}`",
        f"- Answer-order seed: `{comparison['seed']}`",
        f"- Reference: `{reference}`",
        f"- Quantized candidate: `{candidate}`",
        "",
        "| Metric | Reference | Quantized |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {comparison['reference_accuracy']:.1%} | {comparison['candidate_accuracy']:.1%} |",
        f"| Mean contiguous depth (0–4) | {comparison['reference_mean_depth']:.2f} | {comparison['candidate_mean_depth']:.2f} |",
        f"| Wall time including model load | {elapsed['reference']:.1f}s | {elapsed['candidate']:.1f}s |",
        "",
        f"Reference-correct questions lost: **{len(comparison['lost'])}**.",
        f"Candidate-only correct questions gained: **{len(comparison['gained'])}**.",
        f"Reference-correct answer loss rate: **{comparison['knowledge_loss_rate']:.1%}**." if comparison["knowledge_loss_rate"] is not None else "Reference-correct answer loss rate: N/A.",
        "",
        "## Depth by concept",
        "",
        "| Concept | Reference | Quantized |",
        "| --- | ---: | ---: |",
    ]
    for key, before in comparison["reference_depths"].items():
        after = comparison["candidate_depths"][key]
        lines.append(f"| `{key}` | {before}/4 | {after}/4 |")
    lines += [
        "",
        "## Lost question IDs",
        "",
        *(f"- `{item}`" for item in comparison["lost"]),
        "",
        "## Gained question IDs",
        "",
        *(f"- `{item}`" for item in comparison["gained"]),
        "",
        f"This is a {len(comparison['reference_depths']) * 4}-question multiple-choice diagnostic, not a statistically validated claim of improved model quality.",
        "Wall time includes startup and is not a token/second throughput benchmark.",
        "Questions, model settings, and source weights should be held constant for comparisons.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run bundled ConceptLadder on original and quantized GGUF weights.")
    parser.add_argument("--reference", "--original", required=True, type=Path, dest="reference")
    parser.add_argument("--candidate", "--quant", required=True, type=Path, dest="candidate")
    parser.add_argument("--llama-server", "--server", default="llama-server", dest="llama_server")
    parser.add_argument("--out", type=Path, default=Path("depth-results"))
    parser.add_argument("--suite", choices=("hard", "classic"), default="hard")
    parser.add_argument("--variants", type=int, default=4, help="Hard: independent generated variants per concept (1-32, default 4)")
    parser.add_argument("--packs", default="all", help="all or comma-separated pack IDs; use --suite classic for tcp/tls13/sqlite")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--port", type=int, default=18221)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--allow-thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an earlier report in --out")
    args = parser.parse_args(argv)
    if not (1 <= args.port <= 65535) or args.context <= 0 or args.threads <= 0 or args.ngl < 0:
        parser.error("Invalid server port, context, GPU layers, or threads")
    if args.max_tokens <= 0 or args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("Timeouts and max tokens must be positive")
    reference = args.reference.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    if reference == candidate:
        parser.error("Reference and candidate must be different files")
    for path in (reference, candidate):
        if not path.is_file():
            parser.error(f"GGUF file not found: {path}")
    server = resolve_server(args.llama_server)
    out = args.out.expanduser().resolve()
    items = load_suite(args.suite, args.packs, args.seed, args.variants)
    for name in ("reference.json", "candidate.json", "comparison.json", "report.md", "receipt.json"):
        if (out / name).exists() and not args.overwrite:
            parser.error(f"{out / name} exists; use a new --out or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "format": "q4fit-depth-v1",
        "benchmark": "conceptladder-bench/" + ("0.2.0-hard" if args.suite == "hard" else "0.1.0-classic"),
        "suite": args.suite,
        "variants": args.variants,
        "dataset_sha256": fingerprint(items),
        "items": len(items),
        "reference": {"path": str(reference), "sha256": sha256_file(reference), "bytes": reference.stat().st_size},
        "candidate": {"path": str(candidate), "sha256": sha256_file(candidate), "bytes": candidate.stat().st_size},
        "server": str(server),
        "server_sha256": sha256_file(server),
        "seed": args.seed,
        "packs": args.packs,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "gpu_layers": args.ngl,
        "threads": args.threads,
        "temperature": 0,
        "allow_thinking": args.allow_thinking,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
    }
    if receipt["reference"]["sha256"] == receipt["candidate"]["sha256"]:
        parser.error("The two model files have identical content")
    write_json(str(out / "receipt.json"), receipt)
    reference_run, a, time_ref = evaluate(args, reference, "reference", items, server, out)
    candidate_run, b, time_candidate = evaluate(args, candidate, "candidate", items, server, out)
    comparison = build_comparison(reference_run, candidate_run, a, b)
    write_json(str(out / "comparison.json"), comparison)
    (out / "report.md").write_text(render_report(comparison, reference, candidate, {"reference": time_ref, "candidate": time_candidate}), encoding="utf-8")
    print(f"Report: {out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
