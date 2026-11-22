#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import requests

SYSTEM = "Follow the requested output format exactly."
CONTAINER = "python:3.9-alpine"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


def validate_suite(tasks_path, manifest_path):
    tasks = load_jsonl(tasks_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if digest(tasks_path) != manifest["frozen_tasks_sha256"]:
        raise ValueError("Frozen task file SHA256 mismatch")
    by_id = {t["id"]: t for t in tasks}
    if len(by_id) != len(tasks) or len(tasks) != len(manifest["items"]):
        raise ValueError("Incorrect task count or duplicate ids")
    for item in manifest["items"]:
        t = by_id[item["id"]]
        if hashlib.sha256(t["prompt"].encode("utf-8")).hexdigest() != item["prompt_sha256"]:
            raise ValueError("Prompt mismatch for " + item["id"])
    return tasks, manifest


def sandbox_execute(code, task, directory, timeout=9):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = "\n".join(task["test_imports"]) + "\n" + code.rstrip() + "\n\n" + "\n".join(task["test_list"]) + "\n"
    if len(source.encode("utf-8")) > 200000:
        return {"pass": False, "status": "too_large"}
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"pass": False, "status": "syntax_error", "detail": str(exc)[:240]}
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        script = Path(tmp) / "task.py"
        script.write_text(source, encoding="utf-8")
        script.chmod(0o644)
        cmd = [
            "podman", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "32", "--memory", "256m", "--cpus", "1",
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
            "-v", str(script) + ":/work/task.py:ro", "--workdir", "/work",
            CONTAINER, "python", "-I", "-B", "/work/task.py",
        ]
        try:
            process = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
            return {
                "pass": process.returncode == 0,
                "status": "passed" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "stderr": process.stderr[-300:],
            }
        except subprocess.TimeoutExpired:
            return {"pass": False, "status": "timeout"}


def parse_number(response):
    pattern = r"(?im)^\s*FINAL\s*:\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$"
    matches = re.findall(pattern, response)
    try:
        return Decimal(matches[-1].replace(",", "")) if matches else None
    except Exception:
        return None


def parse_code(response):
    fence = re.escape(chr(96) * 3)
    m = re.search(fence + r"(?:python)?\s*\n?(.*?)\n?" + fence, response, re.I | re.S)
    return m.group(1).strip() if m else response.strip()


def start_server(server, model, logpath, port):
    cmd = [
        str(server), "-m", str(model), "-ngl", "999", "-c", "4096", "-np", "1",
        "-t", "8", "-tb", "8", "--split-mode", "none", "--main-gpu", "0",
        "--jinja", "--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}',
        "--host", "127.0.0.1", "--port", str(port),
        "--no-webui", "--no-context-shift", "--seed", "20261123",
    ]
    log = Path(logpath).open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    root = f"http://127.0.0.1:{port}"
    try:
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited early; inspect " + str(logpath))
            try:
                resp = requests.get(root + "/health", timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return proc, log, root
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.4)
        raise RuntimeError("llama-server readiness timeout")
    except Exception:
        proc.terminate()
        proc.wait(timeout=12)
        log.close()
        raise


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log.close()


def run(args):
    tasks, manifest = validate_suite(args.tasks, args.manifest)
    model = Path(args.gguf).resolve()
    folder = Path(args.workdir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": str(model), "model_sha256": digest(model),
        "suite_sha256": manifest["frozen_tasks_sha256"],
        "manifest_sha256": digest(args.manifest),
        "server": str(Path(args.server).resolve()),
        "temperature": 0, "top_k": 1, "seed": 20261123,
        "enable_thinking": False, "system_prompt": SYSTEM,
        "container": CONTAINER, "code_timeout_seconds": 9,
    }
    receipt_path = folder / (args.name + "-receipt.json")
    if receipt_path.exists() and json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Existing receipt differs; cannot mix model or benchmark")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    output_path = folder / (args.name + ".jsonl")
    done = {r["id"] for r in load_jsonl(output_path)}
    tasklist = [t for t in tasks if t["id"] not in done]
    if args.limit is not None:
        tasklist = tasklist[:args.limit]
    print("MODEL", args.name, "SHA256", receipt["model_sha256"], "REMAINING", len(tasklist), flush=True)
    if not tasklist:
        return
    proc, log, url = start_server(args.server, model, folder / (args.name + "-server.log"), args.port)
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for i, task in enumerate(tasklist, 1):
                payload = {
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}],
                    "temperature": 0, "top_k": 1, "seed": 20261123,
                    "max_tokens": 192 if task["domain"].startswith("mgsm") else 320,
                }
                response = requests.post(url + "/v1/chat/completions", json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                generated = result["choices"][0]["message"].get("content") or ""
                row = {
                    "id": task["id"], "domain": task["domain"], "response": generated,
                    "finish_reason": result["choices"][0]["finish_reason"],
                    "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                }
                if task["domain"].startswith("mgsm"):
                    answer = parse_number(generated)
                    row.update({
                        "predicted": str(answer) if answer is not None else None,
                        "correct": answer == Decimal(task["answer"]),
                        "parse_ok": answer is not None,
                    })
                else:
                    code = parse_code(generated)
                    verdict = sandbox_execute(code, task, folder / "sandbox")
                    row.update({
                        "correct": verdict["pass"], "sandbox_status": verdict["status"],
                        "sandbox_result": verdict,
                        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    })
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if i % 8 == 0 or i == len(tasklist):
                    print("PROGRESS", args.name, i, "/", len(tasklist), task["id"], "correct", row["correct"], flush=True)
    finally:
        stop_server(proc, log)


def bootstrap_diff(first, second, iterations=3000):
    items = sorted(set(first) & set(second))
    if not items:
        return None
    values = [int(first[k]["correct"]) - int(second[k]["correct"]) for k in items]
    rng = random.Random(20261123)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iterations))
    discordant = sum(v != 0 for v in values)
    minority = min(sum(v == 1 for v in values), sum(v == -1 for v in values))
    exact_p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(minority + 1)) / (2 ** discordant)) if discordant else 1.0
    return {
        "mcnemar_exact_p_two_sided": exact_p,
        "n": len(items), "delta": sum(values) / len(values),
        "bootstrap_ci_95": [means[int(.025 * iterations)], means[int(.975 * iterations)]],
        "first_only": sum(first[k]["correct"] and not second[k]["correct"] for k in items),
        "second_only": sum(second[k]["correct"] and not first[k]["correct"] for k in items),
        "both_correct": sum(first[k]["correct"] and second[k]["correct"] for k in items),
    }


def clustered_mgsm_diff(first, second, iterations=4000):
    ids = sorted({k.rsplit("-", 1)[-1] for k in first if k.startswith("mgsm-en-")})
    paired = []
    for index in ids:
        keys = ["mgsm-en-" + index, "mgsm-zh-" + index]
        if all(k in first and k in second for k in keys):
            paired.append(sum(int(first[k]["correct"]) - int(second[k]["correct"]) for k in keys))
    if not paired:
        return None
    rng = random.Random(20261123)
    n = len(paired)
    values = sorted(sum(rng.choices(paired, k=n)) / (2 * n) for _ in range(iterations))
    return {
        "question_pairs": n, "language_responses": 2 * n,
        "accuracy_delta": sum(paired) / (2 * n),
        "cluster_bootstrap_ci_95": [values[int(.025 * iterations)], values[int(.975 * iterations)]],
    }


def score(args):
    tasks, manifest = validate_suite(args.tasks, args.manifest)
    expected = {t["id"]: t for t in tasks}
    models = {}
    model_hashes = {}
    for spec in args.result:
        name, path = spec.split("=", 1)
        location = Path(path).resolve()
        receipt_path = location.parent / (name + "-receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("suite_sha256") != manifest["frozen_tasks_sha256"] or receipt.get("manifest_sha256") != digest(args.manifest):
            raise ValueError("Model receipts and task suite differ for " + name)
        rows = load_jsonl(location)
        indexed = {r["id"]: r for r in rows}
        if len(indexed) != len(rows) or not set(indexed).issubset(expected):
            raise ValueError("Duplicate or unrecognized result for " + name)
        models[name] = indexed
        model_hashes[name] = receipt["model_sha256"]
    report = {"suite_sha256": manifest["frozen_tasks_sha256"], "model_sha256": model_hashes, "models": {}, "paired": {}, "clustered_mgsm": {}}
    for name, rows in models.items():
        report["models"][name] = {}
        for domain in ("mgsm_en", "mgsm_zh", "mbpp"):
            a = [x for x in rows.values() if x["domain"] == domain]
            report["models"][name][domain] = {
                "n": len(a), "correct": sum(bool(x["correct"]) for x in a),
                "accuracy": sum(bool(x["correct"]) for x in a) / len(a) if a else None,
                "parse_fail": sum(not x.get("parse_ok", True) for x in a),
                "length_truncated": sum(x["finish_reason"] == "length" for x in a),
            }
    for a in models:
        for b in models:
            if a == b:
                continue
            report["paired"][a + "_minus_" + b] = {}
            for domain in ("mgsm_en", "mgsm_zh", "mbpp"):
                left = {k: v for k, v in models[a].items() if v["domain"] == domain}
                right = {k: v for k, v in models[b].items() if v["domain"] == domain}
                report["paired"][a + "_minus_" + b][domain] = bootstrap_diff(left, right)
            report["clustered_mgsm"][a + "_minus_" + b] = clustered_mgsm_diff(models[a], models[b])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["models"], indent=2, ensure_ascii=False))


def validate(args):
    tasks, _ = validate_suite(args.tasks, args.manifest)
    folder = Path(args.workdir).resolve()
    results = []
    for task in tasks:
        if task["domain"] == "mbpp":
            outcome = sandbox_execute(task["reference_code"], task, folder / "gold-sandbox")
            results.append({"id": task["id"], **outcome})
            print("REFERENCE", task["id"], outcome["status"], flush=True)
    (folder / "gold-reference-validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("GOLD", sum(r["pass"] for r in results), "/", len(results))


def main():
    parser = argparse.ArgumentParser(description="Frozen multi-domain downstream task evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "score", "validate"):
        p = sub.add_parser(command)
        p.add_argument("--tasks", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--workdir", required=True)
        if command == "run":
            p.add_argument("--gguf", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--server", required=True)
            p.add_argument("--port", type=int, default=18884)
            p.add_argument("--limit", type=int)
        elif command == "score":
            p.add_argument("--result", action="append", required=True)
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "score":
        score(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
