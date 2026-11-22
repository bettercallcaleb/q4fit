#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path

SEED = 2026112307
N_MGSM = 24
N_MBPP = 16


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(data_dir, workdir, manifest_path):
    data_dir, workdir, manifest_path = Path(data_dir).resolve(), Path(workdir).resolve(), Path(manifest_path).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {name: data_dir / (name + ".jsonl") for name in ("mgsm_en", "mgsm_zh", "mbpp")}
    sources = {name: read_jsonl(path) for name, path in raw.items()}
    english, chinese, code = sources["mgsm_en"], sources["mgsm_zh"], sources["mbpp"]
    if len(english) != 250 or len(chinese) != 250 or len(code) != 257:
        raise ValueError("Unexpected dataset sizes")
    for i, (en, zh) in enumerate(zip(english, chinese)):
        if en["answer_number"] != zh["answer_number"]:
            raise ValueError(f"MGSM parallel row {i} differs")
    rng = random.Random(SEED)
    earlier = [json.loads(Path(path).read_text(encoding="utf-8")) for path in ["benchmarks/QWEN3_4B_TASK_MANIFEST.json", "benchmarks/QWEN3_4B_THINKING_MANIFEST.json"]]
    excluded_math = {item["source_index"] for manifest in earlier for item in manifest["items"] if item["domain"] == "mgsm_en"}
    excluded_code = {item["source_task_id"] for manifest in earlier for item in manifest["items"] if item["domain"] == "mbpp"}
    math_indices = sorted(rng.sample([i for i in range(len(english)) if i not in excluded_math], N_MGSM))
    code_indices = sorted(rng.sample([i for i in range(len(code)) if code[i]["task_id"] not in excluded_code], N_MBPP))
    tasks = []
    manifest_items = []
    for language, items in (("en", english), ("zh", chinese)):
        for idx in math_indices:
            row = items[idx]
            task_id = f"mgsm-{language}-{idx:03}"
            q = row["question"].strip()
            if language == "en":
                prompt = "Solve this math problem. Give the numeric result only; your final response must be exactly one line in the form FINAL: <integer>.\n\n" + q
            else:
                prompt = "请解决这道数学应用题。只给出数字结果，最后输出必须严格为一行：FINAL: <整数>。\n\n" + q
            t = {"id": task_id, "domain": "mgsm_" + language, "prompt": prompt, "answer": int(row["answer_number"])}
            tasks.append(t)
            manifest_items.append({"id": task_id, "domain": t["domain"], "source_index": idx, "prompt_sha256": digest(prompt.encode("utf-8"))})
    for idx in code_indices:
        row = code[idx]
        task_id = f"mbpp-{row['task_id']:03}"
        function = next((node for node in ast.parse(row["code"]).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if function is None:
            raise ValueError("MBPP reference has no top-level function")
        signature = ast.unparse(function).splitlines()[0]
        prompt = "Implement this exact Python 3 function signature, keeping its name and argument order:\n" + signature + "\n\nTask: " + row["prompt"].strip() + "\n\nReturn only executable Python source code without explanations or Markdown fences. Do not invoke external programs or the network."
        t = {"id": task_id, "domain": "mbpp", "prompt": prompt, "required_signature": signature, "test_imports": row["test_imports"], "test_list": row["test_list"], "reference_code": row["code"]}
        tasks.append(t)
        manifest_items.append({"id": task_id, "domain": "mbpp", "source_task_id": row["task_id"], "required_signature": signature, "prompt_sha256": digest(prompt.encode("utf-8")), "test_sha256": digest(json.dumps(row["test_list"], sort_keys=True).encode())})
    task_path = workdir / "frozen_tasks.jsonl"
    tmp_path = workdir / "frozen_tasks.jsonl.tmp"
    write_jsonl(tmp_path, tasks)
    frozen_hash = sha(tmp_path)
    data = {
        "suite": "phi4-mini-independent-task-pilot-v0.7",
        "exclude_prior_manifests": [x["frozen_tasks_sha256"] for x in earlier],
        "excluded_math_indices": sorted(excluded_math),
        "excluded_mbpp_task_ids": sorted(excluded_code),
        "protocol": "Separate Phi-4-mini end-task split, no overlap with v0.5 or v0.6 Qwen3 samples",
        "seed": SEED,
        "sampling": "random.Random(2026112307) draws, excluding Qwen3-4B v0.5 and v0.6 task indices",
        "size": {"mgsm_en": N_MGSM, "mgsm_zh": N_MGSM, "mbpp": N_MBPP},
        "sources": {
            "mgsm_en": {"repo": "juletxara/mgsm", "config": "en", "split": "test", "rows": 250, "full_source_jsonl_sha256": sha(raw["mgsm_en"])},
            "mgsm_zh": {"repo": "juletxara/mgsm", "config": "zh", "split": "test", "rows": 250, "full_source_jsonl_sha256": sha(raw["mgsm_zh"])},
            "mbpp": {"repo": "google-research-datasets/mbpp", "config": "sanitized", "split": "test", "rows": 257, "full_source_jsonl_sha256": sha(raw["mbpp"])},
        },
        "frozen_tasks_sha256": frozen_hash,
        "decode": {"temperature": 0, "top_k": 1, "seed": 2026112307, "enable_thinking": False, "math_max_tokens": 256, "code_max_tokens": 400},
        "items": manifest_items,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != data:
        tmp_path.unlink()
        raise ValueError("Manifest differs; frozen suite cannot change")
    if task_path.exists() and sha(task_path) != frozen_hash:
        tmp_path.unlink()
        raise ValueError("Frozen task file differs; refuse to overwrite")
    tmp_path.replace(task_path)
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_task_sha256": data["frozen_tasks_sha256"], "n": len(tasks), "math_indices": math_indices, "code_ids": [code[i]["task_id"] for i in code_indices]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    prepare(a.data_dir, a.workdir, a.manifest)


if __name__ == "__main__":
    main()
