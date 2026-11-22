#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

MODELS = ("bf16", "q4", "q4fit", "q4ks")
DOMAINS = ("mgsm_en", "mgsm_zh", "mbpp")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    a = parser.parse_args()
    root = Path(a.root).resolve()
    reportdir = root / "benchmarks"
    work = root / "work/phi4mini"
    taskdir = work / "tasks"
    manifest = load(reportdir / "PHI4MINI_TASK_MANIFEST.json")
    plan = load(reportdir / "PHI4MINI_SEARCH_PLAN.json")
    selection = load(reportdir / "PHI4MINI_SELECTED.json")
    results = load(taskdir / "task-summary.json")
    direct = load(work / "direct-verification.json")
    bank = load(work / "cache/index.json")
    expected = {
        "bf16": bank["source_sha256"],
        "q4": bank["banks"]["Q4_K"]["sha256"],
        "q4fit": selection["selected"]["gguf_sha256"],
        "q4ks": digest(work / "standard-q4-k-s.gguf"),
    }
    ids = [item["id"] for item in manifest["items"]]
    records = {}
    for model in MODELS:
        receipt = load(taskdir / (model + "-receipt.json"))
        rows = load_jsonl(taskdir / (model + ".jsonl"))
        by_id = {r["id"]: r for r in rows}
        assert len(rows) == len(ids) and set(by_id) == set(ids)
        assert receipt["suite_sha256"] == manifest["frozen_tasks_sha256"]
        assert receipt["model_sha256"] == expected[model]
        assert not receipt["enable_thinking"]
        records[model] = by_id
    assert results["suite_sha256"] == manifest["frozen_tasks_sha256"]
    assert selection["search_plan_sha256"] == digest(reportdir / "PHI4MINI_SEARCH_PLAN.json")
    assert selection["frozen_task_suite_sha256"] == manifest["frozen_tasks_sha256"]
    assert direct["byte_identical"] and direct["direct_quant_sha256"] == expected["q4fit"]
    observations = {}
    for model in ("q4", "q4fit", "q4ks"):
        observations[model] = {}
        for name in ("wiki_holdout", "code_holdout"):
            x = load(work / "holdout" / (model + "-" + name + ".json"))
            observations[model][name] = {
                "kld": x["kld"], "p99_kld": x["kld_p99"], "bf16_top_percent": x["same_top_percent"],
                "bytes": x["bytes"], "corpus_sha256": x["corpus_sha256"],
            }
    for name in ("wiki_holdout", "code_holdout"):
        checksums = {observations[m][name]["corpus_sha256"] for m in observations}
        assert len(checksums) == 1
        assert checksums.pop() == next(c["corpus_sha256"] for c in plan["holdout_domains"] if c["name"] == name)
    report = {
        "source": "microsoft/Phi-4-mini-instruct",
        "source_revision": plan["source_hf_revision"],
        "source_gguf_sha256": expected["bf16"],
        "sample_sha256": manifest["frozen_tasks_sha256"],
        "source_models_sha256": expected,
        "budget_bytes": plan["budget_bytes"],
        "selected_bytes": selection["selected"]["actual_bytes"],
        "selected_sha256": expected["q4fit"],
        "direct_quant_byte_identical": direct["byte_identical"],
        "holdout": observations,
        "task_scores": results,
    }
    (reportdir / "PHI4MINI_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (reportdir / "PHI4MINI_ITEM_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["task_id", "domain", *(m + "_correct" for m in MODELS), "q4fit_q4_same_response", *(m + "_truncated" for m in MODELS)])
        for item in manifest["items"]:
            task_id = item["id"]
            writer.writerow([
                task_id, item["domain"],
                *(int(records[m][task_id]["correct"]) for m in MODELS),
                int(records["q4fit"][task_id]["response"] == records["q4"][task_id]["response"]),
                *(int(records[m][task_id]["finish_reason"] == "length") for m in MODELS),
            ])
    print("PUBLISHED_DATA")
    for model in MODELS:
        print(model, [(d, results["models"][model][d]["correct"], results["models"][model][d]["n"]) for d in DOMAINS])
    for model in ("q4", "q4fit", "q4ks"):
        print(model, [(k, observations[model][k]["kld"]) for k in observations[model]])


if __name__ == "__main__":
    main()
