#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    work, out = Path(a.workdir).resolve(), Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids = [row["id"] for row in manifest["items"]]
    labels = ("bf16", "q4", "q4fit", "q4ks")
    expected_model_hashes = {
        "bf16": "c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4",
        "q4": "f141c24ecd34e250e87a80b64d7f22d9be0073547f2ef447082c0ecac2f32725",
        "q4fit": "7292a820225eec6af4e7e544fc0d5de8ac31cbe4486082e20832521cdc935450",
        "q4ks": "b12aba3ec0b611f4b3149dfde087347fe08c4d1e6b411279d7d05c3e20273973",
    }
    rows = {}
    receipts = {}
    for label in labels:
        raw = read_jsonl(work / (label + ".jsonl"))
        receipt = json.loads((work / (label + "-receipt.json")).read_text(encoding="utf-8"))
        if len(raw) != len(ids) or set(x["id"] for x in raw) != set(ids):
            raise ValueError(label + " has missing or duplicate tasks")
        if receipt["suite_sha256"] != manifest["frozen_tasks_sha256"] or not receipt["enable_thinking"] or receipt["model_sha256"] != expected_model_hashes[label]:
            raise ValueError("Benchmark mismatch, wrong model weights, or thinking not enabled: " + label)
        rows[label] = {r["id"]: r for r in raw}
        receipts[label] = receipt
    path = out / "QWEN3_4B_THINKING_ITEM_RESULTS.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["task_id", "domain", "bf16_correct", "q4_correct", "q4fit_correct", "q4ks_correct", "q4fit_q4_response_identical", "bf16_truncated", "q4_truncated", "q4fit_truncated", "q4ks_truncated"])
        for item in manifest["items"]:
            task = item["id"]
            writer.writerow([
                task, item["domain"], *(int(rows[label][task]["correct"]) for label in labels),
                int(rows["q4fit"][task]["response"] == rows["q4"][task]["response"]),
                *(int(rows[label][task]["finish_reason"] == "length") for label in labels),
            ])
    stats = {}
    for label in labels:
        group = list(rows[label].values())
        stats[label] = {
            "n": len(group),
            "completion_tokens_mean": statistics.mean(r["completion_tokens"] for r in group),
            "completion_tokens_median": statistics.median(r["completion_tokens"] for r in group),
            "reasoning_present_count": sum(r["reasoning_present"] for r in group),
            "reasoning_characters_mean": statistics.mean(r["reasoning_characters"] for r in group),
            "length_truncated_count": sum(r["finish_reason"] == "length" for r in group),
            "math_parse_failures": sum(r["domain"].startswith("mgsm") and not r["parse_ok"] for r in group if "parse_ok" in r),
            "code_sandbox_passes": sum(r["domain"] == "mbpp" and r["correct"] for r in group),
            "model_sha256": receipts[label]["model_sha256"],
        }
    similarity = {}
    for domain in ("mgsm_en", "mgsm_zh", "mbpp"):
        group = [i for i in ids if rows["q4fit"][i]["domain"] == domain]
        similarity[domain] = {
            "q4fit_vs_q4_identical_responses": sum(rows["q4fit"][i]["response"] == rows["q4"][i]["response"] for i in group),
            "q4fit_vs_q4_identical_correctness": sum(rows["q4fit"][i]["correct"] == rows["q4"][i]["correct"] for i in group),
            "sample_count": len(group),
        }
    summary = json.loads((work / "task-summary.json").read_text(encoding="utf-8"))
    if summary["suite_sha256"] != manifest["frozen_tasks_sha256"]:
        raise ValueError("Summary checksum mismatch")
    alignment = {}
    indices = sorted({item["source_index"] for item in manifest["items"] if item["domain"] == "mgsm_en"})
    for label in labels:
        pairs = [(bool(rows[label][f"mgsm-en-{index:03}"]["correct"]), bool(rows[label][f"mgsm-zh-{index:03}"]["correct"])) for index in indices]
        alignment[label] = {
            "both_correct": sum(en and zh for en, zh in pairs),
            "english_only_correct": sum(en and not zh for en, zh in pairs),
            "chinese_only_correct": sum(zh and not en for en, zh in pairs),
            "both_incorrect": sum(not en and not zh for en, zh in pairs),
        }
    summary["bounded_thinking_diagnostics"] = stats
    summary["outcome_similarity"] = similarity
    summary["bilingual_question_alignment"] = alignment
    destination = out / "QWEN3_4B_THINKING_RESULTS.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite": manifest["frozen_tasks_sha256"], "stats": stats, "similarity": similarity}, ensure_ascii=False, indent=2))
    print("EXPORTED", destination, path)


if __name__ == "__main__":
    main()
