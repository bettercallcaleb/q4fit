#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from q4cache import compose, load_index
from q4fit import digest, evaluate, get_kl


def scan(index_path, corpus, reference, base_log, ppl, outdir, target, pattern, limit, ctx, chunks, ngl, keep=False):
    idx = load_index(index_path)
    corpus = Path(corpus).resolve()
    reference = Path(reference).resolve()
    base_log = Path(base_log).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    baseline = idx["banks"]["Q4_K"]
    target = target.upper()
    if target == "Q4_K" or target not in idx["banks"]:
        raise ValueError("Target requires a non-Q4 cached tensor bank")
    base_report = json.loads(base_log.with_suffix(".json").read_text(encoding="utf-8"))
    ref_report = json.loads(reference.with_suffix(".baseline.json").read_text(encoding="utf-8"))
    if base_report["model"] != baseline["path"] or base_report["reference"] != str(reference):
        raise ValueError("Base evaluation does not match cache/reference")
    if base_report["corpus_sha256"] != digest(corpus) or base_report["ctx"] != ctx or base_report["chunks"] != chunks:
        raise ValueError("Baseline evaluation settings mismatch")
    if ref_report["model_sha256"] != idx["source_sha256"] or ref_report["corpus_sha256"] != digest(corpus):
        raise ValueError("BF16 reference provenance mismatch")
    if ref_report["ctx"] != ctx or ref_report["chunks"] != chunks:
        raise ValueError("BF16 reference settings mismatch")
    original = get_kl(base_log)
    if original is None:
        raise ValueError("Missing base KLD")
    config = {
        "index_sha256": digest(index_path),
        "corpus_sha256": digest(corpus),
        "reference_sha256": digest(reference),
        "base_log_sha256": digest(base_log),
        "ppl": str(Path(ppl).resolve()),
        "target": target,
        "pattern": pattern,
        "ctx": ctx,
        "chunks": chunks,
        "ngl": ngl,
    }
    config_path = outdir / "scan-config.json"
    previous = config_path.exists()
    if previous and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing scan directory has incompatible provenance")
    if not previous:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    results_path = outdir / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    completed = {r["name"] for r in rows}
    rx = re.compile(pattern)
    matches = []
    for tensor in idx["source_tensors"]:
        name = tensor["name"]
        candidate = idx["banks"][target]["tensors"][name]
        if tensor["eligible"] and rx.search(name) and candidate["type"] == target:
            matches.append(name)
    if not matches:
        raise ValueError("No eligible tensor names match the pattern/type")
    count = 0
    for name in matches:
        if name in completed:
            continue
        if count >= limit:
            break
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", name) + "-" + hashlib.sha256(name.encode()).hexdigest()[:8]
        output = outdir / (safe + ".gguf")
        print("PROBE", name, target, flush=True)
        compose(index_path, output, changes=["^" + re.escape(name) + "$=" + target], verify_output=False, hash_output=False)
        try:
            measured = evaluate(output, corpus, ppl, reference, outdir / (safe + ".eval.log"), ctx, chunks, ngl)
            if measured["kld"] is None:
                raise RuntimeError("Missing KLD")
            old_bytes = baseline["tensors"][name]["bytes"]
            new_bytes = idx["banks"][target]["tensors"][name]["bytes"]
            row = {
                "name": name,
                "target": target,
                "base_kld": original,
                "kld": measured["kld"],
                "delta_kld": measured["kld"] - original,
                "delta_MiB": round((new_bytes - old_bytes) / 2 ** 20, 5),
                "top_percent": measured["same_top_percent"],
                "p99_kld": measured["kld_p99"],
                "model_bytes": measured["bytes"],
                "log": measured["log"],
            }
            rows.append(row)
            results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            fields = ["name", "target", "base_kld", "kld", "delta_kld", "delta_MiB", "top_percent", "p99_kld", "model_bytes", "log"]
            with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            print("RESULT", name, "delta_KLD", round(row["delta_kld"], 6), "delta_MiB", row["delta_MiB"], flush=True)
        finally:
            if not keep:
                output.unlink(missing_ok=True)
        count += 1
    print(json.dumps({"matches": len(matches), "completed": len(rows), "current_run": count, "best_measured": sorted(rows, key=lambda r: r["delta_kld"])[:5]}, indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description="Single-tensor BF16 sensitivity scan using cached GGUF tensors")
    p.add_argument("--index", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--base-log", required=True)
    p.add_argument("--ppl", default="llama-perplexity")
    p.add_argument("--workdir", required=True)
    p.add_argument("--target", choices=["Q3_K", "Q5_K", "Q6_K", "Q8_0"], required=True)
    p.add_argument("--match", default=r"\.ffn_down\.weight$")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--ngl", type=int, default=999)
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()
    scan(a.index, a.corpus, a.reference, a.base_log, a.ppl, a.workdir, a.target, a.match, a.limit, a.ctx, a.chunks, a.ngl, a.keep)


if __name__ == "__main__":
    main()
