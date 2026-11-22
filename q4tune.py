import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from q4cache import compose, load_index
from q4fit import digest, evaluate, get_kl


def config_and_context(config_path, workdir):
    path = Path(config_path).resolve()
    folder = Path(workdir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    index_path = Path(cfg["cache_index"]).resolve()
    index = load_index(index_path)
    base = index["banks"]["Q4_K"]
    if str(Path(cfg["base_model"]).resolve()) != base["path"]:
        raise ValueError("Plain Q4 base differs from indexed bank")
    if len(cfg["domains"]) < 2:
        raise ValueError("At least two tuning domains are required")
    names = [d["name"] for d in cfg["domains"]]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names):
        raise ValueError("Domain names must be unique safe identifiers")
    baselines = {}
    receipt_domains = []
    for domain in cfg["domains"]:
        corpus = Path(domain["corpus"]).resolve()
        reference = Path(domain["reference"]).resolve()
        baseline_log = Path(domain["base_log"]).resolve()
        ref_record = json.loads(reference.with_suffix(".baseline.json").read_text(encoding="utf-8"))
        base_record = json.loads(baseline_log.with_suffix(".json").read_text(encoding="utf-8"))
        if ref_record["model_sha256"] != index["source_sha256"] or ref_record["corpus_sha256"] != digest(corpus):
            raise ValueError("BF16 reference provenance mismatch for " + domain["name"])
        if ref_record["ctx"] != domain["ctx"] or ref_record["chunks"] != domain["chunks"]:
            raise ValueError("Reference context mismatch for " + domain["name"])
        if base_record["model"] != base["path"] or base_record["corpus_sha256"] != digest(corpus) or base_record["reference"] != str(reference):
            raise ValueError("Q4 baseline mismatch for " + domain["name"])
        if base_record["ctx"] != domain["ctx"] or base_record["chunks"] != domain["chunks"]:
            raise ValueError("Q4 context mismatch for " + domain["name"])
        kld = get_kl(baseline_log)
        if kld is None or kld <= 0 or base_record["same_top_percent"] is None:
            raise ValueError("Missing baseline metrics for " + domain["name"])
        baselines[domain["name"]] = {"kld": kld, "top": base_record["same_top_percent"]}
        receipt_domains.append({
            "name": domain["name"],
            "corpus_sha256": digest(corpus),
            "reference_sha256": digest(reference),
            "base_log_sha256": digest(baseline_log),
            "ctx": domain["ctx"],
            "chunks": domain["chunks"],
            "ngl": domain.get("ngl", 999),
        })
    requests = []
    for probe in cfg["probes"]:
        typ, name = probe["type"].upper(), probe["name"]
        if typ not in index["banks"] or typ == "Q4_K":
            raise ValueError("Uncached or invalid target: " + typ)
        if name not in base["tensors"] or not any(t["name"] == name and t["eligible"] for t in index["source_tensors"]):
            raise ValueError("Unquantizable tensor " + name)
        if index["banks"][typ]["tensors"][name]["type"] != typ:
            raise ValueError("Bank does not quantize " + name + " to " + typ)
        requests.append({"name": name, "type": typ})
    if len({(r["name"], r["type"]) for r in requests}) != len(requests):
        raise ValueError("Duplicate tensor probe")
    signature = {
        "index_sha256": digest(index_path),
        "source_sha256": index["source_sha256"],
        "base_bank_sha256": base["sha256"],
        "domains": receipt_domains,
        "probes": requests,
        "ppl": str(Path(cfg["ppl"]).resolve()),
        "target_gib": cfg["target_gib"],
        "max_top_drop_pp": cfg.get("max_top_drop_pp", 0.5),
    }
    receipt_path = folder / "run-config.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Work directory contains an incompatible run")
    else:
        receipt_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return cfg, index_path, index, folder, baselines, requests, signature


def write_rows(folder, rows):
    p = folder / "trials.json"
    tmp = folder / "trials.json.tmp"
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    with (folder / "trials.csv").open("w", newline="", encoding="utf-8") as output:
        cols = ["name", "type", "delta_bytes", "delta_MiB", "worst_delta_ratio"]
        writer = csv.DictWriter(output, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in cols})


def scan(config_path, workdir, limit=20, keep=False):
    cfg, index_path, index, folder, baselines, requests, signature = config_and_context(config_path, workdir)
    rows_path = folder / "trials.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    complete = {(r["name"], r["type"]) for r in rows}
    new = 0
    for request in requests:
        name, typ = request["name"], request["type"]
        if (name, typ) in complete:
            continue
        if new >= limit:
            break
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name) + "-" + typ.lower()
        candidate = folder / (slug + ".gguf")
        print("SCAN", name, typ, flush=True)
        compose(index_path, candidate, changes=["^" + re.escape(name) + "$=" + typ], verify_output=False, hash_output=False)
        try:
            observed = {}
            for domain in cfg["domains"]:
                dname = domain["name"]
                result = evaluate(candidate, domain["corpus"], cfg["ppl"], domain["reference"], folder / (slug + "-" + dname + ".eval.log"), domain["ctx"], domain["chunks"], domain.get("ngl", 999))
                if result["kld"] is None or result["same_top_percent"] is None:
                    raise RuntimeError("Failed to parse tensor probe metrics")
                observed[dname] = {"kld": result["kld"], "top": result["same_top_percent"], "p99": result["kld_p99"]}
            old_size = index["banks"]["Q4_K"]["tensors"][name]["bytes"]
            new_size = index["banks"][typ]["tensors"][name]["bytes"]
            rel = {d: (v["kld"] - baselines[d]["kld"]) / baselines[d]["kld"] for d, v in observed.items()}
            r = {
                "name": name,
                "type": typ,
                "delta_bytes": new_size - old_size,
                "delta_MiB": round((new_size - old_size) / 2 ** 20, 4),
                "worst_delta_ratio": max(rel.values()),
                "relative_kld": rel,
                "observed": observed,
            }
            rows.append(r)
            write_rows(folder, rows)
            complete.add((name, typ))
            new += 1
            print("SCORE", name, typ, "size_MiB", r["delta_MiB"], "relative_KL", {k: round(v, 5) for k, v in rel.items()}, flush=True)
        finally:
            if not keep:
                candidate.unlink(missing_ok=True)
    print("SCAN_COMPLETE", len(rows), "/", len(requests), "new", new, flush=True)
    return rows


def objective(state, domains):
    _, ratios, edits = state
    worst = max(ratios)
    average = sum(ratios) / len(domains)
    return worst, average, len(edits)


def beam_search(rows, baselines, base_bytes, budget, beam_width=2500, bucket_width=1024 * 1024, max_edits=8, max_candidates=12):
    domains = list(baselines)
    groups = defaultdict(list)
    for row in rows:
        groups[row["name"]].append(row)
    order = sorted(groups, key=lambda name: min(row["delta_bytes"] for row in groups[name]))
    states = [(0, tuple([0.0] * len(domains)), ())]
    max_positive = sum(max(0, max(r["delta_bytes"] for r in groups[name])) for name in order)
    for name in order:
        alternatives = groups[name]
        next_states = []
        for delta, ratios, edits in states:
            next_states.append((delta, ratios, edits))
            if len(edits) >= max_edits:
                continue
            for row in alternatives:
                byte_delta = delta + row["delta_bytes"]
                if base_bytes + byte_delta > budget + max_positive:
                    continue
                shifted = tuple(ratios[i] + row["relative_kld"][domain] for i, domain in enumerate(domains))
                next_states.append((byte_delta, shifted, edits + ((row["name"], row["type"]),)))
        buckets = defaultdict(list)
        for state in next_states:
            bucket = math.floor(state[0] / bucket_width)
            buckets[bucket].append(state)
        representatives = []
        extras = []
        cap_per_bucket = max(4, beam_width // max(1, len(buckets)))
        for entries in buckets.values():
            entries.sort(key=lambda s: (objective(s, domains), s[0]))
            representatives.append(entries[0])
            extras.extend(entries[1:cap_per_bucket])
        representatives.sort(key=lambda s: (objective(s, domains), s[0]))
        extras.sort(key=lambda s: (objective(s, domains), s[0]))
        states = (representatives + extras[:max(0, beam_width - len(representatives))])[:beam_width]
    feasible = [s for s in states if base_bytes + s[0] <= budget and s[2]]
    feasible.sort(key=lambda s: (objective(s, domains), s[0]))
    result = []
    seen = set()
    for delta, ratios, edits in feasible:
        if edits in seen:
            continue
        seen.add(edits)
        result.append({
            "edits": list(edits),
            "predicted_bytes": base_bytes + delta,
            "predicted_relative_kld": dict(zip(domains, ratios)),
            "predicted_worst_ratio": 1 + max(ratios),
            "predicted_mean_ratio": 1 + sum(ratios) / len(ratios),
        })
        if len(result) >= max_candidates:
            break
    return result


def q5_anchors(rows, baselines, base_bytes, budget, limit=2, max_edits=8):
    domains = list(baselines)
    positive = [r for r in rows if r["type"] == "Q5_K"]
    candidates = []
    for count in range(1, min(2, max_edits) + 1):
        for combo in itertools.combinations(positive, count):
            size = base_bytes + sum(r["delta_bytes"] for r in combo)
            if size > budget:
                continue
            rel = {d: sum(r["relative_kld"][d] for r in combo) for d in domains}
            if max(rel.values()) >= 0:
                continue
            candidates.append({
                "edits": [(r["name"], r["type"]) for r in combo],
                "predicted_bytes": size,
                "predicted_relative_kld": rel,
                "predicted_worst_ratio": 1 + max(rel.values()),
                "predicted_mean_ratio": 1 + sum(rel.values()) / len(domains),
            })
    candidates.sort(key=lambda c: (c["predicted_worst_ratio"], c["predicted_mean_ratio"], c["predicted_bytes"]))
    return candidates[:limit]


def optimize(config_path, workdir, beam_width=2500, max_edits=8, verify_top=4):
    cfg, index_path, index, folder, baselines, requests, signature = config_and_context(config_path, workdir)
    rows_file = folder / "trials.json"
    if not rows_file.exists():
        raise FileNotFoundError("No measured tensor probes")
    rows = json.loads(rows_file.read_text(encoding="utf-8"))
    if {(r["name"], r["type"]) for r in rows} - {(x["name"], x["type"]) for x in requests}:
        raise ValueError("Trials contain unrequested measurements")
    base_bytes = index["banks"]["Q4_K"]["bytes"]
    budget = int(cfg["target_gib"] * (2 ** 30))
    ranked = beam_search(rows, baselines, base_bytes, budget, beam_width, max_edits=max_edits, max_candidates=max(verify_top * 12, 12))
    result_path = folder / "optimization.json"
    report = {
        "index_sha256": digest(index_path),
        "source_sha256": index["source_sha256"],
        "trial_count": len(rows),
        "target_bytes": budget,
        "baseline_bytes": base_bytes,
        "baselines": baselines,
        "ranked_predictions": ranked,
        "verified": [],
    }
    anchors = q5_anchors(rows, baselines, base_bytes, budget, limit=min(2, verify_top), max_edits=max_edits)
    shortlist = ranked[:max(0, verify_top - len(anchors))] + anchors + ranked
    visited = []
    for candidate in shortlist:
        if len(visited) >= verify_top:
            break
        if candidate["predicted_worst_ratio"] >= 1.0:
            continue
        edits = candidate["edits"]
        signature_edits = {tuple(x) for x in edits}
        if any(len(signature_edits ^ previous) <= 1 for previous in visited):
            continue
        visited.append(signature_edits)
        rules = ["^" + re.escape(name) + "$=" + typ for name, typ in edits]
        target = folder / ("verified-" + str(len(visited)) + ".gguf")
        assembled = compose(index_path, target, changes=rules, verify_banks=True, verify_output=True, hash_output=True)
        observed = {}
        for domain in cfg["domains"]:
            label = domain["name"]
            measured = evaluate(target, domain["corpus"], cfg["ppl"], domain["reference"], folder / (target.stem + "-" + label + ".eval.log"), domain["ctx"], domain["chunks"], domain.get("ngl", 999))
            observed[label] = {"kld": measured["kld"], "ratio": measured["kld"] / baselines[label]["kld"], "top": measured["same_top_percent"], "top_delta_pp": measured["same_top_percent"] - baselines[label]["top"]}
        passed = target.stat().st_size <= budget and all(v["ratio"] < 1 and v["top_delta_pp"] >= -cfg.get("max_top_drop_pp", 0.5) for v in observed.values())
        entry = {
            **candidate,
            "observed": observed,
            "actual_bytes": target.stat().st_size,
            "gguf_sha256": assembled["output_sha256"],
            "passes_tuning": passed,
            "path": str(target),
        }
        report["verified"].append(entry)
        result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("COMBINED", len(visited), "passes", passed, "pred", round(candidate["predicted_worst_ratio"], 5), "actual", {k: round(v["ratio"], 5) for k, v in observed.items()}, flush=True)
        if not passed:
            target.unlink()
            entry["path"] = None
            result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    successes = [x for x in report["verified"] if x["passes_tuning"]]
    if successes:
        winner = min(successes, key=lambda x: (max(d["ratio"] for d in x["observed"].values()), sum(d["ratio"] for d in x["observed"].values()) / len(x["observed"]), x["actual_bytes"]))
        report["selected"] = {"edits": winner["edits"], "observed": winner["observed"], "actual_bytes": winner["actual_bytes"], "gguf_sha256": winner["gguf_sha256"], "path": winner["path"]}
        for entry in successes:
            if entry is not winner:
                Path(entry["path"]).unlink(missing_ok=True)
                entry["path"] = None
    report["status"] = "passes_tuning_domains" if successes else "no_validated_improvement"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OPTIMIZATION", report["status"], "trials", len(rows), "verified", len(report["verified"]), "selected", report.get("selected", {}).get("edits"), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description="Budgeted multi-domain per-tensor GGUF quantization search")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("scan", "optimize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--workdir", required=True)
        if name == "scan":
            cmd.add_argument("--limit", type=int, default=20)
            cmd.add_argument("--keep", action="store_true")
        else:
            cmd.add_argument("--beam-width", type=int, default=2500)
            cmd.add_argument("--max-edits", type=int, default=8)
            cmd.add_argument("--verify-top", type=int, default=4)
    args = p.parse_args()
    if args.command == "scan":
        scan(args.config, args.workdir, args.limit, args.keep)
    else:
        optimize(args.config, args.workdir, args.beam_width, args.max_edits, args.verify_top)


if __name__ == "__main__":
    main()
