import argparse
import itertools
import json
import re
import shutil
from pathlib import Path

from q4fit import BLOCKS, GROUPS, build, digest, evaluate, get_kl, inventory, quant_bytes

def pattern_for(name):
    if "@" not in name:
        if name not in GROUPS:
            raise ValueError("Unknown tensor family: " + name)
        return GROUPS[name]
    family, span = name.split("@", 1)
    if family not in GROUPS or family in ("embedding", "output", "other"):
        raise ValueError("Invalid layered family: " + family)
    match = re.fullmatch(r"(\d+):(\d+)", span)
    if not match:
        raise ValueError("Layer span must be FAMILY@FIRST:LAST")
    first, last = map(int, match.groups())
    if first > last or last > 512:
        raise ValueError("Invalid layer span")
    suffix = GROUPS[family].removeprefix(r"\.")
    return r"^blk\.(?:" + "|".join(map(str, range(first, last + 1))) + r")\." + suffix

def select_recipe(groups, trials, bases, base_bytes, budget):
    options = []
    for group in groups:
        options.append([{"group": group, "type": "Q4_K", "delta_bytes": 0, "deltas": {d: 0.0 for d in bases}}]
                       + [t for t in trials if t["group"] == group])
    chosen = None
    for combo in itertools.product(*options):
        size = base_bytes + sum(t["delta_bytes"] for t in combo)
        if size > budget:
            continue
        scores = {d: (bases[d] + sum(t["deltas"][d] for t in combo)) / bases[d] for d in bases}
        worst = max(scores.values())
        average = sum(scores.values()) / len(scores)
        key = (worst, average, size)
        if chosen is None or key < chosen[0]:
            chosen = (key, list(combo), scores)
    return chosen

def passes_tuning(observed, base_top, budget_ok, max_top_drop_pp=0.5):
    return budget_ok and all(x["kld_ratio"] < 1.0 and (x["same_top_percent"] is None or base_top[name] is None or x["same_top_percent"] >= base_top[name] - max_top_drop_pp) for name, x in observed.items())

def run(config_path, workdir, max_trials=8, keep_trials=False):
    config_path, workdir = Path(config_path).resolve(), Path(workdir).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    workdir.mkdir(parents=True, exist_ok=True)
    source, baseline = Path(cfg["model"]).resolve(), Path(cfg["base_model"]).resolve()
    budget = int(float(cfg["target_gib"]) * 2 ** 30)
    if not source.is_file() or not baseline.is_file():
        raise FileNotFoundError("BF16 source and Q4 base are required")
    ts = inventory(source)
    groups = cfg["groups"]
    types = cfg["types"]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("Groups must be unique and nonempty")
    group_types = cfg.get("group_types", {})
    if not types or any(typ not in ("Q3_K", "Q5_K", "Q6_K", "Q8_0") for typ in types):
        raise ValueError("Unsupported search quant type")
    if any(g not in groups or not choices or any(t not in ("Q3_K", "Q5_K", "Q6_K", "Q8_0") for t in choices) for g, choices in group_types.items()):
        raise ValueError("Invalid group_types configuration")
    group_regex = {g: pattern_for(g) for g in groups}
    matches = {}
    for name, pattern in group_regex.items():
        matches[name] = [t for t in ts if t["eligible"] and re.search(pattern, t["name"])]
        if not matches[name]:
            raise ValueError("No eligible tensors in group " + name)
    names = [t["name"] for rows in matches.values() for t in rows]
    if len(names) != len(set(names)):
        raise ValueError("Overlapping groups are not supported")
    original = json.loads(baseline.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    sha_source = digest(source)
    if original["source_sha256"] != sha_source or original["default"] != "Q4_K" or original["overrides"]:
        raise ValueError("Plain Q4 baseline must be generated from this BF16 file")
    domains = cfg["domains"]
    if len(domains) < 2 or len({d["name"] for d in domains}) != len(domains):
        raise ValueError("Need at least two distinct measured domains")
    base_scores = {}
    base_top = {}
    for d in domains:
        manifest = json.loads(Path(d["reference"]).with_suffix(".baseline.json").read_text(encoding="utf-8"))
        measured = json.loads(Path(d["base_log"]).with_suffix(".json").read_text(encoding="utf-8"))
        if manifest["model_sha256"] != sha_source:
            raise ValueError("Incorrect BF16 source for " + d["name"])
        if manifest["corpus_sha256"] != digest(d["corpus"]) or manifest["ctx"] != d["ctx"] or manifest["chunks"] != d["chunks"]:
            raise ValueError("Incorrect reference settings for " + d["name"])
        if measured["model"] != str(baseline) or measured["corpus_sha256"] != manifest["corpus_sha256"] or measured["reference"] != str(Path(d["reference"]).resolve()):
            raise ValueError("Incorrect Q4 evaluation for " + d["name"])
        score = get_kl(d["base_log"])
        if score is None or score <= 0:
            raise ValueError("Invalid Q4 KLD on " + d["name"])
        base_scores[d["name"]] = score
        base_top[d["name"]] = measured.get("same_top_percent")
    metadata = {
        "source_sha": sha_source,
        "baseline_sha": digest(baseline),
        "groups": groups,
        "types": types,
        "group_types": group_types,
        "target_gib": cfg["target_gib"],
        "max_top_drop_pp": cfg.get("max_top_drop_pp", 0.5),
        "imatrix_sha": digest(cfg["imatrix"]) if cfg.get("imatrix") else None,
        "domains": [{"name": d["name"], "corpus_sha": digest(d["corpus"]), "reference_sha": digest(d["reference"]), "base_log_sha": digest(d["base_log"]), "ctx": d["ctx"], "chunks": d["chunks"]} for d in domains],
        "quantizer": str(Path(cfg["quantizer"]).resolve()),
        "ppl": str(Path(cfg["ppl"]).resolve()),
    }
    receipt = workdir / "run-config.json"
    cache = workdir / "trials.json"
    if cache.exists():
        if not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cached trials have incompatible provenance")
        trials = json.loads(cache.read_text(encoding="utf-8"))
    else:
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing directory has incompatible provenance")
        receipt.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        trials = []
    count = 0
    for group, typ in ((g, typ) for g in groups for typ in group_types.get(g, types)):
        if any(t["group"] == group and t["type"] == typ for t in trials):
            print("REUSE", group, typ, flush=True)
            continue
        if count >= max_trials:
            break
        candidate = workdir / (group.replace("@", "-").replace(":", "to") + "-" + typ.lower() + ".gguf")
        delta = sum(quant_bytes(t, typ) - quant_bytes(t, "Q4_K") for t in matches[group])
        if shutil.disk_usage(workdir).free < baseline.stat().st_size + 4 * 2 ** 30:
            raise RuntimeError("Not enough free disk for a disposable candidate")
        print("TRIAL", group, typ, "delta_MiB", round(delta / 2 ** 20, 1), flush=True)
        build(source, candidate, cfg["quantizer"], "Q4_K", [group_regex[group] + "=" + typ], cfg.get("imatrix"), cfg.get("threads", 12))
        try:
            scores = {}
            for d in domains:
                out = workdir / (candidate.stem + "-" + d["name"] + ".eval.log")
                measured = evaluate(candidate, d["corpus"], cfg["ppl"], d["reference"], out, d["ctx"], d["chunks"], d.get("ngl", 999))
                if measured["kld"] is None:
                    raise RuntimeError("Missing KLD metric")
                scores[d["name"]] = {"kld": measured["kld"], "same_top_percent": measured["same_top_percent"], "kld_p99": measured["kld_p99"]}
            entry = {"group": group, "type": typ, "delta_bytes": delta, "actual_bytes": candidate.stat().st_size, "scores": scores, "deltas": {name: metrics["kld"] - base_scores[name] for name, metrics in scores.items()}}
            trials.append(entry)
            cache.write_text(json.dumps(trials, indent=2), encoding="utf-8")
        finally:
            if candidate.exists() and not keep_trials:
                candidate.unlink()
        count += 1
    selected = select_recipe(groups, trials, base_scores, baseline.stat().st_size, budget)
    result = {
        "source_sha": sha_source, "budget_bytes": budget, "baseline_bytes": baseline.stat().st_size,
        "base_kld": base_scores, "base_top_percent": base_top, "trial_count": len(trials), "trials": trials,
    }
    if selected is None:
        result["status"] = "no_feasible_recipe"
    else:
        (estimated_worst, estimated_mean, estimated_bytes), items, predicted = selected
        changes = [group_regex[t["group"]] + "=" + t["type"] for t in items if t["type"] != "Q4_K"]
        result.update(estimate={"worst_ratio": estimated_worst, "mean_ratio": estimated_mean, "bytes": estimated_bytes, "by_domain": predicted}, chosen=items)
        if changes:
            winner = workdir / "best.gguf"
            build(source, winner, cfg["quantizer"], "Q4_K", changes, cfg.get("imatrix"), cfg.get("threads", 12))
            observed = {}
            for d in domains:
                out = workdir / ("best-" + d["name"] + ".eval.log")
                v = evaluate(winner, d["corpus"], cfg["ppl"], d["reference"], out, d["ctx"], d["chunks"], d.get("ngl", 999))
                observed[d["name"]] = {"kld": v["kld"], "kld_ratio": v["kld"] / base_scores[d["name"]], "same_top_percent": v["same_top_percent"], "kld_p99": v["kld_p99"]}
            good = passes_tuning(observed, base_top, winner.stat().st_size <= budget, cfg.get("max_top_drop_pp", 0.5))
            result.update(status="passes_tuning_domains" if good else "fails_tuning_domains", winner_bytes=winner.stat().st_size, observed=observed)
        else:
            result.update(status="plain_q4_selected", observed={k: {"kld": v, "kld_ratio": 1.0} for k, v in base_scores.items()})
    (workdir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "chosen": result.get("chosen"), "estimate": result.get("estimate"), "observed": result.get("observed")}, indent=2), flush=True)
    return result

def main():
    p = argparse.ArgumentParser(description="Multi-domain BF16-referenced, size-constrained GGUF tensor search")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--max-trials", type=int, default=8)
    p.add_argument("--keep-trials", action="store_true")
    a = p.parse_args()
    run(a.config, a.workdir, a.max_trials, a.keep_trials)

if __name__ == "__main__":
    main()
