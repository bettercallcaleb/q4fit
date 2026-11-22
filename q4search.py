import itertools
import json
import re
import shutil
from pathlib import Path

from q4fit import GROUPS, build, digest, evaluate, get_kl, inventory, quant_bytes

def choose(groups, results, base_bytes, budget):
    options = []
    for g in groups:
        cases = [dict(group=g, typ="Q4_K", delta_bytes=0, delta_kld=0.0)]
        cases.extend(x for x in results if x["group"] == g and x["kld"] is not None)
        options.append(cases)
    best = None
    for combination in itertools.product(*options):
        size = base_bytes + sum(x["delta_bytes"] for x in combination)
        if size > budget:
            continue
        delta = sum(x["delta_kld"] for x in combination)
        key = (delta, size)
        if best is None or key < best[0]:
            best = (key, list(combination))
    return best

def run_search(args):
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_model).resolve()
    if not base_path.is_file():
        raise FileNotFoundError("Build a plain Q4_K base first: " + str(base_path))
    base_kld = get_kl(args.base_log)
    if base_kld is None:
        raise ValueError("Base evaluation log has no mean KLD")
    base_bytes = base_path.stat().st_size
    budget = round(args.target_gib * 1024 ** 3)
    ts = inventory(args.model)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    for g in groups:
        if g not in GROUPS:
            raise ValueError("Unknown group: " + g)
    for t in types:
        if t not in ("Q3_K", "Q5_K", "Q6_K", "Q8_0"):
            raise ValueError("Invalid search type: " + t)
    if len(set(groups)) != len(groups):
        raise ValueError("Groups must be unique")
    groups = [g for g in groups if any(t["eligible"] and re.search(GROUPS[g], t["name"]) for t in ts)]
    model_sha = digest(args.model)
    base_manifest = base_path.with_suffix(".manifest.json")
    reference_path = Path(args.reference).resolve()
    reference_manifest = reference_path.with_suffix(".baseline.json")
    if not base_manifest.exists() or not reference_manifest.exists():
        raise ValueError("Missing model receipts for Q4 baseline or BF16 reference")
    base_meta = json.loads(base_manifest.read_text(encoding="utf-8"))
    reference_meta = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if base_meta.get("source_sha256") != model_sha or reference_meta.get("model_sha256") != model_sha:
        raise ValueError("Q4 model and BF16 reference must share the same source GGUF")
    metadata = {
        "model_sha256": model_sha,
        "base_model_sha256": digest(base_path),
        "base_log_sha256": digest(args.base_log),
        "reference_sha256": digest(reference_path),
        "corpus_sha256": digest(args.corpus),
        "ctx": args.ctx,
        "chunks": args.chunks,
        "ngl": args.ngl,
        "quantizer": str(Path(args.quantizer).resolve()),
        "ppl": str(Path(args.ppl).resolve()),
        "imatrix_sha256": digest(args.imatrix) if args.imatrix else None,
        "groups": groups,
        "types": types,
    }
    meta_file = workdir / "search-run.json"
    cache = workdir / "trials.json"
    if cache.exists():
        if not meta_file.exists() or json.loads(meta_file.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cached trials have different or missing provenance; use a new --workdir")
        results = json.loads(cache.read_text(encoding="utf-8"))
    else:
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results = []
    trial_count = 0
    for group in groups:
        for typ in types:
            if any(r["group"] == group and r["typ"] == typ for r in results):
                print(f"REUSE {group}-{typ} from recorded probe", flush=True)
                continue
            if trial_count >= args.max_trials:
                break
            pattern = GROUPS[group]
            matches = [t for t in ts if t["eligible"] and re.search(pattern, t["name"])]
            delta_bytes = sum(quant_bytes(t, typ) - quant_bytes(t, "Q4_K") for t in matches)
            name = group + "-" + typ.lower()
            output = workdir / (name + ".gguf")
            if shutil.disk_usage(workdir).free < base_bytes + 4 * 1024 ** 3:
                raise RuntimeError("Insufficient free disk for the next candidate")
            print(f"PROBE {trial_count+1}/{args.max_trials}: {name}, estimated delta {delta_bytes/2**20:+.1f} MiB", flush=True)
            build(args.model, output, args.quantizer, "Q4_K", [pattern + "=" + typ], args.imatrix, args.threads)
            try:
                score = evaluate(output, args.corpus, args.ppl, args.reference, workdir / (name + ".eval.log"), args.ctx, args.chunks, args.ngl)
                row = dict(group=group, typ=typ, delta_bytes=delta_bytes, kld=score["kld"], delta_kld=score["kld"] - base_kld if score["kld"] is not None else None, same_top_percent=score["same_top_percent"], actual_bytes=score["bytes"])
                results.append(row)
                (workdir / "trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            finally:
                if output.exists() and not args.keep_trials:
                    output.unlink()
            trial_count += 1
        if trial_count >= args.max_trials:
            break
    winner = choose(groups, results, base_bytes, budget)
    if winner is None:
        report = dict(status="no_feasible_recipe", base_kld=base_kld, base_bytes=base_bytes, budget_bytes=budget, trials=results)
    else:
        (delta, predicted_bytes), selections = winner
        overrides = [GROUPS[x["group"]] + "=" + x["typ"] for x in selections if x["typ"] != "Q4_K"]
        report = dict(status="candidate_found", base_kld=base_kld, base_bytes=base_bytes, budget_bytes=budget, predicted_bytes=predicted_bytes, predicted_kld=base_kld + delta, selected=selections, trials=results)
        if overrides:
            output = workdir / "best.gguf"
            build(args.model, output, args.quantizer, "Q4_K", overrides, args.imatrix, args.threads)
            measured = evaluate(output, args.corpus, args.ppl, args.reference, workdir / "best.eval.log", args.ctx, args.chunks, args.ngl)
            report["validated"] = measured
            report["status"] = "validated_under_budget" if measured["bytes"] <= budget else "actual_size_exceeds_budget"
            if measured["kld"] is None or measured["kld"] >= base_kld:
                report["quality_improved_over_plain_q4"] = False
            else:
                report["quality_improved_over_plain_q4"] = True
        else:
            report["status"] = "plain_q4_is_best_among_evaluated"
            report["validated"] = dict(model=str(base_path), bytes=base_bytes, kld=base_kld)
            report["quality_improved_over_plain_q4"] = False
    (workdir / "search-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
