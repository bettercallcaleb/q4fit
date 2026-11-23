import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PACKS = Path(__file__).resolve().parent / "packs"
LEVELS = ("recognition", "mechanism", "transfer", "boundary")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_packs(selected="all"):
    files = sorted(PACKS.glob("*.json"))
    if selected != "all":
        names = {x.strip() for x in selected.split(",") if x.strip()}
        files = [f for f in files if f.stem in names]
        if len(files) != len(names):
            raise ValueError("Unknown pack name: " + ", ".join(sorted(names - {f.stem for f in files})))
    if not files:
        raise ValueError("No packs available")
    items = []
    seen = set()
    for path in files:
        pack = json.loads(path.read_text(encoding="utf-8"))
        if pack.get("id") != path.stem or not pack.get("source", "").startswith("https://"):
            raise ValueError(f"Invalid pack metadata: {path.name}")
        for concept in pack.get("concepts", []):
            slug = concept.get("id")
            if not slug or not isinstance(concept.get("questions"), list) or len(concept["questions"]) != 4:
                raise ValueError(f"Invalid concept: {path.name}/{slug}")
            for level, question in enumerate(concept["questions"], start=1):
                prompt = question.get("prompt")
                correct = question.get("correct")
                wrong = question.get("wrong")
                if not isinstance(prompt, str) or not prompt.strip() or not isinstance(correct, str) or not isinstance(wrong, list) or len(wrong) != 3:
                    raise ValueError(f"Invalid question: {path.name}/{slug}/{level}")
                if len(set([correct] + wrong)) != 4 or not all(isinstance(x, str) and x.strip() for x in [correct] + wrong):
                    raise ValueError(f"Ambiguous answers: {path.name}/{slug}/{level}")
                item_id = f"{pack['id']}/{slug}/L{level}"
                if item_id in seen:
                    raise ValueError("Duplicate item: " + item_id)
                seen.add(item_id)
                items.append({"id": item_id, "pack": pack["id"], "concept": slug, "level": level, "prompt": prompt, "correct": correct, "wrong": wrong, "source": concept.get("source", pack["source"])})
    if not items:
        raise ValueError("No questions available")
    return items


def fingerprint(items):
    return hashlib.sha256(canonical(items).encode("utf-8")).hexdigest()


def materialize(item, seed):
    nonce = int.from_bytes(hashlib.sha256(f"{seed}|{item['pack']}|{item['concept']}".encode()).digest()[:8], "big")
    slots = list(range(4))
    random.Random(nonce).shuffle(slots)
    target = slots[item["level"] - 1]
    wrong = list(item["wrong"])
    random.Random(nonce ^ (item["level"] * 65537)).shuffle(wrong)
    choices = wrong[:target] + [item["correct"]] + wrong[target:]
    letters = "ABCD"
    answer = letters[target]
    question = item["prompt"] + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return question, answer


def parse_letter(content):
    if isinstance(content, list):
        content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    if not isinstance(content, str):
        return None
    match = re.fullmatch(r"\s*(?:(?:answer|option)\s*[:=]\s*)?([A-Da-d])\s*[.)]?\s*", content)
    return match.group(1).upper() if match else None


def request_answer(url, model, prompt, api_key, timeout, max_tokens):
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "Choose the single best answer. Reply with exactly one uppercase letter: A, B, C, or D. No explanation."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        result = json.load(res)
    return result["choices"][0]["message"].get("content", "")


def grade(run, items):
    if run.get("dataset_sha256") != fingerprint(items):
        raise ValueError("Dataset hash differs; do not compare different benchmark versions")
    expected = {i["id"]: i for i in items}
    responses = run.get("responses", [])
    got = {r.get("id"): r for r in responses}
    if len(got) != len(responses) or set(got) != set(expected):
        raise ValueError("Missing, duplicated, or extra response IDs")
    correct = {}
    per_pack = defaultdict(lambda: {"correct": 0, "total": 0})
    by_concept = defaultdict(dict)
    invalid = 0
    for item in items:
        row = got[item["id"]]
        pred = row.get("prediction")
        if pred not in ("A", "B", "C", "D", None):
            raise ValueError("Invalid prediction: " + item["id"])
        if pred is None:
            invalid += 1
        _, answer = materialize(item, run["seed"])
        passed = pred == answer
        correct[item["id"]] = passed
        per_pack[item["pack"]]["correct"] += int(passed)
        per_pack[item["pack"]]["total"] += 1
        by_concept[(item["pack"], item["concept"])][item["level"]] = passed
    depths = {}
    for (pack, concept), levels in sorted(by_concept.items()):
        depth = 0
        for i in range(1, 5):
            if levels.get(i):
                depth += 1
            else:
                break
        depths[f"{pack}/{concept}"] = depth
    return {
        "accuracy": round(sum(correct.values()) / len(items), 6),
        "correct": sum(correct.values()),
        "total": len(items),
        "invalid": invalid,
        "mean_contiguous_depth": round(sum(depths.values()) / len(depths), 6),
        "depths": depths,
        "packs": dict(sorted(per_pack.items())),
        "pass": correct,
    }


def create_run(args, items):
    responses = []
    for index, item in enumerate(items, 1):
        prompt, _ = materialize(item, args.seed)
        try:
            raw = request_answer(args.url, args.model, prompt, args.api_key, args.timeout, args.max_tokens)
            pred = parse_letter(raw)
            error = None if pred else "unparseable"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            pred = None
            error = type(exc).__name__ + ": " + str(exc)[:160]
        responses.append({"id": item["id"], "prediction": pred, "error": error})
        print(f"{index}/{len(items)} {item['id']} -> {pred or '-'}", file=sys.stderr, flush=True)
        if args.delay:
            time.sleep(args.delay)
    return {
        "format": "conceptladder-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": args.url,
        "seed": args.seed,
        "dataset_sha256": fingerprint(items),
        "responses": responses,
    }


def write_json(path, data):
    output = canonical(data) + "\n"
    if path == "-":
        sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        print("Saved " + str(target), file=sys.stderr)


def load_run(path):
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    if run.get("format") != "conceptladder-run-v1" or not isinstance(run.get("seed"), int):
        raise ValueError("Unsupported or invalid run file")
    return run


def main():
    parser = argparse.ArgumentParser(prog="conceptladder", description="A compact benchmark for knowledge depth and quantization retention.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.add_argument("--packs", default="all")
    p = sub.add_parser("validate")
    p.add_argument("--packs", default="all")
    p = sub.add_parser("run")
    p.add_argument("--packs", default="all")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--delay", type=float, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("file")
    p.add_argument("--out", default="-")
    p = sub.add_parser("compare")
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--out", default="-")
    args = parser.parse_args()
    if args.command in ("list", "validate", "run"):
        items = load_packs(args.packs)
    if args.command == "list":
        packs = defaultdict(set)
        for item in items:
            packs[item["pack"]].add(item["concept"])
        for pack, concepts in sorted(packs.items()):
            print(f"{pack}: {len(concepts)} concepts, {len(concepts) * 4} questions")
    elif args.command == "validate":
        print(f"Validated {len(items)} questions across {len({x['pack'] for x in items})} packs; sha256 {fingerprint(items)}")
    elif args.command == "run":
        result = create_run(args, items)
        write_json(args.out, result)
        stats = grade(result, items)
        print(f"Accuracy {stats['correct']}/{stats['total']}; mean depth {stats['mean_contiguous_depth']:.2f}/4", file=sys.stderr)
    elif args.command == "score":
        result = load_run(args.file)
        ids = {r["id"].split("/")[0] for r in result["responses"]}
        items = load_packs(",".join(sorted(ids)))
        stats = grade(result, items)
        stats.pop("pass")
        write_json(args.out, {"model": result.get("model"), "dataset_sha256": result["dataset_sha256"], **stats})
    elif args.command == "compare":
        base = load_run(args.reference)
        candidate = load_run(args.candidate)
        if base["dataset_sha256"] != candidate["dataset_sha256"] or base["seed"] != candidate["seed"]:
            raise ValueError("Reference and candidate need matching dataset hashes and seeds")
        ids = {r["id"].split("/")[0] for r in base["responses"]}
        items = load_packs(",".join(sorted(ids)))
        a, b = grade(base, items), grade(candidate, items)
        lost = sorted(k for k in a["pass"] if a["pass"][k] and not b["pass"][k])
        gained = sorted(k for k in a["pass"] if not a["pass"][k] and b["pass"][k])
        retained = sum(a["pass"][k] and b["pass"][k] for k in a["pass"])
        output = {
            "reference_model": base.get("model"),
            "candidate_model": candidate.get("model"),
            "dataset_sha256": base["dataset_sha256"],
            "seed": base["seed"],
            "reference_accuracy": a["accuracy"],
            "candidate_accuracy": b["accuracy"],
            "accuracy_delta": round(b["accuracy"] - a["accuracy"], 6),
            "reference_mean_depth": a["mean_contiguous_depth"],
            "candidate_mean_depth": b["mean_contiguous_depth"],
            "depth_delta": round(b["mean_contiguous_depth"] - a["mean_contiguous_depth"], 6),
            "retained_when_reference_correct": retained,
            "reference_correct": a["correct"],
            "knowledge_loss_rate": round(len(lost) / a["correct"], 6) if a["correct"] else None,
            "lost": lost,
            "gained": gained,
            "reference_depths": a["depths"],
            "candidate_depths": b["depths"],
        }
        write_json(args.out, output)


if __name__ == "__main__":
    main()
