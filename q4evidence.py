import argparse
import hashlib
import json
import math
import re
from pathlib import Path

FORMAT = "ramgpt-experiment-v1"
HASH = re.compile(r"^[0-9a-f]{64}$")
INPUTS = ("receipt.json", "reference.json", "candidate.json", "comparison.json")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def require_hash(value, label):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError(f"Invalid {label} SHA-256")
    return value


def require_number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {label}")
    return value


def build(depth_dir):
    folder = Path(depth_dir)
    paths = {name: folder / name for name in INPUTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("Missing completed depth evidence: " + ", ".join(missing))
    receipt = load(paths["receipt.json"])
    comparison = load(paths["comparison.json"])
    reference = load(paths["reference.json"])
    candidate = load(paths["candidate.json"])
    if receipt.get("format") != "q4fit-depth-v1":
        raise ValueError("Unsupported depth receipt format")
    if not str(receipt.get("benchmark", "")).startswith("conceptladder-bench/"):
        raise ValueError("Unsupported benchmark")
    dataset = require_hash(receipt.get("dataset_sha256"), "dataset")
    if comparison.get("dataset_sha256") != dataset:
        raise ValueError("Comparison dataset fingerprint mismatch")
    for label, run in (("reference", reference), ("candidate", candidate)):
        if run.get("dataset_sha256") != dataset:
            raise ValueError(f"{label} dataset fingerprint mismatch")
        if run.get("seed") != receipt.get("seed"):
            raise ValueError(f"{label} seed mismatch")
        if run.get("model") != comparison.get(label + "_model"):
            raise ValueError(f"{label} model label mismatch")
    for source in (comparison, reference, candidate):
        if source.get("answer_seed", source.get("seed")) != receipt.get("answer_seed"):
            raise ValueError("Answer-order seed mismatch")
    if comparison.get("seed") != receipt.get("seed"):
        raise ValueError("Comparison seed mismatch")
    items = receipt.get("items")
    if type(items) is not int or items < 1:
        raise ValueError("Invalid question count")
    for label in ("reference_accuracy", "candidate_accuracy", "knowledge_loss_rate"):
        if comparison.get(label) is not None:
            require_number(comparison[label], label, 0, 1)
    for label in ("reference_mean_depth", "candidate_mean_depth"):
        require_number(comparison.get(label), label, 0, 4)
    for label, low, high in (("accuracy_delta", -1, 1), ("depth_delta", -4, 4)):
        require_number(comparison.get(label), label, low, high)
    if abs(comparison["candidate_accuracy"] - comparison["reference_accuracy"] - comparison["accuracy_delta"]) > 0.00001:
        raise ValueError("Inconsistent accuracy delta")
    if abs(comparison["candidate_mean_depth"] - comparison["reference_mean_depth"] - comparison["depth_delta"]) > 0.00001:
        raise ValueError("Inconsistent depth delta")
    if not isinstance(comparison.get("lost"), list) or not isinstance(comparison.get("gained"), list):
        raise ValueError("Missing paired answer flips")
    correct = comparison.get("reference_correct")
    retained = comparison.get("retained_when_reference_correct")
    if type(correct) is not int or type(retained) is not int or not 0 <= retained <= correct <= items:
        raise ValueError("Invalid reference correct counts")
    if correct - retained != len(comparison["lost"]):
        raise ValueError("Reference retention does not match lost answers")
    for label in ("reference", "candidate"):
        artifact = receipt.get(label)
        if not isinstance(artifact, dict):
            raise ValueError(f"Missing {label} artifact")
        require_hash(artifact.get("sha256"), label)
        if type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0:
            raise ValueError(f"Invalid {label} file size")
    server = require_hash(receipt.get("server_sha256"), "server")
    if receipt["reference"]["sha256"] == receipt["candidate"]["sha256"]:
        raise ValueError("Reference and candidate are byte-identical")
    return {
        "schema": FORMAT,
        "status": "exploratory",
        "producer": {"name": "q4fit", "source_format": receipt["format"]},
        "artifacts": {
            label: {"sha256": receipt[label]["sha256"], "bytes": receipt[label]["bytes"]}
            for label in ("reference", "candidate")
        },
        "evaluation": {
            "benchmark": receipt["benchmark"],
            "suite": receipt.get("suite"),
            "dataset_sha256": dataset,
            "items": items,
            "variants": receipt.get("variants"),
            "seed": receipt["seed"],
            "answer_seed": receipt["answer_seed"],
            "inference_seed": receipt["inference_seed"],
            "context": receipt.get("context"),
            "gpu_layers": receipt.get("gpu_layers"),
            "temperature": receipt.get("temperature"),
            "allow_thinking": receipt.get("allow_thinking"),
            "server_sha256": server,
        },
        "results": {
            key: comparison.get(key) for key in (
                "reference_accuracy", "candidate_accuracy", "accuracy_delta",
                "reference_mean_depth", "candidate_mean_depth", "depth_delta",
                "reference_correct", "retained_when_reference_correct", "knowledge_loss_rate",
            )
        } | {"lost_count": len(comparison["lost"]), "gained_count": len(comparison["gained"])},
        "evidence_sha256": {name: digest(path) for name, path in paths.items()},
        "limitations": [
            "ConceptLadder is a correlated multiple-choice diagnostic, not a validated general capability score.",
            "This export is not a QuantBench verified result or evidence of universal quantization superiority.",
            "The exporter does not verify the model binaries or re-run the benchmark.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a completed q4fit-depth run as a path-free RAMGPT evidence record.")
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.resolve() in {(args.depth_dir / name).resolve() for name in INPUTS}:
        parser.error("Output cannot overwrite input evidence")
    if args.out.exists() and not args.overwrite:
        parser.error("Output exists; choose another --out or pass --overwrite")
    try:
        record = build(args.depth_dir)
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
