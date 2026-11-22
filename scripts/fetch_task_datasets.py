#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


DATASETS = [
    ("mgsm_en", "juletxara/mgsm", "en"),
    ("mgsm_zh", "juletxara/mgsm", "zh"),
    ("mbpp", "google-research-datasets/mbpp", "sanitized"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    from datasets import load_dataset
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for key, repo, config in DATASETS:
        destination = output_dir / (key + ".jsonl")
        if destination.exists():
            observed = sha(destination)
            if observed != manifest["sources"][key]["full_source_jsonl_sha256"]:
                raise ValueError("Existing frozen source differs: " + str(destination))
            print("REUSE", key, observed, flush=True)
            continue
        dataset = load_dataset(repo, config, split="test")
        if len(dataset) != manifest["sources"][key]["rows"]:
            raise ValueError("Unexpected dataset length: " + key)
        temporary = output_dir / (key + ".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for row in dataset:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        observed = sha(temporary)
        if observed != manifest["sources"][key]["full_source_jsonl_sha256"]:
            temporary.unlink()
            raise ValueError("Dataset changed since original freeze: " + key)
        os.replace(temporary, destination)
        print("DOWNLOADED", key, observed, flush=True)


if __name__ == "__main__":
    main()
