#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)

def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def lock_source_version(out, model, revision):
    lock = out / "source-version.json"
    wanted = {"model": model, "revision": revision}
    if lock.exists():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        if previous != wanted:
            raise RuntimeError("The original model version changed. Use a NEW --output-dir to prevent mixing models.")
    elif (out / "original-f16.gguf").exists():
        receipt = out / "publish-receipt.json"
        if not receipt.exists():
            raise RuntimeError("Existing original-f16.gguf has no version record. Use a NEW --output-dir.")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("source_model") != model or previous.get("source_revision") != revision:
            raise RuntimeError("Existing GGUF belongs to a different source version. Use a NEW --output-dir.")
    lock.write_text(json.dumps(wanted, indent=2) + "\n", encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Download an original Hugging Face model, make a Q4 GGUF, and optionally publish it.")
    p.add_argument("--model", required=True, help="Original Hugging Face model, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--llama-cpp", required=True, type=Path, help="Local llama.cpp source folder containing convert_hf_to_gguf.py and build/bin/llama-quantize")
    p.add_argument("--output-dir", default="my-q4", type=Path)
    p.add_argument("--recipe", type=Path, help="Optional Q4Fit recipe file: tensor_name=q3_k or q5_k")
    p.add_argument("--revision", help="Optional fixed Hugging Face source commit")
    p.add_argument("--repo", help="Your Hugging Face destination, e.g. username/MyModel-Q4-GGUF")
    p.add_argument("--publish", action="store_true", help="Upload GGUF and model card to Hugging Face")
    p.add_argument("--private", action="store_true", help="Create the Hugging Face repository as private")
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 4))
    a = p.parse_args()
    if a.publish and not a.repo:
        p.error("--publish requires --repo")
    if a.repo and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", a.repo):
        p.error("--repo must look like YOUR_USERNAME/MODEL-NAME")
    if a.threads < 1:
        p.error("--threads must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", a.model):
        p.error("--model must look like organization/model-name")
    root = a.llama_cpp.expanduser().resolve()
    convert = root / "convert_hf_to_gguf.py"
    quant = root / "build/bin/llama-quantize"
    if not convert.is_file() or not quant.is_file():
        p.error("llama.cpp tools missing: build llama.cpp first and check --llama-cpp")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    info = api.model_info(a.model, revision=a.revision)
    source_revision = info.sha
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_source_version(out, a.model, source_revision)
    source = out / "original"
    print("Step 1/4: Downloading original model (this may take a while).", flush=True)
    snapshot_download(
        repo_id=a.model,
        revision=source_revision,
        local_dir=str(source),
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.vocab", "*.merges", "*.py"],
    )
    if not list(source.glob("*.safetensors")):
        raise RuntimeError("No original safetensors found. Use a supported original-weight Hugging Face model.")
    print("Step 2/4: Converting original weights to F16 GGUF.", flush=True)
    full = out / "original-f16.gguf"
    if not full.is_file():
        run([sys.executable, str(convert), str(source), "--outfile", str(full), "--outtype", "f16"])
    name = a.model.split("/")[-1]
    output = out / (name + "-Q4_K.gguf" if a.recipe is None else name + "-Q4Fit.gguf")
    print("Step 3/4: Creating your Q4 GGUF with Q4Fit.", flush=True)
    command = [sys.executable, "-m", "q4fit", "build", "--model", str(full), "--output", str(output), "--quantizer", str(quant), "--default", "Q4_K", "--threads", str(a.threads)]
    if a.recipe:
        for line in a.recipe.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("Bad recipe line: " + line)
            tensor, kind = line.rsplit("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9_.]+", tensor) or kind.upper() not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16"}:
                raise ValueError("Bad recipe line: " + line)
            pattern = "^" + re.escape(tensor) + "$"
            command.extend(["--set", pattern + "=" + kind.upper()])
    run(command)
    digest = checksum(output)
    original_license = info.card_data.get("license") if hasattr(info.card_data, "get") else None
    license_metadata = "license: " + json.dumps(original_license) + "\n" if isinstance(original_license, str) and original_license else ""
    card = f"""---
base_model: {a.model}
{license_metadata}library_name: gguf
tags:
- gguf
- quantized
- q4fit
---

# {name} — {'Q4Fit mixed Q4' if a.recipe else 'Q4_K'} GGUF

This is a GGUF quantization of [{a.model}](https://huggingface.co/{a.model}), made from the original weights using [Q4Fit](https://github.com/bettercallcaleb/q4fit) and llama.cpp.

- Original model: [{a.model}](https://huggingface.co/{a.model})
- Original source commit: `{source_revision}`
- Quantization: {'Q4_K with selected per-tensor overrides' if a.recipe else 'pure Q4_K'}
- GGUF file: `{output.name}`
- File SHA256: `{digest}`
- Size: {output.stat().st_size:,} bytes

This is a converted copy, not a newly trained model. No independent quality evaluation is claimed. Check the original model's license and usage conditions before redistributing it.
"""
    card_path = out / "README.md"
    card_path.write_text(card, encoding="utf-8")
    receipt = {"source_model": a.model, "source_revision": source_revision, "output_file": output.name, "output_sha256": digest, "bytes": output.stat().st_size, "recipe": str(a.recipe) if a.recipe else None}
    (out / "publish-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("GGUF READY:", output, "SHA256:", digest, flush=True)
    if a.publish:
        print("Step 4/4: Uploading to Hugging Face.", flush=True)
        from huggingface_hub.utils import HfHubHTTPError
        who = api.whoami()
        try:
            api.repo_info(repo_id=a.repo, repo_type="model")
        except HfHubHTTPError as e:
            if e.response.status_code != 404:
                raise
        else:
            raise RuntimeError("Destination repository already exists. Choose a NEW --repo name to avoid overwriting files.")
        api.create_repo(repo_id=a.repo, repo_type="model", private=a.private, exist_ok=False)
        api.upload_file(path_or_fileobj=str(output), path_in_repo=output.name, repo_id=a.repo, repo_type="model", commit_message="Add GGUF quantization")
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=a.repo, repo_type="model", commit_message="Add model card")
        api.upload_file(path_or_fileobj=str(out / "publish-receipt.json"), path_in_repo="publish-receipt.json", repo_id=a.repo, repo_type="model", commit_message="Add reproducibility receipt")
        print("PUBLISHED: https://huggingface.co/" + a.repo, flush=True)
    else:
        print("Step 4/4: Not published. Add --publish --repo YOUR_NAME/YOUR_MODEL to upload.", flush=True)

if __name__ == "__main__":
    main()
