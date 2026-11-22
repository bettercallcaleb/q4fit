#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BLOCKS = {"Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_0": (32, 34), "F16": (1, 2), "BF16": (1, 2)}
GROUPS = {
    "embedding": r"^token_embd\.weight$",
    "output": r"^output\.weight$",
    "attn_q": r"\.attn_q\.weight$",
    "attn_k": r"\.attn_k\.weight$",
    "attn_v": r"\.attn_v\.weight$",
    "attn_output": r"\.attn_output\.weight$",
    "ffn_gate": r"\.ffn_gate\.weight$",
    "ffn_up": r"\.ffn_up\.weight$",
    "ffn_down": r"\.ffn_down\.weight$",
}

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def inventory(path):
    from gguf import GGUFReader, GGMLQuantizationType
    r = GGUFReader(str(path), "r")
    result = []
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        n = 1
        for x in shape:
            n *= x
        typ = GGMLQuantizationType(t.tensor_type).name
        eligible = len(shape) == 2 and shape[0] % 256 == 0 and typ in ("BF16", "F16", "F32")
        result.append(dict(name=t.name, shape=shape, n=n, source=typ, bytes=int(t.n_bytes), eligible=eligible))
    return result

def quant_bytes(t, typ):
    if not t["eligible"]:
        return t["bytes"]
    block, size = BLOCKS[typ]
    if t["shape"][0] % block:
        raise ValueError(f"{t['name']} incompatible with {typ}")
    return t["n"] // block * size

def override_map(ts, rules):
    overrides = {}
    for rule in rules:
        if "=" not in rule:
            raise ValueError(f"Expected REGEX=TYPE, got {rule}")
        pattern, typ = rule.rsplit("=", 1)
        typ = typ.upper()
        if typ not in BLOCKS:
            raise ValueError(f"Unsupported type {typ}")
        count = 0
        for t in ts:
            if t["eligible"] and re.search(pattern, t["name"]) and t["shape"][0] % BLOCKS[typ][0] == 0:
                overrides[t["name"]] = typ
                count += 1
        if count == 0:
            raise ValueError(f"No matching compatible tensors for {rule}")
    return overrides

def recipe(ts, default, overrides, directory):
    directory.mkdir(parents=True, exist_ok=True)
    recipe_file = directory / "tensor-types.txt"
    with recipe_file.open("w", encoding="utf-8") as f:
        for t in ts:
            v = overrides.get(t["name"])
            if v and v != default:
                f.write(f"{t['name']}={v.lower()}\n")
    total = sum(quant_bytes(t, overrides.get(t["name"], default)) for t in ts)
    data = dict(default=default, overrides=overrides, estimated_bytes=total)
    (directory / "recipe.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return recipe_file, total

def execute(cmd, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + shlex.join(map(str, cmd)), flush=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(shlex.join(map(str, cmd)) + "\n")
        f.flush()
        proc = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    tail = log.read_text(encoding="utf-8", errors="replace")[-2600:]
    print(tail, flush=True)
    if proc.returncode:
        raise RuntimeError(f"Exit {proc.returncode}: {log}")
    return tail

def build(model, output, quantizer, default, changes, imatrix=None, threads=12, dry_run=False):
    model, output = Path(model).resolve(), Path(output).resolve()
    ts = inventory(model)
    overrides = override_map(ts, changes)
    recipe_file, predicted = recipe(ts, default, overrides, output.parent / (output.stem + "-recipe"))
    cmd = [quantizer, "--pure", "--tensor-type-file", recipe_file]
    if imatrix:
        cmd += ["--imatrix", Path(imatrix).resolve()]
    if dry_run:
        cmd += ["--dry-run"]
    cmd += [model, output, default.lower(), threads]
    execute(cmd, output.with_suffix(".quant.log"))
    if not dry_run:
        if not output.is_file():
            raise RuntimeError("Missing output GGUF")
        manifest = dict(source=str(model), source_sha256=digest(model), output=str(output), output_sha256=digest(output), bytes=output.stat().st_size, predicted_bytes=predicted, default=default, overrides=overrides, imatrix=imatrix)
        output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return overrides

def baseline(model, corpus, ppl, output, ctx, chunks, ngl):
    model, corpus, output = Path(model).resolve(), Path(corpus).resolve(), Path(output).resolve()
    execute([ppl, "-m", model, "-f", corpus, "-c", ctx, "--chunks", chunks, "-ngl", ngl, "--save-all-logits", output], output.with_suffix(".baseline.log"))
    if not output.is_file() or not output.stat().st_size:
        raise RuntimeError("BF16 logit reference missing")
    record = dict(model=str(model), model_sha256=digest(model), corpus=str(corpus), corpus_sha256=digest(corpus), ctx=ctx, chunks=chunks, reference=str(output), reference_bytes=output.stat().st_size, ppl=str(ppl))
    output.with_suffix(".baseline.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

def read_metric(log, pattern):
    s = Path(log).read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, s, flags=re.I | re.M)
    return float(match.group(1)) if match else None

def get_kl(log):
    value = read_metric(log, r"^\s*Mean\s+KLD:\s*([0-9]+(?:\.[0-9]+)?(?:[Ee][-+]?[0-9]+)?)")
    if value is not None:
        return value
    return read_metric(log, r"^\s*(?:Average|Mean)\s+KL[- ]divergence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")

def evaluate(model, corpus, ppl, reference, log, ctx, chunks, ngl):
    model, corpus, reference, log = map(lambda x: Path(x).resolve(), [model, corpus, reference, log])
    if not reference.is_file():
        raise FileNotFoundError(reference)
    meta = reference.with_suffix(".baseline.json")
    if meta.exists():
        b = json.loads(meta.read_text(encoding="utf-8"))
        if b["corpus_sha256"] != digest(corpus) or b["ctx"] != ctx or b["chunks"] != chunks:
            raise ValueError("Baseline corpus/context/chunks mismatch")
    execute([ppl, "-m", model, "-f", corpus, "-c", ctx, "--chunks", chunks, "-ngl", ngl, "--kl-divergence", "--kl-divergence-base", reference], log)
    result = dict(model=str(model), bytes=model.stat().st_size, kld=get_kl(log), kld_p99=read_metric(log, r"^\s*99\.0%\s+KLD:\s*([0-9.]+)"), same_top_percent=read_metric(log, r"^\s*Same top p:\s*([0-9.]+)"), corpus_sha256=digest(corpus), reference=str(reference), ctx=ctx, chunks=chunks, log=str(log))
    log.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["kld"] is None:
        print("KLD not parsed; do not infer quality without checking logs", file=sys.stderr)
    return result

def inspect(args):
    ts = inventory(args.model)
    groups = defaultdict(lambda: [0, 0, 0])
    for t in ts:
        group = next((k for k, rx in GROUPS.items() if re.search(rx, t["name"])), "other")
        g = groups[group]
        g[0] += 1
        g[1] += t["bytes"]
        g[2] += quant_bytes(t, "Q4_K")
    print(f"tensors={len(ts)} source_GiB={sum(t['bytes'] for t in ts)/2**30:.3f} predicted_Q4_GiB={sum(quant_bytes(t,'Q4_K') for t in ts)/2**30:.3f}")
    for key, (n, src, q) in sorted(groups.items(), key=lambda v: -v[1][1]):
        print(f"{key:15s} {n:4d}   BF16={src/2**30:7.3f} GiB   Q4={q/2**30:7.3f} GiB")
    if args.json:
        Path(args.json).write_text(json.dumps(ts, indent=2), encoding="utf-8")

def probe(args):
    directory = Path(args.workdir).resolve() / args.name
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "candidate.gguf"
    overrides = build(args.model, output, args.quantizer, args.default, args.set, args.imatrix, args.threads)
    result = evaluate(output, args.corpus, args.ppl, args.reference, directory / "eval.log", args.ctx, args.chunks, args.ngl)
    result["overrides"] = overrides
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.keep:
        output.unlink()
        print("Temporary GGUF removed; receipt retained")
    print(json.dumps(result, indent=2))

def main():
    p = argparse.ArgumentParser(description="Q4Fit: reproducible BF16-referenced tensor-wise GGUF quantization")
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--model", required=True)
    i.add_argument("--json")
    i.set_defaults(func=inspect)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--quantizer", default="llama-quantize")
    b.add_argument("--default", choices=BLOCKS, default="Q4_K")
    b.add_argument("--set", action="append", default=[], metavar="REGEX=TYPE")
    b.add_argument("--imatrix")
    b.add_argument("--threads", type=int, default=12)
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=lambda a: build(a.model, a.output, a.quantizer, a.default, a.set, a.imatrix, a.threads, a.dry_run))
    f = sub.add_parser("baseline")
    f.add_argument("--model", required=True)
    f.add_argument("--corpus", required=True)
    f.add_argument("--ppl", default="llama-perplexity")
    f.add_argument("--output", required=True)
    f.add_argument("--ctx", type=int, default=256)
    f.add_argument("--chunks", type=int, default=1)
    f.add_argument("--ngl", type=int, default=999)
    f.set_defaults(func=lambda a: print(json.dumps(baseline(a.model, a.corpus, a.ppl, a.output, a.ctx, a.chunks, a.ngl), indent=2)))
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--corpus", required=True)
    e.add_argument("--ppl", default="llama-perplexity")
    e.add_argument("--reference", required=True)
    e.add_argument("--log", required=True)
    e.add_argument("--ctx", type=int, default=256)
    e.add_argument("--chunks", type=int, default=1)
    e.add_argument("--ngl", type=int, default=999)
    e.set_defaults(func=lambda a: print(json.dumps(evaluate(a.model, a.corpus, a.ppl, a.reference, a.log, a.ctx, a.chunks, a.ngl), indent=2)))
    q = sub.add_parser("probe")
    q.add_argument("--model", required=True)
    q.add_argument("--corpus", required=True)
    q.add_argument("--reference", required=True)
    q.add_argument("--workdir", required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--set", action="append", required=True)
    q.add_argument("--quantizer", default="llama-quantize")
    q.add_argument("--ppl", default="llama-perplexity")
    q.add_argument("--default", choices=BLOCKS, default="Q4_K")
    q.add_argument("--imatrix")
    q.add_argument("--threads", type=int, default=12)
    q.add_argument("--ctx", type=int, default=256)
    q.add_argument("--chunks", type=int, default=1)
    q.add_argument("--ngl", type=int, default=999)
    q.add_argument("--keep", action="store_true")
    q.set_defaults(func=probe)
    s = sub.add_parser("search")
    s.add_argument("--model", required=True)
    s.add_argument("--base-model", required=True)
    s.add_argument("--base-log", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--workdir", required=True)
    s.add_argument("--target-gib", type=float, required=True)
    s.add_argument("--groups", default="attn_v,ffn_down,attn_q")
    s.add_argument("--types", default="Q3_K,Q5_K")
    s.add_argument("--max-trials", type=int, default=6)
    s.add_argument("--quantizer", default="llama-quantize")
    s.add_argument("--ppl", default="llama-perplexity")
    s.add_argument("--imatrix")
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--ctx", type=int, default=256)
    s.add_argument("--chunks", type=int, default=1)
    s.add_argument("--ngl", type=int, default=999)
    s.add_argument("--keep-trials", action="store_true")
    s.set_defaults(func=lambda args: __import__("q4search").run_search(args))
    a = p.parse_args()
    try:
        a.func(a)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        p.exit(1, f"error: {exc}\n")

if __name__ == "__main__":
    main()
