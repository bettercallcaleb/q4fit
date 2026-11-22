# Q4Fit v0.3 — Cached GGUF Tensor Banks

Q4Fit composes a valid, mixed-precision GGUF by reusing independently quantized tensor payloads. Its bit allocation is controlled by the CLI. The cache avoids rerunning the BF16 → Q3/Q4/Q5 quantization pass for each candidate.

## How it works

1. Build one complete GGUF per requested quant type from **the same original BF16**, using the **same llama.cpp quantizer build** and optional imatrix. Existing Q4Fit `build` outputs can be reused.
2. Index tensor names, shapes, actual types, byte ranges, alignment, metadata compatibility, source SHA256, and bank SHA256.
3. A mixed recipe chooses a bank for each compatible tensor. The composer copies the Q4_K header, patches each tensor's type and offset, and streams precomputed payloads into a new aligned GGUF via Linux `sendfile`.
4. A normal compose verifies the resulting GGUF descriptors and hashes the output. `--fast` skips the expensive full-model verification and output SHA **for disposable search candidates only**. It still checks that bank sizes/mtime and base header hash match the index, and writes the destination atomically.
5. Candidate ranking still requires BF16-referenced evaluation; bitwise equality alone does not prove model quality.

**Limitations:** GGUF v3, little-endian, single-file models only; default Q4_K; supported tensor types depend on compatible cached banks. No promise for other quantizers, imatrix modes, or special tensor layouts unless equality is independently tested.

## CLI

```bash
pip install -e .
export BF16=/models/qwen3-4b-bf16.gguf
export QUANT=/opt/llama.cpp/build/bin/llama-quantize
export PPL=/opt/llama.cpp/build/bin/llama-perplexity

q4fit-cache prepare --model "$BF16" --cache work/banks --types Q3_K,Q4_K,Q5_K --quantizer "$QUANT"

q4fit-cache audit --index work/banks/index.json --verify-banks

q4fit-cache compose --index work/banks/index.json --output work/early-down-q5.gguf --set '^blk\.(0|1|2|3)\.ffn_down\.weight$=Q5_K'

q4fit-cache compose --index work/banks/index.json --output work/disposable.gguf --fast --set '^blk\.0\.attn_q\.weight$=Q3_K'
```

`--reuse Q4_K=/path/existing-plain-q4.gguf` avoids storing another Q4_K copy, provided the file was produced by Q4Fit `build` with a compatible manifest. The same option supports reusable Q3/Q5 banks.

Search from the cache via `q4fit-multi` by adding the `cache_index` key to the JSON config. Each candidate uses fast composition; the final combined model uses full descriptor verification and SHA256.

Inspect tensor sensitivity one weight at a time:

```bash
q4fit-scan --index work/banks/index.json --corpus /work/code.txt --reference /work/code-bf16.logits --base-log /work/code-plain-q4.log --ppl "$PPL" --workdir work/sensitivity-code --target Q3_K --match '^blk\.(0|1|2|3)\.ffn_down\.weight$' --limit 4 --ctx 512 --chunks 8
```

The scan writes `results.json`, `results.csv`, BF16 KL evaluation logs, and per-tensor size deltas; it resumes without retesting completed tensors. Each candidate is removed after evaluation by default.

## Real Qwen3-4B comparison

Cache banks: Q3_K 1.616 GiB, Q4_K 2.113 GiB (reused existing), Q5_K 2.582 GiB. Extra banks cost about 4.2 GiB, plus index metadata. Quantization of each bank is done once.

The 18-tensor early-FFN-Down Q5 recipe was rebuilt from the Q3/Q4/Q5 cache with a **SHA256 identical** to direct llama.cpp quantization:

`4c26092b49095c2a7a57d39f386a237682583411530f938436a301f7c8e4534a`

A second 72-tensor mixed Q3_K/Q5_K recipe also produced identical bytes to the independently quantized original:

`0505678cf84cd1b09d653ba4c0423fc1a189b145df908201b8f88c4ecdc780e7`

A one-tensor GGUF compose took **1.56 seconds** (1.67 s including process start) with `--fast`, versus roughly **30–40 seconds** for a complete Qwen3-4B GGUF quantization pass on this host. Speedup is approximately 19–26× for this stage, excluding BF16 evaluation and one-time cache building. Your SSD speed, cache warmth, tensor types and hardware affect performance.

Do not distribute source weights or cached banks as part of the public repository. Check the model's license before distributing quantized artifacts.
