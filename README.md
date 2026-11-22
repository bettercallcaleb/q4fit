# Q4Fit

Experimental CLI workbench for BF16-referenced per-tensor GGUF mixed-precision quantization. The tool uses upstream llama.cpp quantizer and perplexity evaluator, and stores auditable tensor recipes and SHA256 receipts.

**Important result:** The original one-chunk v0.1 optimization **failed to generalize** on a larger WikiText-2 test. At comparable size, v0.1 mixed KLD was 0.125771 versus plain Q4_K 0.110148 (lower is better). See [Benchmark protocol](docs/PROTOCOL.md) for the falsification and larger matched evaluations. Never cite the small-sample result alone as evidence of improvement.

**Earlier v0.3 above-budget Pareto result:** A conservative early-FFN-Down Q5_K upgrade (18 tensors, otherwise Q4_K) has now improved BF16 KLD on two WikiText test segments while adding 2.47% file size, at roughly 1.6% lower single-GPU decode throughput. It is **not within the original 2.12 GiB hard budget**. See [expanded results](benchmarks/QWEN3_4B_EXPANDED.md) and the published [18-tensor recipe](recipes/qwen3-4b-early-down-q5.txt).

Implemented: inspect, build, BF16 baseline/eval, per-tensor probe, group search, multi-domain layer-range search, and v0.3 **prequantized tensor-bank caching, fast GGUF composition, and individual tensor sensitivity scan**. This is an experiment framework, not a claim of new quantization math.

**v0.3 verified:** Precomputed Q3_K/Q4_K/Q5_K tensor banks assembled two Qwen3-4B mixed-bit GGUF models **byte-for-byte identical** to independent upstream llama.cpp quantizations. Fast composition of a ~2.2 GB candidate took 1.56 seconds vs 30–40 seconds for direct quantization (excluding one-time bank creation and BF16 evaluation). See [Cache design and CLI](docs/CACHE.md) and [measurement receipts](benchmarks/QWEN3_4B_CACHE.md).

**v0.4:** `q4fit-tune` performs multi-domain per-tensor scans and approximates byte-budgeted mixed-bit allocation using a beam search. It ranks by worst relative BF16 KL across tuning domains, then independently rebuilds and evaluates combined tensors with actual size and top-token gates. [Methodology and CLI](docs/TENSOR_OPTIMIZER.md). A standalone holdout is mandatory before claiming generalization.

**v0.4 Qwen3-4B measured result:** After a 54-tensor scan, a **seven-tensor Q3_K/Q5_K mix** remained within the original **2.12 GiB** constraint and reduced BF16 KLD on two frozen, independent WikiText/code holdouts by **16.0%/13.7%** versus pure Q4_K. Its bytes were identical to an independent direct llama.cpp quantization. A larger standard Q4_K_S model had worse mean KLD on both sampled holdouts, though some tail/top-token measures favored Q4_K_S. Decode speed on RTX 4090 was approximately **0.4% below** pure Q4_K. [Full report](benchmarks/QWEN3_4B_TENSOR_SEARCH.md), [54 measured tensor probes](benchmarks/QWEN3_4B_TENSOR_SWEEP.csv), [exact recipe](recipes/qwen3-4b-v04-7tensor.txt). One-model PoC only; no broad capability claim.

## Setup

Python 3.10+, gguf, a BF16 or F16 GGUF source, and recent llama.cpp llama-quantize and llama-perplexity binaries supporting the model.

~~~bash
python3 -m pip install -e .
export BF16=/models/model-bf16.gguf
export QUANT=/opt/llama.cpp/build/bin/llama-quantize
export PPL=/opt/llama.cpp/build/bin/llama-perplexity
q4fit inspect --model "$BF16"
q4fit build --model "$BF16" --quantizer "$QUANT" --output work/plain-q4.gguf
q4fit baseline --model "$BF16" --corpus corpus/smoke.txt --ppl "$PPL" --output work/bf16-reference.logits --ctx 256 --chunks 1
q4fit eval --model work/plain-q4.gguf --corpus corpus/smoke.txt --ppl "$PPL" --reference work/bf16-reference.logits --log work/plain-q4-eval.log
q4fit probe --model "$BF16" --corpus corpus/smoke.txt --reference work/bf16-reference.logits --quantizer "$QUANT" --ppl "$PPL" --workdir work/probes --name attn-v-q5 --set '\.attn_v\.weight$=Q5_K'
~~~

## Measured search at a fixed size

A measured search probes one tensor family and bit type at a time against the same BF16 logits, estimates a budget-constrained recipe from the observed changes, and **re-evaluates the combined recipe** before claiming improvement. Delta-KLD estimates are not additive in general; the validated result is the real outcome. A short run is only a smoke test.

~~~bash
q4fit search --model "$BF16" --base-model work/plain-q4.gguf --base-log work/plain-q4-eval.log \
  --corpus corpus/smoke.txt --reference work/bf16-reference.logits --workdir work/search \
  --target-gib 2.12 --groups ffn_down,ffn_gate --types Q3_K,Q5_K --max-trials 4 \
  --quantizer "$QUANT" --ppl "$PPL"
~~~

The search saves each trial's results, final recipe, search-report.json and full evaluation logs. It deletes only temporary GGUF candidates. Output is either a validated best.gguf or a report that no improvement was found among measured options. An absent improvement is a legitimate result.

## Multi-domain layer-range search

After creating BF16 reference logits and plain Q4_K evaluation logs for at least two tuning domains, adapt the [JSON example](examples/search-multi.json) to your local paths.

~~~bash
q4fit-multi --config examples/search-multi.json --workdir work/search-v2 --max-trials 8
~~~

Groups such as `ffn_down@0:17` and `ffn_gate@18:35` change only selected layers of a tensor family. The optimizer minimizes the worst relative KLD across tuning domains under a GGUF byte budget, saves SHA256 receipts, and measures the combined recipe. A tuning pass also requires top-token agreement not to deteriorate beyond the configured allowance. The untouched test set must be evaluated separately.

## Tensor control

Every eligible matrix tensor can be individually assigned through repeatable REGEX=TYPE rules. Exact names are matched using regular expressions; --set '^blk\.0\.attn_v\.weight$=Q6_K' changes only one matching tensor. --set '\.ffn_down\.weight$=Q5_K' changes an entire family. --imatrix is optional, and does not control the allocation policy.

Supported target types: Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16. Only compatible dense 2D tensors are forced to a new type. Other tensors remain governed by llama.cpp.

A probe re-quantizes from the BF16 source, never from an already lossy GGUF, evaluates against BF16 logits, and deletes only its own disposable candidate unless --keep is specified. Receipts, binary paths, config, hashes and raw logs are retained.

## Objective and caveats

Minimize actual next-token KL divergence to BF16 under an output byte budget, not just minimize reconstruction MSE. A single one-chunk, 256-token smoke test checks implementation only. Substantive quality claims need frozen independent multilingual/code/reasoning validation sets and multiple seeds. Model size is not peak VRAM.

Related open projects: llama.cpp, Intel AutoRound, OpenDynamicGGUF, advanced-gguf-quantizer, FIT-GGUF, shoehorn. Q4Fit is not represented as novel algorithmic research. No model weights or third-party training datasets are redistributed.

MIT licensed.
