# Q4Fit research log (v0.1–v0.7)

Archived experiment notes and command examples from Q4Fit v0.1 through v0.7. For current setup and publishing instructions, see the [project README](../README.md).

Experimental CLI workbench for BF16-referenced per-tensor GGUF mixed-precision quantization. The tool uses upstream llama.cpp quantizer and perplexity evaluator, and stores auditable tensor recipes and SHA256 receipts.

**Important result:** The original one-chunk v0.1 optimization **failed to generalize** on a larger WikiText-2 test. At comparable size, v0.1 mixed KLD was 0.125771 versus plain Q4_K 0.110148 (lower is better). See [Benchmark protocol](docs/PROTOCOL.md) for the falsification and larger matched evaluations. Never cite the small-sample result alone as evidence of improvement.

**Earlier v0.3 above-budget Pareto result:** A conservative early-FFN-Down Q5_K upgrade (18 tensors, otherwise Q4_K) has now improved BF16 KLD on two WikiText test segments while adding 2.47% file size, at roughly 1.6% lower single-GPU decode throughput. It is **not within the original 2.12 GiB hard budget**. See [expanded results](benchmarks/QWEN3_4B_EXPANDED.md) and the published [18-tensor recipe](recipes/qwen3-4b-early-down-q5.txt).

Implemented: inspect, build, BF16 baseline/eval, per-tensor probe, group search, multi-domain layer-range search, and v0.3 **prequantized tensor-bank caching, fast GGUF composition, and individual tensor sensitivity scan**. This is an experiment framework, not a claim of new quantization math.

**v0.3 verified:** Precomputed Q3_K/Q4_K/Q5_K tensor banks assembled two Qwen3-4B mixed-bit GGUF models **byte-for-byte identical** to independent upstream llama.cpp quantizations. Fast composition of a ~2.2 GB candidate took 1.56 seconds vs 30–40 seconds for direct quantization (excluding one-time bank creation and BF16 evaluation). See [Cache design and CLI](docs/CACHE.md) and [measurement receipts](benchmarks/QWEN3_4B_CACHE.md).

**v0.4:** `q4fit-tune` performs multi-domain per-tensor scans and approximates byte-budgeted mixed-bit allocation using a beam search. It ranks by worst relative BF16 KL across tuning domains, then independently rebuilds and evaluates combined tensors with actual size and top-token gates. [Methodology and CLI](docs/TENSOR_OPTIMIZER.md). A standalone holdout is mandatory before claiming generalization.

**v0.4 Qwen3-4B measured result:** After a 54-tensor scan, a **seven-tensor Q3_K/Q5_K mix** remained within the original **2.12 GiB** constraint and reduced BF16 KLD on two frozen, independent WikiText/code holdouts by **16.0%/13.7%** versus pure Q4_K. Its bytes were identical to an independent direct llama.cpp quantization. A larger standard Q4_K_S model had worse mean KLD on both sampled holdouts, though some tail/top-token measures favored Q4_K_S. Decode speed on RTX 4090 was approximately **0.4% below** pure Q4_K. [Full report](benchmarks/QWEN3_4B_TENSOR_SEARCH.md), [54 measured tensor probes](benchmarks/QWEN3_4B_TENSOR_SWEEP.csv), [exact recipe](recipes/qwen3-4b-v04-7tensor.txt). One-model PoC only; no broad capability claim.

**v0.5 downstream task check:** The unchanged seven-tensor recipe did **not** show an aggregate task-accuracy advantage over pure Q4_K on a frozen 32-pair English/Chinese MGSM and 24-task MBPP pilot. English MGSM favored Q4Fit (18 vs 14 correct), Chinese favored pure Q4_K (13 vs 17), and MBPP passed 11/24 for both with identical pass/fail items. This contrasts with the earlier BF16-KLD improvements; the 95% clustered bilingual math accuracy-difference interval includes zero. [Full transparent task report](benchmarks/QWEN3_4B_TASK_EVAL.md), [task-level outcomes](benchmarks/QWEN3_4B_TASK_ITEM_RESULTS.csv), [method and corrected protocol](docs/TASK_EVALUATION.md). Original code-prompt interface error and rerun are disclosed. No weights were retuned based on this evaluation.

**v0.6 bounded-thinking replication:** A preregistered, disjoint sample of 48 English/Chinese MGSM pairs and 24 MBPP Python tasks (120 requests/model, four model weights) used Qwen3 thinking with a 256-token reasoning budget. At the original ~2.12GiB size budget, Q4Fit achieved **39/48 EN, 34/48 ZH, 17/24 Python**, versus pure Q4_K **37/48 EN, 35/48 ZH, 15/24 Python**. The bilingual math delta was +1/96 (cluster bootstrap 95% CI −6.25 to +9.38 pp), and the MBPP +2/24 passed exact McNemar p=0.50. **No statistically established task-accuracy gain** despite the earlier KLD advantage. All 480 inference calls had extracted reasoning; truncation was logged. [Frozen protocol](docs/THINKING_EVALUATION.md), [detailed complete report](benchmarks/QWEN3_4B_THINKING_EVAL.md), [120-task outcome CSV](benchmarks/QWEN3_4B_THINKING_ITEM_RESULTS.csv), [statistical data](benchmarks/QWEN3_4B_THINKING_RESULTS.json). The v0.4 tensor recipe remains unchanged.

**v0.7 independent-model replication:** [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), a distinct Phi3-family architecture, was converted from pinned BF16 and independently quantized. After 24 frozen per-tensor probes and 4 independently verified combined candidates, the locked six-tensor Q4Fit recipe added **0.388% bytes** to pure Q4_K, and lowered BF16-reference mean KL on untouched WikiText/code holdouts by **4.59% / 11.60%**. It was byte-for-byte identical to a separate direct upstream quantization. A larger standard Q4_K_S (~8.26% more bytes) achieved substantially better KLD. On 24 paired bilingual MGSM + 16 Python tasks, Q4Fit passed **3/24 EN, 7/24 ZH, 12/16 code**, versus plain Q4_K **2/24 EN, 5/24 ZH, 9/16 code**; none of the individual domain differences was statistically established. [Full report](benchmarks/PHI4MINI_EVAL.md), [frozen plan](benchmarks/PHI4MINI_SEARCH_PLAN.json), [selected recipe](recipes/phi4mini-v07-6tensor.txt), [per-task results](benchmarks/PHI4MINI_ITEM_RESULTS.csv). This is a **cross-family KL replication**, not a universal downstream win. A separately labeled **post-hoc ablation** found Q4_K_S spends 88.6% of its extra bytes on a Q6_K vocabulary embedding; upgrading that tensor alone captures ~92% of its WikiText mean-KLD gain and ~83% of its code mean-KLD gain, but is outside Q4Fit's 0.4% byte cap. [Exploratory ablation](benchmarks/PHI4MINI_ABLATION.json).

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
