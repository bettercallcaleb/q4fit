# Qwen3-4B v0.4 — measured tensor allocation under 2.12 GiB

Date: 2026-11-23. Experimental one-model PoC. Source: Qwen3-4B BF16 GGUF, 398 tensors, SHA256 `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`. No imatrix. Baseline: pure Q4_K, 2,269,269,344 bytes. Hard size budget: **2,276,332,666 bytes** (2.12 GiB).

## Protocol and selection

Precomputed Q3_K/Q4_K/Q5_K banks from the original BF16; reuse of exact tensor byte ranges through `q4cache.py`. All 54 predeclared single-tensor probes were evaluated on two tuning domains, each against frozen BF16 next-token distributions:

- WikiText-2 **raw validation**, 16 × 512 tokens, corpus SHA256 `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`.
- Q4Fit pre-v0.4 Python code source, 8 × 512 tokens, SHA256 `e6ef9fe3f2caeec31e89968c4fab131e79a0620021675ed7df37ec4cb74c423b`.

The scan tested **18 distinct Q5_K upgrades** in early FFN Down and **36 Q3_K reductions** across FFN Gate, FFN Up, Attention Q, Attention Output, Attention K and V. Four Q5_K upgrades independently reduced mean KLD in *both* tuning domains. None of the 36 Q3_K reductions independently reduced KLD in both. Full per-tensor results: [54-row CSV](QWEN3_4B_TENSOR_SWEEP.csv).

The heuristic used a byte-bucketed, max-eight-edit beam search (width 5,000), minimizing the worst additive **relative** KL change across domains subject to actual indexed bank byte deltas. It also proposed two simple Q5-only anchors. Six combined candidates were reconstructed and evaluated independently; all six passed tuning-only gates. The selection was determined by **measured**, not predicted, worst-domain KLD ratio. Scan wall time: **481.55s**; optimizer including full checks and six evaluations: **96.29s**. These are single-host experimental timings.

## Selected mixed-precision recipe

[Exact seven-tensor override file](../recipes/qwen3-4b-v04-7tensor.txt)

| Target type | Tensor names |
|---|---|
| Q5_K | `blk.1.ffn_down.weight`, `blk.6.ffn_down.weight`, `blk.11.ffn_down.weight`, `blk.16.ffn_down.weight` |
| Q3_K | `blk.1.ffn_up.weight`, `blk.0.attn_output.weight`, `blk.5.attn_q.weight` |
| Q4_K | All other eligible tensors |

Serialized GGUF: **2,275,628,384 bytes (2.11934 GiB)**, only **704,282 bytes** under the budget. SHA256 `7292a820225eec6af4e7e544fc0d5de8ac31cbe4486082e20832521cdc935450`. An independent `llama-quantize --pure --tensor-type-file` pass from BF16 generated the **identical full GGUF SHA256** (36.30s quantizer time). The cache-generated model and the direct quantizer model are byte-identical for this recipe.

## Tuning-domain results (not holdout)

| Metric | Pure Q4_K | Selected |
|---|---:|---:|
| WikiText valid mean KLD ↓ | 0.099345 | **0.083499** (−15.9%) |
| WikiText valid BF16 top-token agreement ↑ | 87.917% | **89.020%** |
| Code mean KLD ↓ | 0.083428 | **0.065311** (−21.7%) |
| Code BF16 top-token agreement ↑ | 92.647% | **92.990%** |

Single-tensor KL deltas were **not** additive: the beam's best predicted recipe was not the best actually measured recipe. Selecting solely by the additive prediction would have chosen a different model.

## Frozen independent holdouts

The holdouts were fixed while the single-tensor scan was running, **before combined-recipe ranking, verification or any holdout candidate measurement**. They did not influence search rules, byte budget, or candidate choice. WikiText test text came from a source-character slice between 32% and 50% of the raw test file, without overlap with earlier sampled beginning or tail segments (16 × 512 tokens). Independent code used v0.3 files `q4cache.py`, `q4scan.py`, and `tests/test_cache.py` (8 × 512 tokens), disjoint from the tuning code set. Holdout corpus SHA256: WikiText `9163f19687349a1c5fceccf2a580feec95b749bc9f291e9c8cadc5526eeae8d5`; code `2c3f4b988150d90b343090198d8d05128dcade8e539446b927504b90019f856e`.

| Model | GGUF bytes | WikiText holdout KLD ↓ | Code holdout KLD ↓ |
|---|---:|---:|---:|
| Pure Q4_K | 2,269,269,344 | 0.121471 | 0.056639 |
| **Q4Fit v0.4 selected** | **2,275,628,384** | **0.102051** | **0.048889** |
| Standard Q4_K_S | 2,383,309,664 | 0.110088 | 0.051099 |
| Standard Q4_K_M | 2,497,280,864 | 0.088310 | 0.036757 |

Relative to pure Q4_K, the selected recipe reduced **holdout mean KLD by 16.0% on WikiText and 13.7% on code**. BF16 top-token agreement increased from 87.745% to 88.407% (WikiText) and from 92.157% to 92.941% (code). The 99th-percentile KLD also fell: 1.054519 → 0.999119 (WikiText) and 0.661441 → 0.611104 (code). Both predefined holdout gates passed.

The selected model is **4.52% smaller** than standard Q4_K_S and showed lower *mean KLD* on both sampled holdouts. It does not strictly dominate Q4_K_S on every metric: for example, Q4_K_S achieved higher WikiText holdout top-token agreement and lower WikiText p99 KLD. Q4_K_M is substantially larger and achieved lower mean KLD on both holdouts.

## RTX 4090 single-GPU inference benchmark

Same host, llama-bench build `70adb1b4c`, `CUDA_VISIBLE_DEVICES=0`, `-ngl 999`, 5 repetitions, 512 prompt tokens / 128 generated tokens. Higher token/s is better.

| Model | Prompt pp512 tokens/s | Decode tg128 tokens/s |
|---|---:|---:|
| Pure Q4_K | 19,326.23 ± 2,176.90 | 279.83 ± 0.38 |
| Q4Fit v0.4 selected | 19,256.28 ± 1,946.57 | **278.71 ± 0.49** |
| Standard Q4_K_S | 18,385.85 ± 1,834.35 | 270.60 ± 0.50 |

Selected decoder throughput was approximately **0.4% below** pure Q4_K, a small measured tradeoff. Prompt throughput uncertainty overlaps strongly. No claim is made about longer contexts, multi-batch throughput, or other GPUs.

## Scope and reproducibility limitations

This is one Qwen3-4B source, limited WikiText and code slices, one deterministic GGUF/llama.cpp build and no replicated seeds. The 54-probe selection creates multiple-comparison/overfitting risk, and an independent holdout success on two small domains does **not** establish multilingual, safety, instruction-following, reasoning, or downstream-task superiority. Nor is the beam result a globally optimal tensor allocation. Cross-model tests, independent task suites, repeated selection/evaluation, and matched-size comparisons remain necessary before scientific claims.

Public source includes the deterministic recipe, all 54 single-tensor tabular measurements, search code and protocol. Private/local model weights, caches, raw logits and text corpora are intentionally not committed.
