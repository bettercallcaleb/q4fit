# Phi-4-mini-instruct: Cross-family Q4Fit replication (v0.7)

**2026-11-23 — Complete, frozen single-model study.** Q4Fit improves mean BF16-reference KLD on two held-out corpora of a non-Qwen model under a +0.4% GGUF-byte budget. The pilot does **not** establish a statistically significant task-accuracy advantage.

## Provenance and separation of selection from testing

- Official source: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (Phi3ForCausalLM, 32 layers); pinned Hugging Face revision `cfbefacb99257ffa30c83adab238a50856ac3083`. Original BF16 converted with llama.cpp to an **7,680,694,080-byte GGUF**; SHA256 `4a967298f4692c596f78c3e19d0f0e6f69f56e353e61389d738bb41b42d784a1`.
- [Full search plan](PHI4MINI_SEARCH_PLAN.json) and [new task IDs](PHI4MINI_TASK_MANIFEST.json) were frozen in commit [`9e86543`](https://github.com/bettercallcaleb/q4fit/commit/9e86543) before searching this model. The 24 paired English/Chinese MGSM and 16 MBPP tasks exclude all v0.5/v0.6 question IDs.
- [Chosen six-tensor model](PHI4MINI_SELECTED.json) and [exact recipe](../recipes/phi4mini-v07-6tensor.txt) were separately committed as [`a4eb894`](https://github.com/bettercallcaleb/q4fit/commit/a4eb894) **before evaluating any holdout or downstream task**.
- 24 individual scans (14 FFN Down Q5_K upgrades, 10 FFN Up/attention Q3_K reductions); two fixed tuning domains, WikiText-2 raw validation and Python source, 6×256-token chunks each. Four independently composed candidates were actually evaluated; minimax **observed** relative KLD determined selection. All inputs and holdout SHA256 are in the frozen plan. No imatrix.

## Selected mix and exact byte verification

| Model | GGUF bytes | Versus pure Q4_K |
|---|---:|---:|
| BF16 original | 7,680,694,080 | reference precision |
| Pure Q4_K (all eligible tensors Q4_K) | 2,166,699,840 | reference size |
| **Q4Fit six-tensor mix** | **2,175,104,832** | **+0.388%** |
| Standard llama.cpp Q4_K_S | 2,345,597,760 | +8.26% |

The **frozen hard cap** was 2,175,366,639 bytes (+0.4% vs plain Q4_K), leaving 261,807 bytes of headroom for the selected candidate. All other eligible tensors remain Q4_K.

~~~text
blk.0.attn_qkv.weight=q3_k
blk.0.ffn_down.weight=q5_k
blk.1.ffn_down.weight=q5_k
blk.2.attn_qkv.weight=q3_k
blk.3.ffn_down.weight=q5_k
blk.7.ffn_down.weight=q5_k
~~~

Final GGUF SHA256 `e406e04c7f8b4cc74942d126436595f5b11bdfa1f1125ad7578ff0a2df985661`. Independent, direct llama.cpp `--pure --tensor-type-file` BF16 quantization produced a **byte-identical** file and the same SHA256; the cached composition was independently validated.

## KLD evidence

| Corpus (6×256 tokens) | Pure Q4_K KLD ↓ | Q4Fit KLD ↓ | Relative improvement |
|---|---:|---:|---:|
| **Tuning** WikiText validation | 0.432330 | 0.419072 | −3.07% |
| **Tuning** Python code | 0.201518 | 0.173133 | −14.09% |
| **Holdout** WikiText test slice | 0.504851 | 0.481667 | **−4.59%** |
| **Holdout** independent Python code | 0.262617 | 0.232166 | **−11.60%** |

On the **independent holdouts**, BF16 top-token agreement rose from 61.942% to 62.336% (WikiText) and 80.840% to 81.890% (code). The 99th-percentile WikiText KLD **worsened slightly**, 1.545369 to 1.553318, while code p99 improved 1.391777 to 1.346947. Thus the result is a *mean* KLD benefit, not universal dominance.

**Larger standard Q4_K_S is a strong counterexample**: for 8.26% larger bytes than pure Q4_K, its independent WikiText KLD was 0.152122 and code KLD 0.069703, far below Q4Fit. That is not a comparable byte budget, and should not be presented as one.

## Independent actual-task outcomes

24 English MGSM math problems, the same 24 translated into Chinese, and 16 MBPP Python problems, all disjoint from previous Qwen evaluations. Zero-shot non-thinking, greedy temperature 0, top_k 1, fixed seed 2026112307, 256 generated-token math cap and 400-token code cap. For math, an explicit FINAL numerical line is required (optional terminal period allowed for every model). Python code is graded against official asserts in an unprivileged, network-disabled, read-only Podman sandbox; exact function name/interface is supplied but test contents are hidden. All 16 gold code solutions passed the sandbox checks.

| Model | English /24 | Chinese /24 | Python /16 | Descriptive total /64 |
|---|---:|---:|---:|---:|
| BF16 | 5 | 9 | 13 | 27 |
| Pure Q4_K | 2 | 5 | 9 | 16 |
| **Q4Fit** | **3** | **7** | **12** | **22** |
| Q4_K_S | 5 | 7 | 11 | 23 |

Compared with pure Q4_K, Q4Fit was +1 English, +2 Chinese, +3 Python. Exact paired McNemar p-values were **1.0, 0.625 and 0.25**, respectively: **none is statistically significant**. The +3/48 bilingual-math accuracy shift yields a source-question-cluster bootstrap **95% CI of −4.17 to +18.75 percentage points**. BF16's low zero-shot English-math result (5/24) further limits capability conclusions. Do not pool the three domains as independent correctness observations.

## RTX 4090 inference speed

Single GPU, same CUDA llama-bench executable, `-p 512 -n 128 -r 5 -ngl 999`.

| GGUF | Prompt pp512 (tokens/s) | Generation tg128 (tokens/s) |
|---|---:|---:|
| Pure Q4_K | 23,180.80 ± 2,051.21 | **316.20 ± 0.56** |
| Q4Fit | 23,819.39 ± 1,999.57 | 315.01 ± 0.65 |
| Q4_K_S | 22,341.07 ± 1,813.59 | 298.15 ± 0.42 |

Q4Fit generated ~0.38% fewer tokens/sec than pure Q4_K. Prompt processing uncertainties overlap; no prompt-speed gain is established.

## Post-hoc mechanism diagnostic (not preregistered)

After seeing that standard Q4_K_S has much lower mean KLD, inspect GGUF tensor dtypes. Compared with pure Q4_K, Q4_K_S changes **exactly nine tensors**: `token_embd.weight` Q4_K → Q6_K (+158,450,688 bytes), plus four FFN Down and four merged attention QKV tensors (layers 0–3) Q4_K → Q5_K (+20,447,232 bytes). The Q6 embedding accounts for 88.57% of Q4_K_S's additional file bytes.

Two *new* BF16-to-GGUF ablation builds tested these two groups separately on the **already exposed holdouts**. These are diagnostic rather than confirmatory results.

| Model | Actual GGUF bytes | WikiText holdout KLD ↓ | Code holdout KLD ↓ |
|---|---:|---:|---:|
| Pure Q4_K | 2,166,699,840 | 0.504851 | 0.262617 |
| Only 8 early Q5_K upgrades | 2,187,147,072 | 0.477308 | 0.224100 |
| Only token embedding Q6_K | 2,325,150,528 | **0.180764** | **0.102948** |
| Both groups, upstream Q4_K_S | 2,345,597,760 | **0.152122** | **0.069703** |

Applying the **embedding upgrade alone** produces 91.9% of the full Q4_K_S WikiText mean-KLD reduction and 82.8% of its code reduction (ratios of observed differences, not additive causal contributions). The eight smaller upgrades also help; combined changes need not be additive. The large vocabulary embedding is therefore a high-value *post-hoc hypothesis* for future size-allocation experiments, not a newly validated search discovery. Its +158MB cost is far outside the preregistered +0.4% Q4Fit cap.

All tensor-type changes, quantized GGUF hashes, corpus results and this caveat are recorded in [the ablation JSON](PHI4MINI_ABLATION.json).

## Interpretation and reusable evidence

The earlier Qwen3-4B experiment reported independent mean-KLD gains, but lacked robust downstream-task effects. This Phi-4-mini follow-up supports **cross-architecture feasibility** of the small-budget mixed-bit optimizer: both prespecified holdout corpora improved and upstream direct quantization reproduced exact bytes. It does **not** demonstrate better overall performance than standard larger quantizers or reliable task correctness improvement. No model-specific new tensor search should be performed on these already-revealed holdouts.

- [Complete numeric evidence, model hashes and paired inferences](PHI4MINI_RESULTS.json)
- [64-row per-task boolean outcomes](PHI4MINI_ITEM_RESULTS.csv)
- [Frozen plan](PHI4MINI_SEARCH_PLAN.json), [preselected task manifest](PHI4MINI_TASK_MANIFEST.json), [pre-holdout selected recipe](PHI4MINI_SELECTED.json)
- Model weights, BF16 logit matrices and raw generated answers are stored only under gitignored local work directories.
