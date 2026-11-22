# Qwen3-4B Expanded Validation — 2026-11-23

Model: Qwen3-4B BF16 GGUF; 398 tensors. Source SHA256 `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`. All candidates quantized independently from original BF16 using local llama.cpp. No imatrix.

## External holdout: WikiText-2 raw TEST (first 24 x 512 tokens)

| Model | GGUF size (bytes) | Mean KL vs BF16 | Top-1 agreement |
|---|---:|---:|---:|
| Pure Q4_K | 2,269,269,344 | 0.110148 | 87.271% |
| v0.1 mixed (down Q5, gate Q3) | 2,262,265,184 | 0.125771 | 86.928% |
| Standard Q4_K_S | 2,383,309,664 | 0.095972 | 88.399% |
| Standard Q4_K_M | 2,497,280,864 | 0.072403 | 90.294% |

**The v0.1 mixed recipe failed to generalize**. Q4_K_S and Q4_K_M use more bytes, so these comparisons describe the quality-size tradeoff, not an equal-size win.

## Code domain (8 x 512 tokens, Q4Fit source files)

| Model | Mean KL vs BF16 | Top-1 agreement |
|---|---:|---:|
| Pure Q4_K | 0.083428 | 92.647% |
| v0.1 mixed | 0.085674 | 92.402% |
| Standard Q4_K_S | 0.076870 | 93.431% |
| Standard Q4_K_M | 0.057964 | 94.265% |

## v0.2 measured search: WikiText-2 validation and code

Tuning domains: WikiText-2 raw validation, 16 x 512 tokens (SHA256 `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`); code source, 8 x 512 tokens (SHA256 `e6ef9fe3f2caeec31e89968c4fab131e79a0620021675ed7df37ec4cb74c423b`).

Plain Q4_K KLD: WikiText validation **0.099345**, code **0.083428**. Tested eight tensor-family-by-layer-half Q3/Q5 variations involving `ffn_down` and `ffn_gate`. The minimax relative-KLD search with a **2.12 GiB** hard budget selected plain Q4_K: no acceptable balanced improvement in that search space.

The result motivated a **predeclared** follow-up testing Q3 reductions in FFN Up / Attention to offset the consistently positive contribution of Q5 in early FFN Down layers. The later portion of WikiText TEST was held separately for final evaluation.

Neither the 8-candidate experiment nor the follow-up establishes global optimality. No trained parameters, source model weights, or copied WikiText test content are redistributed.

## v0.2 targeted compensation search

A second predeclared five-trial search kept early FFN Down Q5_K as the candidate upgrade and separately tested Q3_K for FFN Up layers 0–17, FFN Up layers 18–35, all Attention Q layers, and all Attention Output layers. At the same 2.12 GiB budget, the minimax multi-domain selector again chose plain Q4_K.

| Isolated change from plain Q4_K | WikiText validation KLD | Code KLD |
|---|---:|---:|
| FFN Down 0–17 Q5_K | **0.084527** | **0.062109** |
| FFN Up 0–17 Q3_K | 0.124308 | 0.090769 |
| FFN Up 18–35 Q3_K | 0.133792 | 0.099397 |
| Attention Q 0–35 Q3_K | 0.114821 | 0.094472 |
| Attention Output 0–35 Q3_K | 0.117937 | 0.095748 |

All Q3_K compensations introduced enough KL degradation to prevent a validated under-budget win among these tested combinations. A controlled one-upgrade Pareto candidate (early FFN Down Q5_K) is evaluated separately, above the strict 2.12 GiB budget. Results should not be conflated with an equal-size improvement.

## Validated Pareto point above the strict budget

One single-family upgrade was evaluated without compensating Q3 reductions: **FFN Down layers 0–17 Q5_K, all other eligible weights Q4_K**. Serialized size: **2,325,302,624 bytes** (2.166 GiB), **+2.47% vs plain Q4_K**, below standard Q4_K_S's 2,383,309,664 bytes.

| Model | WikiText test head KLD (24 × 512) | WikiText test tail KLD (16 × 512) | Tail top-1 |
|---|---:|---:|---:|
| Plain Q4_K | 0.110148 | 0.120019 | 85.368% |
| Early FFN Down Q5_K | **0.084098** | **0.097860** | **86.642%** |
| Standard Q4_K_S | 0.095972 | 0.112718 | 86.029% |
| Standard Q4_K_M | 0.072403 | 0.079459 | 88.137% |

The tail consists of the raw WikiText-2 test file from the first line after approximately 60% of source characters, with SHA256 `c6cdfd04d3e293ca728dbaaa21e383d6957e4d8b095d75ebb2fef0c0c06d3de3`, not used during search. The early Down Q5 model improved mean KLD by 23.6% on the previously examined test-head subset and 18.5% on the separately held test-tail subset. It also improved tail top-token agreement by 1.27 percentage points. This is a **limited one-model empirical tradeoff**, not proof of superiority on broader tasks or models.

## 4090-only runtime

llama-bench build `70adb1b4c` (10588); `CUDA_VISIBLE_DEVICES=0`, `-ngl 999`, `-p 512`, `-n 128`, 3 repetitions; same host and run. Higher t/s is better.

| Model | pp512 tok/s | tg128 tok/s |
|---|---:|---:|
| Plain Q4_K | 18,393.82 ± 2,688.64 | 279.25 ± 0.50 |
| Early FFN Down Q5_K | 18,313.57 ± 2,415.22 | 274.88 ± 0.56 |
| Standard Q4_K_S | 17,672.13 ± 2,528.12 | 270.17 ± 0.73 |

The early Down Q5 choice is about 1.6% slower than plain Q4_K in this short decode benchmark. Prompt throughput results overlap substantially; treat speed observations as workload-specific, not universal.

The exact 18-tensor recipe is published in `recipes/qwen3-4b-early-down-q5.txt`. Result GGUF SHA256: `4c26092b49095c2a7a57d39f386a237682583411530f938436a301f7c8e4534a`. Weights are intentionally excluded from GitHub.
