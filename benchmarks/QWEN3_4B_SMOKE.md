# Qwen3-4B BF16-referenced mixed-bit smoke test

Date: 2026-11-23. Experimental CLI validation; these small samples are **not** a production benchmark, paper result, or claims about Inkling 2 Mini.

Source model: Qwen3-4B BF16 GGUF, 398 tensors, 8,051,285,344 bytes. SHA256: `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`.

Method: llama.cpp `llama-quantize --pure --tensor-type-file` from original BF16 for each candidate; llama-perplexity `--save-all-logits` on BF16 and `--kl-divergence --kl-divergence-base` on variants. No imatrix. Runtime: NVIDIA RTX 4090, CUDA. Size is serialized GGUF file size.

## Search text (one 256-token chunk)

| Configuration | GGUF bytes | Mean KLD vs BF16 | Same top token |
|---|---:|---:|---:|
| Plain Q4_K | 2,269,269,344 | 0.085821 | 90.551% |
| FFN Down Q3_K, rest Q4_K | 2,150,198,624 | 0.082382 | 91.339% |
| FFN Down Q5_K, rest Q4_K | 2,381,335,904 | 0.067136 | 88.976% |
| FFN Gate Q3_K, rest Q4_K | 2,150,198,624 | 0.076767 | 89.764% |
| FFN Gate Q5_K, rest Q4_K | 2,381,335,904 | 0.077394 | 88.189% |
| **Search choice: FFN Down Q5_K + FFN Gate Q3_K, remaining Q4_K** | **2,262,265,184** | **0.060225** | **92.913%** |

Search budget: **2.12 GiB**. Only two tensor families and Q3_K/Q5_K were searched; no claim of global optimum. The selected final model was **freshly quantized from BF16 and empirically evaluated**; the combinatorial estimated KLD was 0.058082 and the measured KLD was 0.060225, demonstrating non-additivity.

## Independent authored holdout (two 512-token chunks)

| Configuration | Mean KLD vs BF16 | 99th-percentile KLD | Same top token |
|---|---:|---:|---:|
| Plain Q4_K | 0.084034 | 0.432001 | 85.686% |
| Selected mixed Q4 | **0.079793** | **0.338358** | 84.706% |

Holdout KLD improved by about **5.0%**, but top-token agreement fell by **0.98 percentage points**. Thus no across-the-board superiority claim is justified. The holdout was authored separately and was not used to choose bits. Both text samples are very small and conclusions must be independently replicated on a larger frozen corpus and downstream tasks.

To reproduce, set paths for your own BF16 weights and llama.cpp binaries and follow the README. For best.gguf provenance, `work/search/search-report.json` is retained locally and excluded from Git by design. Model weights are **not distributed** with the source.
