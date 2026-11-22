# Q4Fit v0.3 Cached Tensor-Bank Validation

Date: 2026-11-23. Qwen3-4B BF16 GGUF, source SHA256 `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`. Host Ubuntu 24.04, NVIDIA RTX 4090/3060; llama.cpp upstream local CPU quantizer and CUDA perplexity evaluator. No imatrix.

## Bank precomputation

| Bank type | GGUF size | Note |
|---|---:|---|
| Q3_K | 1.616 GiB | New bank |
| Q4_K | 2.113 GiB | Reused previous plain-Q4_K baseline |
| Q5_K | 2.582 GiB | New bank |

Q3 and Q5 were quantized once from the same 7.49-GiB BF16 source via `llama-quantize --pure`. Their combined additional disk footprint is approximately **4.2 GiB**. Q4 bank was reused without copying. The GGUF bank index stores original source hash, bank hashes and file stats, tensor dimensions/offsets/types, the base header fingerprint and descriptor locations. It rejects mismatched provenance or incompatible metadata. Default Q4_K; GGUF v3 little endian, single-file only.

## Two byte-identical full GGUF reconstructions

| Recipe | Recompose SHA256 | Direct llama.cpp SHA256 |
|---|---|---|
| FFN Down layers 0–17 Q5_K (18 overrides); rest Q4_K | `4c26092b49095c2a7a57d39f386a237682583411530f938436a301f7c8e4534a` | identical |
| All FFN Down Q5_K plus all FFN Gate Q3_K (72 overrides); rest Q4_K | `0505678cf84cd1b09d653ba4c0423fc1a189b145df908201b8f88c4ecdc780e7` | identical |

Both full GGUF byte streams were compared with `cmp` to independently created earlier models. This supports exact compatibility for these recipes, banks, and the tested llama.cpp build; it does not prove every supported architecture or imatrix mode will match.

## Candidate generation benchmark

| Stage | Wall-clock time |
|---|---:|
| Direct full-model mixed llama.cpp quantization | 30–40 seconds |
| Cached fast compose (single-tensor override, GGUF ~2.2 GB) | **1.67 seconds**, of which reported compose was **1.564 seconds** |

Approximate candidate generation speedup: 18–24×, excluding one-time bank creation, model evaluation, and final full-file verification. Verification mode reads the result again and computes SHA256, so takes longer. The fast path writes atomically, checks cache size and mtime, checks the original Q4 header SHA, and leaves output hashes unset until validated.

## Per-tensor sensitivity scan

Frozen evaluation: code text, eight 512-token chunks; BF16 reference logits. Plain Q4_K mean KL = **0.083428**. Each scan candidate changes exactly one FFN Down tensor from Q4_K to Q3_K.

| Tensor | Delta GGUF size | KL divergence | Delta KL |
|---|---:|---:|---:|
| `blk.0.ffn_down.weight` | −3.154 MiB | 0.085642 | +0.002214 |
| `blk.1.ffn_down.weight` | −3.154 MiB | 0.085292 | +0.001864 |
| `blk.2.ffn_down.weight` | −3.154 MiB | 0.085760 | +0.002332 |
| `blk.3.ffn_down.weight` | −3.154 MiB | 0.087894 | +0.004466 |

All four tensors degraded KL on this code corpus. The scan used the precomputed bank, saved individual raw logs and CSV/JSON scores, deleted disposable GGUF candidates and completed in **18.69 seconds total** including GPU evaluations. No claim of general model quality from this short test.

## Multi-domain search integration

The cached search evaluated FFN Down layers 0–17 Q5_K and FFN Up layers 0–17 Q3_K, using the original WikiText validation and code domains. Cached metrics reproduced earlier direct-quantization scores exactly to the six displayed decimal places: (0.084527, 0.062109) and (0.124308, 0.090769), respectively. Under the **2.12-GiB** strict budget, the selector again chose the unmodified Q4_K baseline, preserving the negative result.

No full model weights, bank GGUF files, source logits or copyrighted test data are included in the public source repo. Reproduction requires the same BF16 source, quantizer version, and text data.
