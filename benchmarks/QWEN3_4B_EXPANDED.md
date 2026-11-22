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
