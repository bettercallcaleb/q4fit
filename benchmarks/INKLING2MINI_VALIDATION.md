# Inkling 2 Mini — measured Q4Fit MIX versus standard Q4_K

**Conclusion:** The mixed GGUF loads and handles text and basic image input, but this particular Q3/Q4/Q5 recipe has **no demonstrated quality or performance benefit** over standard Q4_K. It is an experimental artifact, not an optimized quant. [Machine-readable results](INKLING2MINI_VALIDATION.json).

## Same-machine speed

Pinned llama.cpp commit `f4f6f2e7db76e98c95cc9b7fc608ea669c806044`, RTX 4090 only, `llama-bench`, 512 prompt tokens, 128 generated tokens, five repeats per run, two runs per model in Q4 → MIX → MIX → Q4 order.

| Metric | Standard Q4_K | Q4Fit MIX |
| --- | ---: | ---: |
| Prefill throughput (tokens/s) | 2,999.55 | 2,994.47 |
| Generation throughput (tokens/s) | 79.05 | 78.84 |
| File size (bytes) | 15,822,563,008 | 15,811,946,176 |

The MIX is ~0.27% slower at generation and 0.067% smaller. These tiny throughput differences should not be interpreted as material.

## BF16-reference comparison

The reference is the **52.23 GB BF16 GGUF upcast** from the officially released, already-quantized NVFP4/MXFP8 weights. It is not an untouched BF16 training checkpoint.

Two text-only sets were frozen and used for earlier Phi-4-mini research; no Inkling-specific prompts were selected. A pinned `llama-perplexity` binary evaluated **20 chunks × 256 context tokens** per set (5,120 positions). The BF16 reference ran with automatic GPU/CPU allocation across RTX 4090, RTX 3060 and system RAM; both quantized GGUFs ran on the RTX 4090. All models used the same evaluator, text, context length and logits format.

| Metric | BF16 upcast reference | Standard Q4_K | Q4Fit MIX |
| --- | ---: | ---: | ---: |
| WikiText perplexity ↓ | 8.7301 | 8.8649 | **8.8545** |
| Code perplexity ↓ | 2.1852 | **2.1999** | 2.2090 |
| WikiText mean KLD to BF16 ↓ | 0 | **0.023035** | 0.023562 |
| Code mean KLD to BF16 ↓ | 0 | **0.014362** | 0.015191 |
| WikiText top-token agreement with BF16 ↑ | 100% | **93.661%** | 93.307% |
| Code top-token agreement with BF16 ↑ | 100% | **96.929%** | 96.142% |

Relative to standard Q4_K, MIX's BF16-reference KLD is **2.29% higher** on WikiText and **5.77% higher** on code. The mixed model has slightly lower WikiText perplexity but higher code perplexity; KLD and perplexity measure different things.

Exploratory paired resampling across 20 chunks gave absolute mean-KLD-difference confidence intervals (MIX minus Q4_K): WikiText **−0.00100 to +0.00179**; code **−0.00026 to +0.00199**. Both include zero. These are small, contiguous text windows, not an independent multi-seed capability benchmark. Neither direction of quality effect is statistically established.

A preceding **6 × 256-token** run yielded WikiText KLD 0.022842 (Q4) versus 0.023117 (MIX), and code 0.014093 (Q4) versus 0.017083 (MIX). The larger-sample code gap is substantially smaller. Do not cite the short run alone.

## Reproduction and provenance

- Original model: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- Input: [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF), revision `e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`
- Input GGUF SHA256: `7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec`
- Standard Q4_K GGUF SHA256: `259aa3d0fd739f6d902abfefcb1030fb3198d87f6d0a8cf830927efcd66fd148`
- MIX GGUF SHA256: `ba344ac332e5a9d91c0f6f7b79b97c1bc470fb408250e7063d53a487ea2333bc`
- [Frozen mixed-precision recipe](../recipes/inkling2mini-v08-mix.txt), [original experiment plan](INKLING2MINI_PILOT_PLAN.json)
- Source test corpora: `work/phi4mini/data/wiki_holdout.txt` and `work/phi4mini/data/code_holdout.txt`, retained locally.
- Measurement logs and logits: `work/inkling2mini/eval/extended20/`; speed logs: `work/inkling2mini/bench-v08/`.
- The [Hugging Face release](https://huggingface.co/ramgpt/Inkling-2-Mini-Q4Fit-MIX-GGUF) has already been published with an experimental designation, not a superiority claim.

Benchmark commands, with model paths adapted to your workstation:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-bench -m Inkling-2-Mini-Q4_K-PURE.gguf -p 512 -n 128 -r 5 -ngl 99 -t 8 -o json
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-bench -m Inkling-2-Mini-Q4Fit-MIX.gguf -p 512 -n 128 -r 5 -ngl 99 -t 8 -o json
CUDA_VISIBLE_DEVICES=0,1 ./llama.cpp/build/bin/llama-perplexity -m Inkling-2-Mini-BF16.gguf -f wiki_holdout.txt -c 256 -b 256 -ub 256 --chunks 20 -ngl auto -sm layer --save-all-logits wiki.logits
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-perplexity -m Inkling-2-Mini-Q4Fit-MIX.gguf -f wiki_holdout.txt -c 256 -b 256 -ub 256 --chunks 20 -ngl 99 --kl-divergence --kl-divergence-base wiki.logits
```

The four-tensor MIX is not justified as the default download. Prefer the ordinary Q4_K if its size and hardware requirements are acceptable.
