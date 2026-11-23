# Inkling 2 Mini: first Q4Fit mixed GGUF

This is a working, experimental mixed-precision version of [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), not a claim that Q4Fit improves the model's accuracy.

## Files and checks

| Output | Actual bytes | Size |
| --- | ---: | ---: |
| Standard Q4_K from the same source | 15,822,563,008 | 15.823 GB |
| Q4Fit Q3/Q4/Q5 mix | 15,811,946,176 | 15.812 GB |

The mix is **10,616,832 bytes (0.067%) smaller** than pure Q4_K.

- Mixed GGUF SHA256: `ba344ac332e5a9d91c0f6f7b79b97c1bc470fb408250e7063d53a487ea2333bc`
- Pure Q4_K SHA256: `259aa3d0fd739f6d902abfefcb1030fb3198d87f6d0a8cf830927efcd66fd148`
- Input GGUF SHA256: `7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec`
- Frozen build plan: [INKLING2MINI_PILOT_PLAN.json](INKLING2MINI_PILOT_PLAN.json), committed as [aae008c](https://github.com/bettercallcaleb/q4fit/commit/aae008c) before testing
- llama.cpp commit: `f4f6f2e7db76e98c95cc9b7fc608ea669c806044`

## What changed?

The base is `Q4_K` with four exact tensor overrides ([recipe](../recipes/inkling2mini-v08-mix.txt)):

| Tensor | Type |
| --- | --- |
| `blk.0.ffn_gate_up_exps.weight` | Q3_K |
| `blk.1.ffn_gate_up_exps.weight` | Q3_K |
| `blk.6.ffn_gate_up_exps.weight` | Q5_K |
| `blk.12.ffn_gate_up_exps.weight` | Q5_K |

The model repeats blocks 6–17 three times. The experiment trades precision from blocks 0 and 1 for extra precision in two repeated blocks, without increasing file size. The allocation was chosen from architecture, **not from a measured tensor search**.

Actual tensor types in the 704-tensor GGUF: **198 Q4_K, 24 Q5_0, 2 Q3_K, 2 Q5_K, and 478 F32**. The 24 Q5_0 tensors are automatic llama.cpp fallbacks: expert down-projection rows of 1,152 elements are incompatible with 256-element Q4_K blocks.

Q4Fit v0.8 supports compatible three-dimensional MoE expert tensors and estimates the fallback byte cost. Actual file size remains the authoritative number.

## Does it work?

Yes. Both the pure Q4_K and mixed files loaded successfully on an **RTX 4090** using the pinned recent llama.cpp build. Eight short prompts involving arithmetic, translation, Linux, Python, JSON, English and Chinese returned correct answers from both; formatting differed slightly. The mixed model also identified a red square using the unchanged BF16 vision projector.

These are basic functional checks. **No representative KLD benchmark, long-context benchmark, reasoning test or broad accuracy improvement has been established.**

## Source and reproduction

- Original: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache 2.0
- GGUF source: [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF), pinned revision `e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`
- Source BF16 GGUF: 52,226,509,504 bytes

**The ggml-org BF16 GGUF is an upcast of the officially released NVFP4/MXFP8 weights, not a pristine original BF16 checkpoint.** The comparison is against that same upcast source, not against hypothetical original unquantized weights.

The model was built with:

```bash
q4fit build \
  --model Inkling-2-Mini-BF16.gguf \
  --quantizer /path/to/llama.cpp/build/bin/llama-quantize \
  --default Q4_K \
  --set '^blk\.0\.ffn_gate_up_exps\.weight$=Q3_K' \
  --set '^blk\.1\.ffn_gate_up_exps\.weight$=Q3_K' \
  --set '^blk\.6\.ffn_gate_up_exps\.weight$=Q5_K' \
  --set '^blk\.12\.ffn_gate_up_exps\.weight$=Q5_K' \
  --output Inkling-2-Mini-Q4Fit-MIX.gguf
```

The completed GGUF and original projector are staged locally in `work/inkling2mini/publish/` for a Hugging Face upload. No public model repository was created, because Hugging Face write authentication is not configured on the workstation.
