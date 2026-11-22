# Reproduce a selected tensor recipe

The checked-in recipe specifies exact GGUF tensor-name/type overrides. It contains no model weights.

Run from repository root with a compatible recent llama.cpp build and the exact same Qwen3-4B BF16 source (SHA256 `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`):

~~~bash
llama-quantize --pure --tensor-type-file recipes/qwen3-4b-early-down-q5.txt /models/qwen3-4b-bf16.gguf /work/early-down-q5.gguf q4_k 12
~~~

The tested GGUF produced SHA256 `4c26092b49095c2a7a57d39f386a237682583411530f938436a301f7c8e4534a`. Matching output bytes depends on quantizer version and model metadata; pin the same upstream build if comparing exact hashes. The associated benchmark is in `benchmarks/QWEN3_4B_EXPANDED.md`.
