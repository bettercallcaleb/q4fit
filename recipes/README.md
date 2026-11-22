# Reproduce a selected tensor recipe

The checked-in recipe specifies exact GGUF tensor-name/type overrides. It contains no model weights.

Run from repository root with a compatible recent llama.cpp build and the exact same Qwen3-4B BF16 source (SHA256 `c4e947bd9d399a8ad87cc210db04fbddc916d8a39f54957235391674a8811ca4`):

~~~bash
llama-quantize --pure --tensor-type-file recipes/qwen3-4b-early-down-q5.txt /models/qwen3-4b-bf16.gguf /work/early-down-q5.gguf q4_k 12
~~~

The tested GGUF produced SHA256 `4c26092b49095c2a7a57d39f386a237682583411530f938436a301f7c8e4534a`. Matching output bytes depends on quantizer version and model metadata; pin the same upstream build if comparing exact hashes. The associated benchmark is in `benchmarks/QWEN3_4B_EXPANDED.md`.

## v0.4 seven-tensor under-budget allocation

This experiment's exact tensor-type list is `qwen3-4b-v04-7tensor.txt`.

~~~bash
llama-quantize --pure --tensor-type-file recipes/qwen3-4b-v04-7tensor.txt /models/qwen3-4b-bf16.gguf /work/qwen3-4b-q4fit-v04.gguf q4_k 12
~~~

Tested output: 2,275,628,384 bytes, SHA256 `7292a820225eec6af4e7e544fc0d5de8ac31cbe4486082e20832521cdc935450`. See [full frozen-holdout results](../benchmarks/QWEN3_4B_TENSOR_SEARCH.md). The seven overrides are measured for one specific BF16 model and should not be presumed to transfer to another model or quantizer version.
