# Qwen3-4B v0.5 — frozen downstream task benchmark

Date: 2026-11-23. **Outcome: no demonstrated aggregate task-accuracy improvement** from the v0.4 seven-tensor allocation, despite its previously validated BF16 KLD improvement. The fixed mixed-precision recipe was not retuned on these tasks.

## Model comparison

| GGUF | File bytes | MGSM English (n=32) | MGSM Chinese (n=32) | MBPP sanitized Python (n=24) |
|---|---:|---:|---:|---:|
| BF16 reference | 8,051,285,344 | 11/32 (34.4%) | 9/32 (28.1%) | 10/24 (41.7%) |
| Pure Q4_K | 2,269,269,344 | 14/32 (43.8%) | **17/32 (53.1%)** | 11/24 (45.8%) |
| **Q4Fit v0.4, 7-tensor** | **2,275,628,384** | **18/32 (56.3%)** | 13/32 (40.6%) | 11/24 (45.8%) |
| Standard Q4_K_S | 2,383,309,664 | 13/32 (40.6%) | 13/32 (40.6%) | 11/24 (45.8%) |

Compared with pure Q4_K, Q4Fit is **6,359,040 bytes (0.280%) larger**, within the original 2.12 GiB limit. It answered 4 more English math questions correctly but **4 fewer** Chinese translations of the same sampled problems. Across 32 source questions × two languages, both models answered **31 of 64** correctly. Exactly **11/24 MBPP tasks** passed for each model, and all 24 Python pass/fail outcomes are identical. Generated source text differed on **9/24** tasks, yet neither gained any additional passing case.

## Paired uncertainty: Q4Fit minus plain Q4_K

| Domain | Accuracy difference | Q4Fit-only correct | Q4-only correct | 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| English MGSM | +12.5 pp | 5 | 1 | −3.1 to +28.1 pp | 0.21875 |
| Chinese MGSM | −12.5 pp | 2 | 6 | −28.1 to +3.1 pp | 0.28906 |
| MBPP | 0.0 pp | 0 | 0 | 0.0 to 0.0 pp | 1.00000 |

English and Chinese items are paired translations. Resampling at the **32 original-question clusters**, rather than treating all 64 language responses as independent, gives an overall bilingual math accuracy difference **0.0 pp**, cluster bootstrap 95% CI **−9.4 to +10.9 pp**. These data do not establish task-accuracy superiority. The MBPP interval of zero is an exact property of these 24 deterministic outcomes, not evidence of equivalence in the general task distribution.

## Frozen protocol

- [Exact public sample indices, task IDs, prompt hashes and dataset-source JSONL hashes](QWEN3_4B_TASK_MANIFEST.json), frozen suite SHA256 `6017f4385c725f9653f992de3038846579f7c1eb8152f57ddcbf614b9946784c`.
- English and Chinese parallel **MGSM test** splits from [juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm), 32 shared indices sampled with Python `random.Random(20261123)`. Gold numerical answers matched across all 250 parallel source rows.
- **MBPP sanitized test** from [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp), 24 fixed IDs. The task prompt includes the required Python function **signature**, extracted mechanically from the official reference definition, but not test assertions. All 24 reference solutions passed the exact tests inside the sandbox. Test runs used rootless Podman, Python 3.9 Alpine, no network, read-only root, unprivileged user, 256MB/1-CPU/9s limits.
- One RTX 4090; same CUDA-enabled `llama-server` executable and Jinja chat template for all four weights. Greedy `temperature=0`, `top_k=1`, seed 20261123, `--reasoning off` / `enable_thinking=false`. MGSM maximum 192 new tokens and strict `FINAL: <number>` line; MBPP max 320 tokens, zero-shot code, official assertions.
- 4 model runs × 88 task responses = **352 deterministic requests**; results derived from per-model auditable receipts containing source GGUF SHA256 and suite SHA256.

**Protocol correction disclosed:** The first exploratory execution omitted the exact MBPP callable signature and produced many `NameError` failures. That version is intentionally **not used for the published code results**; it remains locally archived under `work/task-v05/` with its original manifest saved in `work/task-v05-invalid/`. The corrected manifest retains the same 32 math question indices and 24 MBPP IDs, unchanged MGSM prompts and hidden assertions, but inserts a fixed callable signature for each code task. The entire four-model 88-task suite was rerun under the corrected manifest. See [methodology](../docs/TASK_EVALUATION.md).

## Interpretation and relation to v0.4

The [v0.4 held-out BF16-KLD evaluation](QWEN3_4B_TENSOR_SEARCH.md) reported **16.0% less WikiText KLD and 13.7% less code-text KLD** for Q4Fit than pure Q4_K at nearly the same size. This v0.5 task experiment did **not** reproduce those improvements as a general accuracy gain. The most salient observable effect was language-specific answer flipping: the 32 matched English/Chinese math questions moved in opposite directions while the aggregate score stayed constant.

One 4B model, small samples, greedy non-thinking decoding and potentially contaminated public benchmarks limit interpretation. Some BF16 accuracy counts are lower than quantized counts; this can occur with discrete outcomes near decision boundaries and is **not** evidence that lower precision is intrinsically more capable. Model-level KL and task correctness measure different phenomena. Do not conflate sampled answer accuracy with broad reasoning, coding or multilingual superiority.

## Results and reproducibility

- [88-row per-task boolean outcome CSV](QWEN3_4B_TASK_ITEM_RESULTS.csv) (no questions, gold answers, hidden tests or model responses).
- [Model-level scores, per-domain McNemar tests and paired/bootstrap intervals JSON](QWEN3_4B_TASK_RESULTS.json).
- [Source code, sandbox configuration and exact CLI](../docs/TASK_EVALUATION.md).
- The final quantized tensor recipe and size/KLD measurements remain unchanged at [v0.4](../recipes/qwen3-4b-v04-7tensor.txt). No new optimizer weights were selected from task results.
