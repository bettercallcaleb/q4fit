# Q4Fit v0.6 — Independent bounded-thinking replication

This protocol was frozen **before any candidate-specific outcome was inspected**, with Q4Fit's v0.4 seven-tensor recipe unchanged. This is a second, disjoint end-task sample intended to check whether an observed BF16-KLD advantage persists in English/Chinese math and Python task outcomes.

## Frozen task design

- MGSM test splits (`juletxara/mgsm`): **48 new, identical row indices** in English and simplified Chinese, 96 responses/model. The indices explicitly exclude the 32 MGSM rows in the earlier v0.5 task manifest.
- MBPP sanitized test split (`google-research-datasets/mbpp`): **24 new tasks**, task IDs explicitly exclude the earlier v0.5 24. All reference solutions passed the official test cases inside the same rootless container sandbox.
- Seed `2026112306`, deterministic Python `random.Random` sample, identical source JSONL hashes as v0.5, no selection based on any Q4Fit/quantization result. See [immutable manifest](../benchmarks/QWEN3_4B_THINKING_MANIFEST.json), including IDs, source hashes, prompt hashes, task-suite hash, exclusions and decoding parameters.

## Frozen model protocol

- Four source-identical Qwen3-4B weights: original BF16, pure Q4_K, Q4Fit v0.4 fixed 7-tensor mix, and standard Q4_K_S. Exact SHA256 per model is recorded with each run.
- Same CUDA-enabled llama.cpp `llama-server` executable on RTX 4090, `--jinja --reasoning on --reasoning-budget 256 --reasoning-format deepseek -ngl 999 -np 1 -c 4096`. Reasoning is **bounded at 256 tokens**, not full unbounded reasoning; only final `message.content` is graded. Per-request presence/character length of `reasoning_content` is recorded but private chain-of-thought content is not published.
- Deterministic `temperature=0`, `top_k=1`, seed `2026112306`, no few-shot examples. MGSM and Python completion maximum **1024 tokens each**, inclusive of the reasoning portion. A length-truncated response, missing final answer or invalid output is graded incorrect; record the rate by model and task.
- MGSM prompt/score unchanged from v0.5: last full-line `FINAL: <integer>` parsed as a number, exact numeric match to dataset ground truth.
- MBPP prompt/score unchanged from corrected v0.5: includes exact Python callable interface signature, not hidden assertions; generated code is evaluated against official tests inside a network-disabled rootless Podman sandbox with read-only root, non-root user, one CPU, 256MiB, 32 PIDs, 9-second deadline.

## Analysis fixed in advance

Compare Q4Fit vs pure Q4_K **first**, separately by English and Chinese math, and then by code. Report paired success/failure discordances, exact McNemar two-sided p values, and 95% paired bootstrap intervals. Treat English and Chinese as translated pairs of the same 48 source problems; report a cluster bootstrap across source indices for combined bilingual math. Present BF16 and Q4_K_S as additional reference comparators. Do not interpret any single-domain positive difference as cross-model generalized superiority unless larger independent studies confirm it. Do not reselect the Q4Fit recipe after inspecting results.

Negative outcomes, decoding failures, generation truncation and differences in answer distribution are part of the report. The earlier v0.5 non-thinking benchmark is **not directly comparable** due to disjoint tasks and different reasoning settings.
