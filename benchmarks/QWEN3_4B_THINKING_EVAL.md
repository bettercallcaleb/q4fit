# Qwen3-4B v0.6 — independent bounded-thinking replication

**2026-11-23 | Complete | 480/480 inference requests | Negative for a statistically established Q4Fit task advantage.**

The exact seven-tensor [v0.4 Q4Fit recipe](../recipes/qwen3-4b-v04-7tensor.txt) was **not changed**. This is a new task benchmark of the same GGUF bytes, not a new quantizer optimization. The evaluation protocol and **disjoint frozen task manifest were committed separately before any Q4Fit or Q4_K_S evaluation**, in [preregistration commit 545c28f](https://github.com/bettercallcaleb/q4fit/commit/545c28f).

## Actual frozen-task outcomes

| Model | GGUF bytes | MGSM English (48) | MGSM Chinese (48) | MBPP sanitized (24) | Truncated (120) |
|---|---:|---:|---:|---:|---:|
| BF16 | 8,051,285,344 | 37/48 (77.1%) | 33/48 (68.8%) | 17/24 (70.8%) | 2 |
| Pure Q4_K | 2,269,269,344 | 37/48 (77.1%) | **35/48 (72.9%)** | 15/24 (62.5%) | 10 |
| **Q4Fit 7-tensor** | **2,275,628,384** | **39/48 (81.3%)** | 34/48 (70.8%) | **17/24 (70.8%)** | 6 |
| Standard Q4_K_S | 2,383,309,664 | 38/48 (79.2%) | 34/48 (70.8%) | 14/24 (58.3%) | 5 |

All four models completed 120/120 tasks with nonempty `reasoning_content` returned by llama.cpp. Mean generation length, inclusive of bounded reasoning: BF16 302.5 tokens; pure Q4_K 392.4; Q4Fit 342.8; Q4_K_S 342.3. The frozen per-request cap was **1024 tokens**, including reasoning, and any failure to produce a parseable final answer counted as incorrect.

## Preregistered primary comparison: Q4Fit minus pure Q4_K

| Task domain | Correctness delta | Q4Fit-only correct | Q4-only correct | 95% paired bootstrap interval | McNemar exact two-sided p |
|---|---:|---:|---:|---:|---:|
| English MGSM | +2/48 (+4.17 pp) | 5 | 3 | −6.25 to +16.67 pp | 0.7266 |
| Chinese MGSM | −1/48 (−2.08 pp) | 3 | 4 | −12.50 to +8.33 pp | 1.0000 |
| MBPP Python | +2/24 (+8.33 pp) | 2 | 0 | 0.00 to +20.83 pp | 0.5000 |

Math totals across the same 48 source questions in two languages: pure Q4_K **72/96**, Q4Fit **73/96**. A paired cluster bootstrap by original question ID (English and Chinese outcomes sampled together) yields **+1.04 percentage points (95% CI −6.25 to +9.38 pp)**. All three primary domain comparisons have exact p-values well above 0.05, before any multiplicity correction.

The extra two Q4Fit code passes over pure Q4_K were actual official-test improvements on the frozen sample, but with n=24, two favorable discordances and no unfavorable ones still give **exact two-sided p=0.50**, not statistically established superiority. Q4Fit and BF16 passed exactly the same **17/24** code tasks. Q4Fit and Q4 produced *different* code text on 17/24 tasks, underscoring that textual differences need not imply success/failure differences.

## Comparison with larger standard Q4_K_S

Q4Fit uses **107,681,280 fewer bytes** than Q4_K_S, approximately 4.52% less storage relative to Q4_K_S. Q4Fit yielded +1 English correct, the same Chinese count and +3 code correct, but the MBPP two-sided McNemar exact p-value was 0.25; none of these sampled improvements establishes generalization. Standard Q4_K_S is also outside the 2.12GiB hard budget previously imposed on Q4Fit optimization.

## Cross-language alignment (48 identical translated questions)

| Model | Both translations right | Only English right | Only Chinese right | Neither right |
|---|---:|---:|---:|---:|
| BF16 | 29 | 8 | 4 | 7 |
| Pure Q4_K | 31 | 6 | 4 | 7 |
| Q4Fit | 31 | 8 | 3 | 6 |
| Q4_K_S | 28 | 10 | 6 | 4 |

The opposite-signed English/Chinese shifts in v0.5 (English Q4Fit +4, Chinese −4 vs pure Q4_K on a 32-pair non-thinking sample) were not reproduced at comparable magnitude in v0.6 (English +2, Chinese −1). Different sampled questions and reasoning protocols **prevent pooling the two rounds as a direct treatment-effect estimate**.

## Methodological receipts and reproducibility

- [Frozen v0.6 manifest](QWEN3_4B_THINKING_MANIFEST.json), sample size **48 parallel English/Chinese MGSM test pairs + 24 MBPP sanitized test tasks**, distinct from all v0.5 sampled IDs. Source row hashes, per-task prompt hashes, source-index exclusions and model-independent exact decoding rules were frozen before quantized candidates were evaluated. Frozen private suite SHA256: `dbb052d2c324ec0942874be2c5ef05b0c1ad315bd4b3e47ef2fc6de3ca4732ca`.
- `llama-server` CLI executable SHA256 `314574e46827fd3a65109e3d5d2f13a7641882e12fa54e55fa8e7a3e43762481`; RTX 4090, Jinja chat formatting, `--reasoning on --reasoning-budget 256 --reasoning-format deepseek`, greedy temperature 0 and top_k 1, seed 2026112306, maximum 1024 generated tokens.
- Math scoring: exact numerical answer from a `FINAL: <number>` content line. MBPP: official test assertions in offline rootless Podman, executable code AST check, no network, read-only filesystem, unprivileged user, CPU/memory/PID/time limits; the exact expected function **signature** was included in the question, hidden test assertions never supplied to the model. All 24 reference solutions passed inside the same sandbox.
- [Full per-model aggregate and paired inference JSON](QWEN3_4B_THINKING_RESULTS.json). [120-task model correctness/response-equality/truncation CSV](QWEN3_4B_THINKING_ITEM_RESULTS.csv). Rebuild and run with [v0.6 methodology](../docs/THINKING_EVALUATION.md) and `scripts/prepare_thinking_suite.py` / `q4tasks.py`. Raw tasks, model-generated content and tensor weights stay local, outside Git.

## Exact evaluation CLI

The repository root contains the following replay workflow. Set the four model paths to the recorded GGUF SHA256 weights listed in `QWEN3_4B_THINKING_RESULTS.json` before running:

~~~bash
python3 -m pip install -e . datasets
python3 scripts/fetch_task_datasets.py --output-dir work/task-v05/datasets --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json
python3 scripts/prepare_thinking_suite.py --data-dir work/task-v05/datasets --workdir work/task-v06 --manifest benchmarks/QWEN3_4B_THINKING_MANIFEST.json
python3 q4tasks.py validate --tasks work/task-v06/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_THINKING_MANIFEST.json --workdir work/task-v06
python3 q4tasks.py run --tasks work/task-v06/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_THINKING_MANIFEST.json --workdir work/task-v06 --name q4fit --gguf "$Q4FIT" --server "$SERVER"
python3 q4tasks.py score --tasks work/task-v06/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_THINKING_MANIFEST.json --workdir work/task-v06 --result bf16=work/task-v06/bf16.jsonl --result q4=work/task-v06/q4.jsonl --result q4fit=work/task-v06/q4fit.jsonl --result q4ks=work/task-v06/q4ks.jsonl --output work/task-v06/task-summary.json
python3 scripts/export_thinking_results.py --manifest benchmarks/QWEN3_4B_THINKING_MANIFEST.json --workdir work/task-v06 --outdir benchmarks
~~~

Run `q4tasks.py run` analogously for `bf16`, `q4`, and `q4ks` before the score/export commands. The runner refuses a mismatched model/suite receipt and resumes a partial run safely.

## Scientific interpretation

The prior [v0.4 experiment](QWEN3_4B_TENSOR_SEARCH.md) showed mean BF16-reference KLD reductions of **16.0% on WikiText holdout and 13.7% on code holdout**, at 2.119GiB, compared with pure Q4_K. v0.5 (nonthinking, 32 translation pairs + 24 code) showed **no aggregate task advantage**. v0.6 (bounded thinking, disjoint 48 translation pairs + 24 code) has small favorable Q4Fit point differences but no statistically supported task advantage. KL reduction therefore **does not guarantee discrete downstream correctness gains**.

This is a single-model, deterministic, small, public-benchmark pilot with potentially trained-on questions. Paired bootstrap intervals and exact McNemar p-values are conditional on these samples and do not establish a population-wide absence of effects. The model is run with a **bounded 256-token reasoning allowance**, not unlimited thinking; inference-time configuration and truncation can affect correctness. No claim is made about reasoning beyond these tasks, other languages, security, instruction-following, different models or other llama.cpp versions. More optimizer searches on these same tested domains would be test-set contamination, not new evidence.
