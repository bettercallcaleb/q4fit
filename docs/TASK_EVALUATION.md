# Frozen downstream task evaluation (v0.5)

Quantization KL is a distributional proxy. This experiment tests **exact question outcomes** using Qwen3-4B BF16, pure Q4_K, Q4Fit v0.4 mixed Q3/Q4/Q5, and standard Q4_K_S. This is a small single-model exploratory benchmark, not a broad capability guarantee.

## Public sources and immutable sampling

- `juletxara/mgsm`: official MGSM parallel English and Simplified Chinese **test** splits, each 250 rows. 32 fixed shared row indices, so the two language variants have the same numerical answers.
- `google-research-datasets/mbpp`: **sanitized test** split, 257 rows. 24 fixed task IDs, 3–5 hidden official assertions per code task.
- Fixed Python seed 20261123, documented indices, JSONL source hashes and per-task prompt hashes in [the frozen manifest](../benchmarks/QWEN3_4B_TASK_MANIFEST.json). Private copies of exact questions, gold answers, tests and generated outputs are under `work/`; they are not published.
- Pilot correction recorded transparently: an initial trial omitted required MBPP function names, leading mostly to `NameError`. Those original 4-model results are retained locally as **invalid for code ability assessment**. The corrected v0.5.1 task manifest has unchanged sampled IDs, identical MGSM prompts and hidden MBPP assertions, and mechanically adds the required function signature to each MBPP prompt. The corrected 88-task suite was then frozen **before the corrected model runs**, with no task-specific adjustments based on previous performance.
- All 24 gold MBPP solutions passed the isolated official tests before inference, in both protocols.

## Inference protocol

- One GPU: RTX 4090. Same `llama-server` executable, model chat template, prompts and seed for every model.
- llama-server chat parameters: `--jinja --reasoning off --chat-template-kwargs '{"enable_thinking":false}' -ngl 999 -np 1 -c 4096`.
- Every API call: `temperature=0`, `top_k=1`, `seed=20261123`, system message `Follow the requested output format exactly.`
- MGSM: no few-shot examples, max 192 completion tokens; extract strictly from a line `FINAL: <number>`, compare numerically with dataset answer_number. Failure to emit a parseable final line counts as incorrect.
- MBPP: no few-shot examples, max 320 tokens. Model sees the task description plus the required Python callable **signature** derived mechanically from the reference definition; the signature is interface information, not code logic. It **never sees hidden test assertions**. Generated Python (fenced or plain) is parsed by AST first, then executed against all official asserts. Pass only when all assertions succeed.
- Tests run via rootless Podman with Python 3.9 Alpine, no network, read-only root FS, no Linux capabilities, unprivileged user, pids <= 32, 256 MiB RAM, 1 CPU, 9-second timeout, task script mounted read-only. Do not run generated code on the host.
- One model loaded at a time. Model SHA256, frozen suite SHA256 and exact responses recorded; interrupted runs resume without mixing model versions.

## Reproduction

The following fetch helper pins the local JSONL data to the source hashes in the public manifest, refusing any changed dataset. The `datasets` dependency is only needed for source retrieval, not for Q4Fit core quantization:

~~~bash
python3 -m pip install datasets
python3 scripts/fetch_task_datasets.py --output-dir work/task-v05/datasets --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json
python3 scripts/prepare_task_suite.py --data-dir work/task-v05/datasets --workdir work/task-v05-fixed --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json
q4fit-tasks validate --tasks work/task-v05-fixed/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json --workdir work/task-v05-fixed
q4fit-tasks run --tasks work/task-v05-fixed/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json --workdir work/task-v05-fixed --name q4fit --gguf /models/qwen3-4b-v04.gguf --server /opt/llama.cpp/build/bin/llama-server
q4fit-tasks score --tasks work/task-v05-fixed/frozen_tasks.jsonl --manifest benchmarks/QWEN3_4B_TASK_MANIFEST.json --workdir work/task-v05-fixed --result bf16=work/task-v05-fixed/bf16.jsonl --result q4=work/task-v05-fixed/q4.jsonl --result q4fit=work/task-v05-fixed/q4fit.jsonl --result q4ks=work/task-v05-fixed/q4ks.jsonl --output work/task-v05-fixed/task-summary.json
~~~

## Interpretation

Measure each domain separately (English MGSM, Chinese MGSM, Python MBPP), with exact per-item discordances and paired bootstrap 95% intervals. No instruction tuning or calibration data is used to adjust the fixed 7-tensor recipe after observing benchmark results. Small n, possible benchmark contamination, constrained response formats and disabled thinking limit generalizability. BF16 can itself be wrong; KL improvement does not ensure better task accuracy.
