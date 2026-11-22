# Per-tensor byte-budget search (v0.4)

`q4fit-tune` measures single-tensor quantization changes against a frozen BF16 logits reference in multiple tuning domains, then proposes actual multi-tensor GGUF models under a hard byte budget.

## Setup and execution

Build Q3_K/Q4_K/Q5_K banks with `q4fit-cache prepare` first. For every domain, generate BF16 logits using `q4fit baseline` and evaluate the unmodified, cache-indexed plain Q4_K model using `q4fit eval`. Copy [the JSON template](../examples/tensor-search.json), fix all paths, and list exact tensor names and candidate quantization types.

```bash
q4fit-tune scan --config examples/tensor-search.json --workdir work/tensor-run --limit 60
q4fit-tune optimize --config examples/tensor-search.json --workdir work/tensor-run --beam-width 2500 --max-edits 8 --verify-top 4
```

The scanner measures each requested single tensor against **each** tuning domain and records raw evaluation logs, KL divergence, 99th-percentile KL, BF16 top-token agreement, byte delta, and source/measurement hashes. The scanner deletes disposable 2–3GB GGUF candidates; verified combinations are kept only if they pass all tuning gates. `scan` resumes by skipping already recorded `(tensor name, type)` pairs and rejects changed corpus, reference, model, evaluator path or configuration.

## Search objective

For a measured type change `j` and domain `k`, define the relative empirical KL delta

`d[j,k] = (KLD_single[j,k] - KLD_Q4[k]) / KLD_Q4[k]`.

The beam search approximates `min max_k (1 + sum_{j in recipe} d[j,k])` subject to `GGUF bytes <= hard budget`, `<= max-edits` changes, and at most one candidate quant type per tensor. Byte deltas come from actual cached bank tensor sizes. It retains states across multiple byte-size buckets, rather than greedily picking only the lowest KL delta.

**The additive KL assumption is only a ranking heuristic.** Quantization errors can interact, and effects on top-token agreement can differ from mean KL. The program materializes and re-evaluates combined GGUF candidates, checking actual bytes, actual BF16 mean KL on **every tuning domain**, and top-token agreement drop within a fixed tolerance. Besides the beam-ranked combinations it retains two simple, measured Q5-only single/pair anchors; after validating the shortlist, it selects the best **actually observed** multi-domain result, not necessarily the best additive prediction. It labels runs without a validated improvement as such; there is no guaranteed winner.

## Leakage and interpretation

Freeze independent holdout text and code **before** optimization. Do not feed their outcomes into tensor selection, byte budget, evaluation-domain weights, or search hyperparameter tuning. Run any holdout only on the final, preselected recipe and publish negative results.

This is a beam approximation, not an exact globally optimal mixed-integer allocation. Individual probes plus an additive approximation can overfit a small evaluation set, and false improvements become more likely with many tested candidates. Evaluation is a BF16 output-distribution proxy, not downstream task accuracy.

## Reproducibility and data safety

Only code, JSON configuration templates, and tabular results are committed. Quantized model weights, cached banks, raw logits and corpora stay under `work/` and are excluded from Git. The byte-level GGUF composer is documented in [CACHE.md](CACHE.md).
