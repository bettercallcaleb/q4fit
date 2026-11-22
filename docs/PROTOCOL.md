# Benchmark protocol

Q4Fit is an experimental GGUF quantization search harness, not a quantizer algorithm.

## Evaluation hierarchy

1. Save reference logits from one immutable BF16 GGUF.
2. Build a plain Q4_K baseline directly from BF16 with no imatrix.
3. Tune on independent development domains: e.g., WikiText-2 **validation** and source-code text.
4. Search contiguous layer groups (for example, FFN Down layers 0–17) using measured BF16 KL divergence per domain.
5. Select a byte-budgeted candidate by minimizing the worst per-domain **relative KLD**. The group effects are only an approximation.
6. Build the **combined recipe** again from the BF16 source. Re-evaluate every tuning domain.
7. Require no KLD degradation in either tuning domain, within byte budget, and no top-token agreement drop beyond the configured tolerance. If those tests fail, report failure.
8. Independently assess on untouched WikiText-2 **test**, with no further tuning based on that score.

## Report each corpus separately

Do not average tokens across incompatible domains without explicitly specifying weights. Include mean KLD, 99th-percentile KLD, top-1 token agreement, actual GGUF bytes, and optionally downstream accuracy and generation speed.

A lower average KLD is not enough to conclude a model is better: top-token accuracy, safety, multilingual behavior, instruction following, and tail errors may deteriorate.

## Qwen3-4B v0.1 falsification

The initial Q4Fit mixed recipe was selected from 256 tokens of hand-authored smoke text. It showed an apparent 29.8% reduction in KLD against plain Q4_K on the selection data, and about 5% on a two-chunk authored holdout.

On a larger independent WikiText-2 raw **test** evaluation (24 chunks × 512), the result reversed:

| Candidate | GGUF bytes | Mean KL vs BF16 | Top-1 agreement |
|---|---:|---:|---:|
| Plain Q4_K | 2,269,269,344 | 0.110148 | 87.271% |
| v0.1 mixed | 2,262,265,184 | **0.125771 (worse)** | 86.928% |
| Standard Q4_K_S | 2,383,309,664 | 0.095972 | 88.399% |
| Standard Q4_K_M | 2,497,280,864 | 0.072403 | 90.294% |

On 8 chunks × 512 of separately collected source code: plain Q4_K KL 0.083428; v0.1 mixed 0.085674; standard Q4_K_M 0.057964. **Q4_K_M is larger by approximately 10%, so this is not an equal-size comparison.**

The original v0.1 claim does not generalize. Preserve and cite this falsification in future README and results.

Source test data: WikiText-2 raw archive; distribute only a fetch/preparation script or source instructions, not copied dataset contents. All evaluation artifacts, model weights, and caches should stay out of the Git repository.

## Open limitations

The group-level search assumes approximately additive KLD changes. Validation of the combined GGUF is essential, and there is no claim of a global optimum. More rigorous work would use layer-by-layer or tensor-by-tensor budget search, multiple architectures, broader independent evaluation suites, and downstream tasks. Do not treat a one-model PoC as a general quantization finding.
