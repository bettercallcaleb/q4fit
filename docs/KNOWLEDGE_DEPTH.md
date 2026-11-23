# ConceptLadder: technical scenario benchmark for GGUF models

Q4Fit bundles four benchmark suites. **Exact v0.4 is the default and recommended one.** Classic v0.1 (36 questions), Hard v0.2 (240 questions), and Ladder v0.3 (240 questions) remain available only to reproduce older multiple-choice runs.

Exact uses 240 open-answer scenarios: 15 technical concepts, four nominal levels and four generated parameter variants. Subjects include TCP/IP, DNS, SQLite WAL and window functions, Linux file descriptors/permissions/bitfields, JWT claims, Kubernetes RBAC and scheduling.

## Run

Install Q4Fit and build a compatible llama.cpp llama-server. Then run:

    q4fit-depth --reference original-f16.gguf --candidate model-Q4_K.gguf       --llama-server ../llama.cpp/build/bin/llama-server --out depth-results

Optional arguments:
- --suite exact | ladder | hard | classic (default exact)
- --seed 17: scenario generation seed; use 41 for another generated form
- --variants 4: parameter versions per concept (1–32)
- --packs net-hard | db-hard | sys-hard | security-hard for one cluster
- --inference-seed 17: fixed llama.cpp RNG, independent of scenario generation
- --max-tokens 64 and --allow-thinking: control answer budget and server thinking mode
- --ngl 0 to avoid GPU offload

Answer-option ordering (--answer-seed) has no effect on Exact because there are no options; it is retained for historical multiple-choice suites.

## What it writes

- receipt.json: model and server SHA256, source dataset SHA256, suite, inference parameters
- reference.json and candidate.json: exact raw text and parsed predictions, including invalid responses
- comparison.json: matched correctness, answer losses/gains, per-concept diagnostics and invalid counts
- report.md: readable summary
- reference-server.log, candidate-server.log: llama.cpp logs

A valid Exact answer is a decimal, octal, hex, error code, interface name, two-character string or ordered binary mask. Scoring uses strict typed normalization; no LLM judge or numeric-option guessing. A model's explanation without an unambiguous final value can be scored invalid. This is a response-format confound, so review invalid counts.

## Independent adversarial audit

Ladder v0.3 had a serious flaw: a rule choosing the second-smallest numeric option, without reading the question, scored about 64–65%. Do not rank models using its historical MCQ scores.

Exact v0.4 removes answer alternatives, varies more underlying outcomes and uses binary decision masks rather than narrow counts where possible. A no-question, family-and-level most-frequent-answer attack, trained on scenario seeds 17/41, scored 87/720 (12.1%) on held-out seeds 91/123/777.

In two local RTX4090 Q4 pilots (seeds 17/41; no thinking, temperature 0, 64 output tokens): 4B-class 25/30, 14B 72/81, 32B 91/98 out of 240 per form. The 4B scored roughly at a prompt-blind floor. Even the 32B answered only 12/120 L4 items over the two forms. The four difficulty levels and contiguous-depth metric are NOT independently validated.

These are different model sizes, not matched original-vs-Q4 pairs. For quantizer evaluation, use the **same underlying model** in high precision and quantized variants, identical prompt/inference parameters, multiple fresh seeds, and paired per-concept analysis. Report accuracy, invalid-format rate and raw answer changes; do not interpret small differences near a low reference score.

See [upstream Exact audit](https://github.com/bettercallcaleb/conceptladder-bench/blob/main/docs/EXACT_V04_AUDIT.md).
