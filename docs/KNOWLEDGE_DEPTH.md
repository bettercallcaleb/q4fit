# Knowledge-depth test for GGUF models

Q4Fit bundles [ConceptLadder Bench](https://github.com/bettercallcaleb/conceptladder-bench), with three complementary suites:

- **Ladder v0.3 (default):** 240 technical scenario questions, with easier L1 entry cases, advanced L2–L4 cases, and independent question-versus-answer-order seeds. Networking, SQLite, Linux and Kubernetes/security.
- **Hard v0.2:** the original 240 scenarios, unchanged and still available for reproducing old runs.
- **Classic v0.1:** the original 36 basic questions covering TCP, TLS 1.3, and SQLite.

The 240-question sets contain 15 underlying concepts, four levels per concept, and four parameter variants. Answers come from executable algorithms or standard-library SQLite/POSIX operations; all options have equal-length strings.

The same questions, seed, and inference settings are used for original and quantized GGUF weights.

## One command

First build `llama-server` in llama.cpp:

```bash
cmake --build ../llama.cpp/build --target llama-server -j 4
```

Then run this from the Q4Fit folder after `pip install -e .`:

```bash
q4fit-depth \
  --reference my-first-q4/original-f16.gguf \
  --candidate my-first-q4/Qwen3-0.6B-Q4_K.gguf \
  --llama-server ../llama.cpp/build/bin/llama-server \
  --out my-first-q4/depth
```

This defaults to `--suite ladder --variants 4 --seed 17 --answer-seed 17`, generating 240 questions. To use the previous uncalibrated 240-question version add `--suite hard`; for the old 36 questions use `--suite classic`.

Use `--seed 41 --answer-seed 17` to generate new questions while preserving answer-position randomization, or `--seed 17 --answer-seed 63` to reorder the answers to exactly the same questions. The model backend RNG uses independent `--inference-seed 17` unless changed explicitly. Keep that fixed for controlled comparisons.

If your build has no GPU, add `--ngl 0`. For Ladder or Hard, select `--packs net-hard`, `--packs db-hard`, `--packs sys-hard`, or `--packs security-hard`. For Classic, use `--packs tcp`, `--packs tls13`, or `--packs sqlite`.

For a Q4Fit mixed-precision model, replace `--candidate` with the mixed GGUF file. To compare standard Q4_K against a mixed recipe fairly, run the same original-model reference for each pair and use matching seeds and server settings.

The command starts one llama-server process at a time, tests the original first and the quant second, and closes each process when finished. It does not upload models or benchmark results.

## Files produced

| File | Purpose |
| --- | --- |
| `receipt.json` | Model file hashes, benchmark fingerprint, inference settings |
| `reference.json` | Original model's answers |
| `candidate.json` | Quantized model's answers |
| `comparison.json` | Per-question losses and gains, accuracy, depth |
| `report.md` | A readable comparison report |
| `*-server.log` | llama-server logs for troubleshooting |

## How to read the score

- **Accuracy:** total correct answers / total questions.
- **Contiguous depth (0–4):** the number of levels passed in order before the first failure, averaged over concepts.
- **Knowledge-loss rate:** questions the reference answered correctly but the quant got wrong, divided by all questions the reference answered correctly.
- **Gained answers:** questions the quant got right but the reference got wrong, reported separately.

If a model passes level 1 and 2 but fails level 3, its contiguous depth on that concept is 2, even when it happens to guess level 4 correctly.

This is a *technical diagnostic*, not a validated general-intelligence score. The 25% multiple-choice chance baseline matters: when both models score near it, measuring their knowledge-retention difference is unreliable. Across 240 Hard items, repeated parameter variants are correlated because they share 15 underlying concepts; do not treat them as 240 independent concepts. A one-item flip is not reliable evidence.

In a local Ladder v0.3 pilot, Qwen3-14B Q4 scored 133/240, 125/240 and 118/240 on three question/option forms. A separate 4B-class Q4 model scored 88/240, 81/240 and 87/240. Mean accuracy: 52.2% vs 35.6%. L1→L4 accuracy descended in the 14B aggregate, but the 4B L2/L3 levels still inverted; some individual concepts also invert. Reordering answers alone changed scores by about 3 percentage points. These two different-size models **do not** establish a BF16-to-Q4 retention effect. See the upstream [calibration report](https://github.com/bettercallcaleb/conceptladder-bench/blob/main/docs/LADDER_V03_PILOT.md).

To compare multiple saved results, run `python3 -m conceptladder calibrate path/to/run1.json path/to/run2.json --out calibration.json`. Only compare question-identical runs with the same answer order when attributing item-level changes.

## Practical notes

- Both models must be **GGUF**, preferably from the same original model and revision.
- The script hashes both GGUF files to capture the exact artifacts.
- By default thinking is disabled for compatible chat templates; add `--allow-thinking` for models that require reasoning output, and make sure the answer parser can extract a single letter. Plain letters and clearly labeled answers such as `B. the selected option` are accepted. Other unparseable answers count as incorrect; raw outputs are preserved for audit.
- The default port is `18221`; choose another with `--port` if in use.
- Existing reports are preserved unless you pass `--overwrite`.
- The bundled question set has a fixed SHA256. The independent [ConceptLadder repository](https://github.com/bettercallcaleb/conceptladder-bench) remains the upstream benchmark.
- If a model needs more response budget, pass `--max-tokens 128` and keep it identical across the pair.
