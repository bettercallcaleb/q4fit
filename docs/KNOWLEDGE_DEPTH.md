# Knowledge-depth test for GGUF models

Q4Fit bundles [ConceptLadder Bench](https://github.com/bettercallcaleb/conceptladder-bench), with two complementary suites:

- **Hard (default):** 240 deterministic scenarios from 15 technical concepts, each tested at four levels and four parameter variants. Networking, SQLite, Linux, and Kubernetes/security. Expected answers come from executable/algorithmic rules, not from an AI judge. All four answer options have the same number of characters.
- **Classic:** the previous 36 basic questions covering TCP, TLS 1.3, and SQLite.

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

This defaults to `--suite hard --variants 4 --seed 17`, generating 240 questions. To use the old 36 introductory questions, add `--suite classic`.

If your build has no GPU, add `--ngl 0`. For Hard, select `--packs net-hard`, `--packs db-hard`, `--packs sys-hard`, or `--packs security-hard`. For Classic, use `--packs tcp`, `--packs tls13`, or `--packs sqlite`.

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

A one-run local pilot with a roughly 4B-class GGUF pair under constrained decoding scored BF16 **96/240 (40.0%)** and Q4_K_M **87/240 (36.3%)**. There were 29 BF16-only correct answers and 20 Q4-only correct answers. These are exploratory, not a statistically established quantization effect. The older Classic suite was nearly saturated by the same model family. Report both, and calibrate Hard against stronger models before using it to rank quantizers.

## Practical notes

- Both models must be **GGUF**, preferably from the same original model and revision.
- The script hashes both GGUF files to capture the exact artifacts.
- By default thinking is disabled for compatible chat templates; add `--allow-thinking` for models that require reasoning output, and make sure the answer parser can extract a single letter. Plain letters and clearly labeled answers such as `B. the selected option` are accepted. Other unparseable answers count as incorrect; raw outputs are preserved for audit.
- The default port is `18221`; choose another with `--port` if in use.
- Existing reports are preserved unless you pass `--overwrite`.
- The bundled question set has a fixed SHA256. The independent [ConceptLadder repository](https://github.com/bettercallcaleb/conceptladder-bench) remains the upstream benchmark.
- If a model needs more response budget, pass `--max-tokens 128` and keep it identical across the pair.
