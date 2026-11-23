# Knowledge-depth test for GGUF models

Q4Fit includes a fixed snapshot of [ConceptLadder Bench](https://github.com/bettercallcaleb/conceptladder-bench): 36 multiple-choice questions across TCP, TLS 1.3, and SQLite.

The same question set is asked of an original GGUF model and a quantized GGUF model. Each of the nine concepts has four questions, from basic recognition to harder boundary cases.

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

If your build has no GPU, add `--ngl 0`. You may restrict to one pack with `--packs tcp`, `--packs tls13`, or `--packs sqlite`.

For a Q4Fit mixed-precision model, replace `--candidate` with the mixed GGUF file. To compare standard Q4_K against a mixed recipe fairly, run the same original-model reference for each pair and use matching seeds and server settings.

The command starts one llama-server process at a time, tests the original first and the quant second, and closes each process when finished. It does not upload models or benchmark results.

## Files produced

| File | Purpose |
| --- | --- |
| `receipt.json` | Model file hashes, benchmark fingerprint, inference settings |
| `reference.json` | Original model's 36 answers |
| `candidate.json` | Quantized model's 36 answers |
| `comparison.json` | Per-question losses and gains, accuracy, depth |
| `report.md` | A readable comparison report |
| `*-server.log` | llama-server logs for troubleshooting |

## How to read the score

- **Accuracy:** total correct answers / total questions.
- **Contiguous depth (0–4):** the number of levels passed in order before the first failure, averaged over concepts.
- **Knowledge-loss rate:** questions the reference answered correctly but the quant got wrong, divided by all questions the reference answered correctly.
- **Gained answers:** questions the quant got right but the reference got wrong, reported separately.

If a model passes level 1 and 2 but fails level 3, its contiguous depth on that concept is 2, even when it happens to guess level 4 correctly.

This is a *small diagnostic* with a 25% chance baseline, not a validated general-intelligence score. A single answer flip can be random. Repeat with several seeds and additional unseen questions before claiming one quantization is better.

## Practical notes

- Both models must be **GGUF**, preferably from the same original model and revision.
- The script hashes both GGUF files to capture the exact artifacts.
- By default thinking is disabled for compatible chat templates; add `--allow-thinking` for models that require reasoning output, and make sure the answer parser can extract a single letter. Unparseable answers count as incorrect.
- The default port is `18221`; choose another with `--port` if in use.
- Existing reports are preserved unless you pass `--overwrite`.
- The bundled question set has a fixed SHA256. The independent [ConceptLadder repository](https://github.com/bettercallcaleb/conceptladder-bench) remains the upstream benchmark.
- If a model needs more response budget, pass `--max-tokens 128` and keep it identical across the pair.
