# Export Q4Fit evidence to RAMGPT

After a completed `q4fit-depth` comparison, export a small, portable evidence record:

```bash
q4fit-evidence --depth-dir depth-results --out depth-results/ramgpt-experiment.json
```

The exporter reads the original `receipt.json`, `reference.json`, `candidate.json`, and `comparison.json`. It checks the dataset identity, model labels, seeds, reported deltas, and SHA-256 formatting before exporting.

The output uses `ramgpt-experiment-v1`. It includes artifact hashes and sizes, benchmark version and seeds, paired accuracy and depth changes, answer-flip counts, and hashes of the four input files. It deliberately excludes machine-local paths, model weights, raw responses and server logs.

This is a **Q4Fit/ConceptLadder diagnostic record**, not a QuantBench VERIFIED result. It does not claim a quantization win, re-run inference, inspect the model payload, or substitute for the QuantBench frozen BF16 protocol.

## Where it fits

1. Q4Fit makes the candidate GGUF and preserves its production receipt.
2. `q4fit-depth` produces paired ConceptLadder evidence.
3. `q4fit-evidence` exports the path-free evidence JSON.
4. QuantBench and llama-autotune retain independent measurement procedures and output formats. Do not merge their scores or silently change their runtime settings.
5. An editor may reference the JSON, original source evidence, and limitations while preparing a RAMGPT article through the existing publication workflow.

Publishing this JSON does not grant any RAMGPT verification designation. An editorial review and the relevant tool's own verification procedure remain separate.
