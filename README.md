# Q4Fit

**Build a GGUF model, control its size, and publish it on Hugging Face.**

Most 4-bit conversions start from a standard preset. Q4Fit uses [llama.cpp](https://github.com/ggml-org/llama.cpp) to make the GGUF, but also lets you choose which parts of the model get more or fewer bits. The result is a model file you can run in tools such as [LM Studio](https://lmstudio.ai/) or share on Hugging Face.

You can use Q4Fit in two ways: make a standard Q4 model without worrying about the details, or experiment with different precision settings and measure what changes.

## Make and publish a Q4 model

This example uses [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), a small model with publicly available original weights. The commands below are for Ubuntu Linux. A GPU is not required to create the file.

**1. Install the tools**

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-venv
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-quantize -j 4
git clone https://github.com/bettercallcaleb/q4fit
cd q4fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

**2. Create the GGUF**

```bash
python scripts/publish_my_q4.py \
  --model Qwen/Qwen3-0.6B \
  --llama-cpp ../llama.cpp \
  --output-dir my-first-q4
```

This downloads the original model, converts it to GGUF, then makes `my-first-q4/Qwen3-0.6B-Q4_K.gguf`. It also creates a model description and a record of the source version and file checksum. Nothing is uploaded yet.

**3. Publish it**

Create a [Hugging Face account](https://huggingface.co/join) and a [write-access token](https://huggingface.co/settings/tokens), then sign in:

```bash
hf auth login
```

Replace `YOUR_USERNAME` with your Hugging Face username:

```bash
python scripts/publish_my_q4.py \
  --model Qwen/Qwen3-0.6B \
  --llama-cpp ../llama.cpp \
  --output-dir my-first-q4 \
  --publish \
  --repo YOUR_USERNAME/Qwen3-0.6B-Q4-GGUF
```

Your GGUF and model description will appear at `https://huggingface.co/YOUR_USERNAME/Qwen3-0.6B-Q4-GGUF`.

The upload only happens when you pass `--publish`. Check the original model's license before distributing any quantized weights.

[More examples, troubleshooting, and other models](docs/MAKE_YOUR_OWN_Q4.md)

## What does Q4Fit add?

The example above produces **standard Q4_K**: a smaller, 4-bit version of the original model. Q4Fit also supports **mixed precision**—for example, keeping most weights at Q4_K while storing a few selected weights at Q5_K.

Why bother? Some parts of a model are more sensitive to compression than others. Giving those parts more bits may keep the quantized model closer to the original, without making the entire file much larger. But the right choice depends on the model; a custom quant is **not automatically better**.

For a concrete example, Q4Fit includes a six-tensor recipe tested on Phi-4-mini:

```bash
python scripts/publish_my_q4.py \
  --model microsoft/Phi-4-mini-instruct \
  --llama-cpp ../llama.cpp \
  --recipe recipes/phi4mini-v07-6tensor.txt \
  --output-dir my-phi-q4fit
```

Add `--publish --repo YOUR_USERNAME/Phi-4-mini-Q4Fit-GGUF` to upload it. Recipes are **model-specific**. Do not apply this Phi recipe to a different model.

Q4Fit also supports compatible MoE expert weights. The [Inkling 2 Mini mixed GGUF](benchmarks/INKLING2MINI_PILOT.md) is a working Q3/Q4/Q5 example, with no claim that it improves answer quality.

For your own experiments, Q4Fit can compare a quant against the original model, test individual weight changes, and search for combinations that fit a file-size limit. These tools need a high-precision GGUF and sample text for evaluation. See [quantization search](docs/TENSOR_OPTIMIZER.md) and [tensor caching](docs/CACHE.md).

## What have we measured?

| Model | Extra file size over Q4_K | Difference from the original model on held-out text |
| --- | ---: | --- |
| Qwen3-4B | +0.28% | 16.0% less difference on WikiText; 13.7% less on code |
| Phi-4-mini-instruct | +0.388% | 4.59% less difference on WikiText; 11.60% less on code |

These figures use **KLD**, a measure of how closely a quantized model's next-word predictions match its high-precision original. Lower is better **for that measure**. The tests have **not shown a reliable improvement in real task accuracy**. A smaller KLD is not a guarantee of better answers.

Read the [Qwen3 results](benchmarks/QWEN3_4B_TENSOR_SEARCH.md), [Phi-4-mini results](benchmarks/PHI4MINI_EVAL.md), or the [complete research log](docs/RESEARCH_LOG.md), including unsuccessful experiments.

Q4Fit is an experimental wrapper around upstream quantization tools, not a new training method or a claim of universally superior Q4 models.

**License:** MIT.
