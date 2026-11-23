# Make your own Q4 model and put it on Hugging Face

Create a GGUF from original Hugging Face weights, then publish it to your own model repository. The walkthrough uses Qwen3-0.6B so you can try the process with a small model.

You need a Linux computer, enough free disk space for the original model **and** two GGUF files, an internet connection, and a Hugging Face account. Start with a small model. This guide uses **Qwen3-0.6B**. You do **not** need a GPU to make the GGUF; CPU quantization is slower.

## 1. Install the tools (once)

Run these commands in an Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-venv
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-quantize -j 4
git clone https://github.com/bettercallcaleb/q4fit
cd q4fit
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

## 2. Make your first Q4

From the `q4fit` directory, run:

```bash
python scripts/publish_my_q4.py --model Qwen/Qwen3-0.6B --llama-cpp ../llama.cpp --output-dir my-first-q4
```

The program downloads the **original model**, converts it to GGUF, and makes `my-first-q4/Qwen3-0.6B-Q4_K.gguf`. The command prints **GGUF READY** when the file is complete. The first run downloads model weights and builds the GGUF. No Hugging Face upload happens yet.

**What you made:** Q4_K stores most model weights with roughly four bits instead of the original 16 bits, reducing the file size. Answers can change; quantization does not train the model.

## 3. Upload your Q4 to Hugging Face

Make a free account at https://huggingface.co/join. Open https://huggingface.co/settings/tokens and create a **write** token. Keep it private. In your terminal, run:

```bash
hf auth login
```

Paste your token when asked. Then replace `YOUR_USERNAME` below with your Hugging Face username:

```bash
python scripts/publish_my_q4.py --model Qwen/Qwen3-0.6B --llama-cpp ../llama.cpp --output-dir my-first-q4 --publish --repo YOUR_USERNAME/Qwen3-0.6B-Q4-GGUF
```

Your model page will be:

`https://huggingface.co/YOUR_USERNAME/Qwen3-0.6B-Q4-GGUF`

The program uploads the GGUF file, a simple model description, and a receipt showing exactly which original model and file hash were used. **Before sharing publicly**, check the original model's license and redistribution terms. If you want to test privately first, add `--private`.

## 4. Use a different model

Change `--model` to another Hugging Face model containing original `safetensors` weights supported by your llama.cpp version. Also change the output folder and repo name.

```bash
python scripts/publish_my_q4.py --model microsoft/Phi-4-mini-instruct --llama-cpp ../llama.cpp --output-dir my-phi-q4 --publish --repo YOUR_USERNAME/Phi-4-mini-Q4-GGUF
```

Bigger models need more disk space and take longer. If conversion fails, update llama.cpp and its Python dependencies. Not every model architecture is supported.

## 5. Optional: use Q4Fit's special mixed-Q4 recipe

The default command makes a standard **Q4_K** model. With Q4Fit, you can also choose different precision for selected weights. A recipe must match the **specific model architecture**. Do not reuse the Phi recipe on Qwen, or vice versa.

For the **exact official Phi-4-mini-instruct** model, try the published, tested recipe:

```bash
python scripts/publish_my_q4.py --model microsoft/Phi-4-mini-instruct --llama-cpp ../llama.cpp --recipe recipes/phi4mini-v07-6tensor.txt --output-dir my-phi-q4fit --publish --repo YOUR_USERNAME/Phi-4-mini-Q4Fit-GGUF
```

That recipe improved average next-token similarity to the original Phi model on two small held-out samples. It **has not** been proven to improve overall answer quality. For any other model, use the ordinary Q4_K recipe first.

## Common questions

**Do I need an RTX 4090?** No. This beginner workflow uses CPU tools. A GPU can help with running the resulting model, but is not required to create it.

**Will this cost money?** These commands use open-source local tools. Hugging Face accounts have free options, but storage, access, and upload policies may change.

**Why is it downloading again when I publish?** Using the same `--output-dir` lets the Hugging Face downloader reuse existing files and the converter skip its completed F16 GGUF. Quantization is rerun to verify the requested output.

**Can I upload someone else's model?** Only if its license and applicable terms allow redistribution. Always credit the original model.

**What if `hf` is missing?** Activate the virtual environment again with `. .venv/bin/activate` and run `python -m pip install -U huggingface-hub`.

**What if I see "No original safetensors found"?** Choose the model's original-weight repository, not an existing GGUF repository.

**What does the script publish?** Only when `--publish` is present: the GGUF, model card, and reproducibility receipt. It does not publish the original full-precision weights.

## Want the advanced automatic search?

For more control, [Q4Fit's search tools](TENSOR_OPTIMIZER.md) test precision changes against the original model using representative text. You do not need them to produce a standard Q4, and their output is not a guarantee of better responses.
