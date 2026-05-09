# NLP Assignment 04 — Run Guide

**IBA | Track 2, Option D | Ablation Study with Qwen3 + LoRA SFT**

---

## What this project does

Fine-tunes `Qwen3-0.6B` on two different reasoning datasets using LoRA (5 trials each),
evaluates against a baseline, then compares using `Kimi-K2-Instruct` as an LLM judge.

---

## Requirements

### Your machine needs:
- CUDA GPU (tested on RTX 3090 — 24 GB VRAM is plenty)
- Python 3.10+
- PyTorch with CUDA support

### One-time environment setup (do this before opening any notebook):

```bash
# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Jupyter so you can run the notebooks
pip install jupyter
```

Then open Jupyter:
```bash
cd /path/to/NLP_Project_2/notebooks
jupyter notebook
```

Each notebook installs its own Python dependencies in the first cell — you do not need to pre-install them.

---

## Environment variables

The notebooks call the Hugging Face OpenAI-compatible router to run the Kimi-K2 judge. You need an HF token with access to the configured inference provider.

**Create `.env` in this project folder** (the notebooks load it automatically):

```
HF_TOKEN=your_huggingface_token_here
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a read-only token is sufficient. Make sure `.env` is in the same folder as the notebooks.

> `.env` is listed in `.gitignore` — do not commit it.

---

## Notebook order — run them in sequence

### `00_baseline.ipynb` — Baseline evaluation
- Runs `Qwen3-0.6B` and `Qwen3-1.7B` locally on 10 reasoning prompts (no training)
- Scores responses with Kimi-K2 judge via the Hugging Face OpenAI-compatible router
- **Output files (needed by notebooks 01–03):**
  - `baseline_results.csv`
  - `prompts.json`
  - `gold_answers.json`
- **Approx time:** 20–40 min depending on GPU (runs two ~1–3 GB models sequentially)

---

### `01_sft_datasetA.ipynb` — SFT on Crownelius reasoning dataset
- Fine-tunes `Qwen3-0.6B` using LoRA in 5 ablation trials
- Dataset: `Crownelius/Opus-4.6-Reasoning-3300x` (~2,160 rows, general reasoning with CoT traces)
- After each trial: saves adapter, evaluates on 10 prompts, scores with judge
- **Output files:**
  - `trial_results_datasetA.csv`
  - `all_scores_datasetA.csv`
  - `best_trial_datasetA.json`
  - `adapters_datasetA/T1` through `adapters_datasetA/T5` (LoRA weights)
- **Approx time:** 3–6 hours total (5 trials × 30–60 min each on 3090)

---

### `02_sft_datasetB.ipynb` — SFT on Nvidia Nemotron math dataset
- Same model, same 5 trial configs as notebook 01 — **only the dataset differs**
- Dataset: `nvidia/Nemotron-SFT-Math-v3` (~2,160-row CoT-only subset)
- **Output files:**
  - `trial_results_datasetB.csv`
  - `all_scores_datasetB.csv`
  - `best_trial_datasetB.json`
  - `adapters_datasetB/T1` through `adapters_datasetB/T5` (LoRA weights)
- **Approx time:** 3–6 hours total

---

### `03_evaluation.ipynb` — Final comparison and report tables
- Loads the best adapter from each dataset (auto-detected from the JSON files above)
- Runs 3-way comparison: `base` vs `sft_A` vs `sft_B` on all 10 prompts
- Cross-domain eval: general-trained model on math prompts, math-trained on general
- Difficulty-stratified eval on Dataset A problems (easy / medium / hard)
- Prints all 5 report tables ready to paste into the assignment write-up
- **Output files:**
  - `comparison_3way.csv`
  - `cross_domain_eval.csv`
  - `difficulty_eval.csv`
- **Approx time:** 1–2 hours

---

## VRAM notes (RTX 3090 — 24 GB)

| Notebook | Peak VRAM usage |
|----------|----------------|
| 00 | ~4 GB (1.7B model, fp16) |
| 01 / 02 | ~6–8 GB (0.6B model, 4-bit + LoRA training) |
| 03 | ~6 GB (0.6B adapters, 4-bit inference) |

If you run out of VRAM between trials, restart the notebook kernel before the next one.

---

## If something breaks

- **`CUDA out of memory`** — restart kernel, run the failing cell again. The models are unloaded between trials.
- **`unsloth install fails`** — the install cell tries the CUDA-specific wheel first, then falls back to generic unsloth. If both fail: `pip install unsloth` manually in terminal.
- **HF router/provider rate limits** (judge calls) — there is a short `time.sleep(...)` between judge calls. If you still hit rate limits, increase the delay in the `judge_response` calls.
- **`FileNotFoundError: prompts.json`** — make sure your Jupyter working directory is `notebooks/`. Run `pwd` in a notebook cell to check; it should end in `.../notebooks`.
- **Adapter not found in notebook 03** — notebooks 01 and 02 must complete fully before running 03. Check that `best_trial_datasetA.json` and `best_trial_datasetB.json` exist in the `notebooks/` folder.

---

## Quick sanity check before starting

Run this in a new notebook cell to confirm everything is ready:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "n/a")
```

Expected output on 3090:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
```
