# Qwen3 Reasoning Fine-Tune — LoRA Ablation Study

**IBA · NLP with Deep Learning · Assignment 04 · Track 2, Option D**

Fine-tunes `Qwen/Qwen3-0.6B` on two distinct reasoning datasets via LoRA SFT (5 ablation trials each), then evaluates the resulting adapters against a baseline using `Kimi-K2-Instruct` as an LLM judge.

---

## Overview

| Stage | What happens |
|-------|-------------|
| **Baseline** | `Qwen3-0.6B` and `Qwen3-1.7B` run on 10 fixed reasoning prompts with no training; scored by Kimi judge |
| **SFT — Dataset A** | 5 LoRA trials on `Crownelius/Opus-4.6-Reasoning-3300x` (general reasoning + CoT traces) |
| **SFT — Dataset B** | Same 5 trial configs on `nvidia/Nemotron-SFT-Math-v3` (math-focused CoT) |
| **Evaluation** | 3-way comparison (base vs sft_A vs sft_B), cross-domain transfer, difficulty stratification |

The ablation varies two axes across the 5 trials:

| Trial | Data used | LoRA rank | Target modules | Purpose |
|-------|-----------|-----------|----------------|---------|
| T1 | 25 % | 16 | q, k, v, o | Data efficiency — low |
| T2 | 50 % | 16 | q, k, v, o | Data efficiency — mid |
| T3 | 100 % | 16 | q, k, v, o | Data efficiency — full |
| T4 | 100 % | 32 | q, k, v, o, gate, up, down | LoRA capacity — wider |
| T5 | 100 % | 64 | q, k, v, o | LoRA capacity — deeper |

---

## Requirements

- CUDA GPU · tested on RTX 3090 (24 GB VRAM)
- Python 3.10+

```bash
# PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install jupyter
```

Each notebook installs its own remaining dependencies (Unsloth, TRL, PEFT, Transformers, Datasets, etc.) in its first cell — no additional pre-installation needed.

---

## Environment setup

The notebooks call the Hugging Face OpenAI-compatible router to run the Kimi-K2 judge. Create `.env` in the project root before running anything:

```bash
HF_TOKEN=your_huggingface_token_here
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — read-only scope is sufficient.

---

## Notebooks — run in order

### `00_baseline.ipynb` · Baseline evaluation
Loads `Qwen3-0.6B` and `Qwen3-1.7B` locally (no training), runs both on 10 prompts, scores with Kimi judge.

**Produces** (required by all later notebooks):
- `baseline_results.csv`
- `prompts.json`
- `gold_answers.json`

*Runtime: ~20–40 min*

---

### `01_sft_datasetA.ipynb` · SFT on general reasoning
Runs 5 LoRA trials on `Crownelius/Opus-4.6-Reasoning-3300x` (2,160 rows with `problem`, `thinking`, `solution` fields). The `thinking` field is wrapped in `<think>` tags inside the assistant turn, giving the model explicit chain-of-thought supervision.

**Produces:**
- `trial_results_datasetA.csv`, `all_scores_datasetA.csv`, `best_trial_datasetA.json`
- `adapters_datasetA/T1` – `adapters_datasetA/T5` *(gitignored)*

*Runtime: ~3–6 hours (5 trials)*

---

### `02_sft_datasetB.ipynb` · SFT on math reasoning
Identical trial configs to notebook 01. Only the dataset changes: `nvidia/Nemotron-SFT-Math-v3` (math-only CoT subset, same 2,160-row slice). No `thinking` field — solution-only supervision.

**Produces:**
- `trial_results_datasetB.csv`, `all_scores_datasetB.csv`, `best_trial_datasetB.json`
- `adapters_datasetB/T1` – `adapters_datasetB/T5` *(gitignored)*

*Runtime: ~3–6 hours (5 trials)*

---

### `03_evaluation.ipynb` · Final report
Auto-detects the best adapter from each dataset via the JSON files above, then produces three analyses:

1. **3-way comparison** — base vs sft_A vs sft_B on all 10 prompts
2. **Cross-domain transfer** — general-trained model on math prompts, math-trained on general prompts
3. **Difficulty stratification** — Dataset A scores broken down by easy / medium / hard

**Produces:** `comparison_3way.csv`, `cross_domain_eval.csv`, `difficulty_eval.csv`

*Runtime: ~1–2 hours*

---

## Evaluation design

| Component | Detail |
|-----------|--------|
| Prompts | 10 fixed: 5 general logic (G1–G5) + 5 math word problems (M1–M5) |
| Gold answers | Generated via ChatGPT; stored in `gold_answers.json` |
| Judge model | `moonshotai/Kimi-K2-Instruct:novita` via HF router |
| Scoring scale | 1 = wrong, no reasoning · 3 = partial · 5 = correct, clear step-by-step |

---

## VRAM reference (RTX 3090 · 24 GB)

| Notebook | Peak VRAM |
|----------|-----------|
| 00 | ~4 GB (1.7B, fp16) |
| 01 / 02 | ~6–8 GB (0.6B, 4-bit + LoRA) |
| 03 | ~6 GB (0.6B adapters, 4-bit inference) |

If you hit OOM mid-trial, restart the kernel — the next trial reloads from the base checkpoint.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Restart kernel; models are unloaded between trials |
| `unsloth install fails` | The install cell falls back automatically; if both fail run `pip install unsloth` manually |
| HF router rate limits | Increase the `time.sleep(...)` delay between judge calls in `judge_response` |
| `FileNotFoundError: prompts.json` | Run notebook 00 first; Jupyter working directory must be the project root |
| Adapter not found in notebook 03 | Notebooks 01 and 02 must complete fully; check that `best_trial_datasetA.json` and `best_trial_datasetB.json` exist |

---

## Baseline results (reference)

| Model | Avg judge score (/ 5) |
|-------|-----------------------|
| Qwen3-0.6B (base) | 4.2 |
| Qwen3-1.7B (base) | 4.8 |
