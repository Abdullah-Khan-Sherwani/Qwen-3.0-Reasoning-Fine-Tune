# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

IBA NLP Assignment 04 — ablation study fine-tuning `Qwen/Qwen3-0.6B` on two reasoning datasets using LoRA via Unsloth/TRL, then comparing the resulting adapters with `Kimi-K2-Instruct` as an LLM judge.

## Environment setup

```bash
# One-time: PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install jupyter
```

Each notebook installs its own dependencies in the first cell (unsloth, trl, peft, transformers, datasets, openai, python-dotenv, etc.).

**Required:** create `.env` in the project root:
```
HF_TOKEN=your_huggingface_token_here
```

Sanity-check GPU before running:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Notebook execution order

Run notebooks sequentially — each produces files the next depends on:

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| `00_baseline.ipynb` | Eval Qwen3-0.6B & 1.7B base; no training | `baseline_results.csv`, `prompts.json`, `gold_answers.json` |
| `01_sft_datasetA.ipynb` | 5-trial LoRA SFT on `Crownelius/Opus-4.6-Reasoning-3300x` | `trial_results_datasetA.csv`, `all_scores_datasetA.csv`, `best_trial_datasetA.json`, `adapters_datasetA/T1–T5/` |
| `02_sft_datasetB.ipynb` | Same 5-trial configs on `nvidia/Nemotron-SFT-Math-v3` | `trial_results_datasetB.csv`, `all_scores_datasetB.csv`, `best_trial_datasetB.json`, `adapters_datasetB/T1–T5/` |
| `03_evaluation.ipynb` | 3-way comparison, cross-domain eval, difficulty stratification | `comparison_3way.csv`, `cross_domain_eval.csv`, `difficulty_eval.csv` |

## Architecture

**Training (notebooks 01 & 02):**
- `FastLanguageModel.from_pretrained` loads base model in 4-bit quantization via Unsloth
- `FastLanguageModel.get_peft_model` wraps it with LoRA adapters
- `SFTTrainer` (TRL) does supervised fine-tuning on chat-templated text
- Dataset A formats rows as `<think>{thinking}</think>\n{solution}` inside the assistant turn for explicit CoT supervision; Dataset B uses solution-only
- After each trial: adapter saved, model switched to inference mode via `FastLanguageModel.for_inference`, evaluated on 10 fixed prompts, scored by Kimi judge

**5 ablation trials per dataset:**
| Trial | Variable | Config |
|-------|----------|--------|
| T1 | Data % | 25%, rank 16, q/k/v/o |
| T2 | Data % | 50%, rank 16, q/k/v/o |
| T3 | Data % | 100%, rank 16, q/k/v/o |
| T4 | LoRA width | 100%, rank 32, q/k/v/o + gate/up/down |
| T5 | LoRA depth | 100%, rank 64, q/k/v/o |

**Evaluation:**
- 10 fixed prompts: 5 general logic (G1–G5) + 5 math word problems (M1–M5)
- Gold answers from ChatGPT, stored in `gold_answers.json`
- Judge: `moonshotai/Kimi-K2-Instruct:novita` via HF OpenAI-compatible router (`https://router.huggingface.co/v1`)
- Scoring: 1–5 scale, parsed from `SCORE: <n>` in judge output with 3-retry logic

**Key compatibility note:** `load_best_model_at_end` is intentionally disabled in SFTConfig — it is incompatible with Unsloth 4-bit quantized models.

## VRAM / runtime

- Notebooks 01/02: ~6–8 GB VRAM, 3–6 hours per notebook (5 trials × 30–60 min on RTX 3090)
- Between trials: `del model, trainer; torch.cuda.empty_cache()` — restart kernel if OOM persists
- Notebook 03: loads best adapters from the JSON files auto-detected from 01 & 02

## Gitignored outputs

`adapters_datasetA/`, `adapters_datasetB/`, and `.env` are gitignored. CSV/JSON result files are tracked.
