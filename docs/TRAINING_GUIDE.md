# Training Guide

## Core Hyperparameters

- **Mini model** (`models.py` → `MiniBitNet`)
  - `dim=768`, `n_layers=12`, `n_heads=12`, `head_dim=64` (demo)
  - MLP expansion = `4×`
  - Embeddings + LM head in full precision
  - BitLinear layers: ternary weights with learned per-channel scales (STE)
- **Precision**: `TORCH_DTYPE=bf16|fp16|fp32` (`.env`)
- **Mixed precision**: autocast + GradScaler (fp16) enabled when CUDA
- **Optimizer**: AdamW (`lr=6e-4`, `betas=(0.9,0.95)`, `eps=1e-8`, `wd=0.1`)
- **Schedule**: CosineAnnealingLR (`T_max=TOTAL_STEPS`, `eta_min=6e-5`)
- **Clip**: `grad_norm=1.0`
- **Batch/Seq**: `TRAIN_BATCH_SIZE=4`, `MAX_SEQ_LEN=256`
- **Steps**: `TOTAL_STEPS=1000` (Colab-friendly)

## KD / Losses

- **KD temperature**: `KD_TAU=1.3`
- **CE weight**: `KD_CE_WEIGHT=0.25`
- **Format loss weight** (JSON-ish): `FORMAT_LOSS_WEIGHT=0.2`
- **Next-token alignment**: logit at t predicts token t+1
- **Teacher distribution**: Top-K logprobs + `other_logprob=log(1 - sum(exp(topk)))`
- **Projection**: teacher token strings → student tokenizer **first-subtoken**; duplicates merged via **log-sum-exp** (LSE).

## Activation Flip Policy (A8→A4)

- Global A8 activations during most of training
- Flip to A4 when `seen_tokens ≥ FLIP_FRACTION × BUDGET_TOKENS`
  - `.env`: `BUDGET_TOKENS=1_000_000`, `FLIP_FRACTION=0.9`
- Flip is one-way; you can experiment with a warmup schedule.

## Stability Tips

- Always align to next token (`logits[:, :-1]` vs `labels[:, 1:]`).
- Safe padding:
  - KD invalid positions force `P(other)=1` (zero Top-K mass) to avoid NaNs.
  - CE uses `ignore_index=-100`.
- Attention:
  - Combine **causal mask** with **key padding mask** so PAD tokens are never attended.
- AMP:
  - Enable autocast only on supported dtypes/devices.
  - Use GradScaler for fp16 to prevent underflow.
- If NaNs appear:
  - Lower LR (`3e-4`), increase `eta_min`, or turn off AMP as a probe.
  - Validate KD parquet mass invariants.

