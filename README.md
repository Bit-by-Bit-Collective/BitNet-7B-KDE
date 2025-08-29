# BitNet-7B-KDE (Knowledge Distillation Engine)
## BitNet-7B PoC — KD Distillation (Mini Training + 7B Dry-Run)

> A practical, Colab-friendly pipeline that validates a BitNet-style transformer with **ternary weights**, **A8→A4 activation flip**, and **knowledge distillation (Top-K + Other)** from a locked teacher. It trains a compact “mini” model end-to-end and performs a **7B forward-pass dry-run** for memory checks.

<div align="center">

  <!-- Row 1: Badges (same height for perfect alignment) -->
  <a href="https://colab.research.google.com/github/xgrayfoxss21/BitNet-7B-KDE/blob/main/notebooks/Colab_Bootstrap.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run in Colab" height="28">
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/xgrayfoxss21/BitNet-7B-KDE/actions">
    <img src="https://img.shields.io/badge/status-experimental-orange" alt="status" height="28">
  </a>

  <br><br>

  <!-- Row 2: Links -->
  <a href="https://bit.foxesden.xyz/">🌐 Project Site</a>
  &nbsp;•&nbsp;
  <a href="https://discord.gg/Sefg6cte">💬 Discord</a>

</div>

---

<div align="center">

## 🙌 Support

If this project helps you, consider supporting development.

**Donations**

<code>qqnmnnu8x7a9gvh3vd5q2f8n2z4gfdz54u4hp7f8nx</code>

<!-- Optional: if this is actually BTC, use a bitcoin: URI -->
<!-- <a href="bitcoin:bc1_your_address_here?label=BitNet-7B-KDE&message=Support%20the%20project">Send via wallet</a> -->

<sub><em>Heads-up: addresses starting with <code>qq…</code> are typically Bitcoin Cash (BCH) CashAddr.  
BTC Address.</em></sub>

</div>

---

## Highlights

- **Pluggable teacher baseline (deterministic/greedy).** DeepSeek in examples; swap any provider that returns logprobs/top-k.
- **KD traces → Parquet.** Persist **Top-K tokens + logprobs** plus the residual **“Other”** probability mass.
- **Tokenizer projection with de-dup.** Map teacher candidates via **first-subtoken**; merge collisions via **log-sum-exp**.
- **Mini BitNet student.** Ternary weights (STE) with **A8 activations** that **flip to A4** by real token budget.
- **Aligned losses.** Temperature-matched **KL(Top-K+Other)** + **CE** + **format loss** for JSON-like structure.
- **Strict next-token training.** Predict *t+1* from context ≤ *t* (no label leakage).
- **Stability baked-in.** Autocast + GradScaler, causal + key-padding masks, safe padding (invalid ⇒ **P(other)=1**).
- **7B dry-run.** Forward-only memory check and A8→A4 flip validation.
- **QEI metrics.** Quick **quality-efficiency** proxy vs teacher; replace placeholder with your benchmark later.


---

## Quickstart

### Option A — Colab (recommended for first run)

1. Open:  
   **https://colab.research.google.com/github/xgrayfoxss21/BitNet-7B-KDE/blob/main/notebooks/Colab_Bootstrap.ipynb**
2. Run all cells. The notebook will:
   - Mount Google Drive (default: `/content/drive`)
   - Install dependencies
   - Read your `.env` (or prompt for a teacher API key)
   - Run **teacher baseline → KD collection → training → eval → 7B dry-run**
3. Artifacts default to:  
   `/content/drive/MyDrive/bitnet_poc/`  
   (Override in `.env` → `DRIVE_ROOT`)

### Option B — Local / Server

```bash
git clone https://github.com/xgrayfoxss21/BitNet-7B-KDE.git
cd BitNet-7B-KDE
```

# 1) Create and edit your env
```bash
cp .env.example .env

# (fill in API keys, PROVIDER, storage backend, DRIVE_ROOT, etc.)
```

# 2) Install
```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

# 3) Run common tasks
```bash
make teacher   # deterministic baseline
```
```bash
make collect   # KD traces → parquet
```
```bash
make train     # mini BitNet training
```
```bash
make eval      # eval + QEI report
```
```bash
make dryrun    # 7B forward-pass memory test
```

Multi-Provider API Support

Choose a provider in .env:

```env
PROVIDER=openai | anthropic | groq | aimlapi | gemini
```

Set the matching *_API_KEY, and optionally *_BASE_URL for OpenAI-compatible endpoints.

You can separately choose a TEACHER_PROVIDER / TEACHER_MODEL for KD/baseline, so you can chat with one provider but distill from another (that supports logprobs/top-k).

See: docs/EVALUATION.md
 (teacher baseline details) and docs/ARCHITECTURE.md

Storage Backends (Drive / Cloud / DB)

Configuration is .env-driven. Out of the box, Google Drive is used in Colab:

```env
AUTO_MOUNT_GDRIVE=1
GDRIVE_MOUNT_POINT=/content/drive
DRIVE_ROOT=/content/drive/MyDrive/bitnet_poc
```

You can switch to other targets by filling the relevant section in .env:

Google Drive / OneDrive / Dropbox / iCloud / Box / Nextcloud

Amazon S3 (S3_*), Custom WebDAV (WEBDAV_*)

Firebase / AWS Amplify / Supabase / MongoDB Atlas / Amazon DynamoDB

The helper scripts/storage.py reads only .env and prepares paths.

Full details: docs/STORAGE.md
 (layout, env variables, examples).

Outputs

Defaults (can be changed via .env):

```pgsql
DRIVE_ROOT/
  checkpoints/
    mini_bitnet_step_*.pt
    mini_bitnet_step_*_health.json
  data/
    kd_shard_000.parquet
  reports/
    teacher_baseline.json
    mini_evaluation_report.json
    pipeline_summary.json
  logs/
```

Makefile Targets

install — install from requirements.txt

teacher — deterministic teacher baseline (greedy)

collect — KD traces (Top-K + Other) to Parquet

train — mini BitNet training (KD + CE + format losses)

eval — eval loop + QEI report

dryrun — 7B forward-pass memory test

ensure_dirs — storage init from .env

check_env — quick env validation

clean — clean caches

All targets auto-source .env. See Makefile for details.

Evaluation & QEI

We report tokens/sec and a quick QEI proxy:

QEI=Ms​/Mt​Qs​/Qt​​QEIspeed​=QEI×TPSt​TPSs​​

Where Qs/Qt are student/teacher quality scores (placeholder in the PoC).
Swap in real scores from LiveBench / tool-use / system benchmarks. See docs/EVALUATION.md

Methodology Highlights

Next-token alignment (KD & CE)

Temperature-matched KL (student/teacher at same 𝜏τ)

Safe KD padding: invalid step ⇒ P(other)=1

Causal + key-padding masks (no attending to PAD)

STE for ternary weights & fake-quant activations

A8→A4 activation flip triggered by real seen tokens

Hyperparams, flip policy, and stability tips: docs/TRAINING_GUIDE.md

Docs

Architecture: docs/ARCHITECTURE.md

Training Guide: docs/TRAINING_GUIDE.md

Evaluation: docs/EVALUATION.md

Storage Backends: docs/STORAGE.md

Testing Plan: docs/TESTING.md

Benchmarks Plan: docs/BENCHMARKS.md

Release Process: docs/RELEASE_PROCESS.md

Roadmap: docs/ROADMAP.md

Troubleshooting

401/429 — check your API key & quotas; reduce request rate; shrink prompt sets.

CUDA OOM — reduce batch_size, MAX_SEQ_LEN, or model dims; ensure GPU runtime in Colab.

NaNs — guards are in place; if they persist, inspect KD parquet for malformed rows.

Slow CPU runs — expected; use GPU runtime.

Roadmap

Scale KD data to 30–40M tokens

Multi-GPU training (ZeRO-3 or equivalent)

Full 7B training

Replace placeholders with LiveBench / tool-use / system benchmarks

Gate against teacher success thresholds

See docs/ROADMAP.md
 for v0 → v1 milestones.
```makefile
::contentReference[oaicite:0]{index=0}

                                 ┌───────────────────────────┐
                                 │         .env             │
                                 │  (keys, storage, model)  │
                                 └────────────┬──────────────┘
                                              │
                                              v
┌───────────────────────────┐     ┌───────────────────────────┐
│     User / Colab / CLI    │     │       Makefile targets    │
│  (Colab nb or terminal)   │───▶ │  teacher | collect | ...  │
└─────────────┬─────────────┘     └────────────┬──────────────┘
              │                                  │
              │                                  │ invokes
              │                                  v
              │               ┌───────────────────────────────────────────┐
              │               │                  scripts/                 │
              │               │  run_teacher_baseline.py   (teacher)     │
              │               │  collect_kd_traces.py      (collect)     │
              │               │  train_mini_bitnet.py      (train)       │
              │               │  eval_and_qei.py           (eval)        │
              │               │  dry_run_7b_memory.py      (dryrun)      │
              │               └───────────────┬──────────────────────────┘
              │                               │ imports / calls
              │                               v
              │               ┌───────────────────────────────────────────┐
              │               │               src/bitnet/                 │
              │               │  ├─ apis/provider_client.py  (HTTP AI)    │
              │               │  ├─ data.py                 (Parquet/DS)  │
              │               │  ├─ losses.py               (KD/CE/format)│
              │               │  ├─ models.py               (BitNet mini) │
              │               │  ├─ qei.py                  (metrics)     │
              │               │  ├─ storage.py              (paths/cache) │
              │               │  └─ utils/env.py            (env parsing) │
              │               └───────────────┬──────────────────────────┘
              │                               │
              │                               │ uses
              │                               v
              │               ┌───────────────────────────────────────────┐
              │               │            Provider backends              │
              │               │  OpenAI | Anthropic | Groq | AIMLAPI |   │
              │               │  Gemini  (via provider_client.py)        │
              │               └───────────────────────────────────────────┘
              │
              │ writes/reads
              v
┌─────────────────────────────────────────────────────────────────────────┐
│                            Storage backends                             │
│  (resolved in storage.py via .env)                                      │
│   • Google Drive (Colab default)  • S3 / WebDAV / OneDrive / Dropbox    │
│   • iCloud / Box / Nextcloud      • DBs: Supabase / Firebase / DynamoDB │
│                                                                         │
│ Folder layout (example):                                                │
│   DRIVE_ROOT/                                                           │
│     ├─ checkpoints/   mini_bitnet_step_*.pt, *_health.json              │
│     ├─ data/          kd_shard_000.parquet                              │
│     ├─ reports/       teacher_baseline.json, mini_evaluation_report.json│
│     └─ logs/                                                          │
└─────────────────────────────────────────────────────────────────────────┘


                ▲                                                   ▲
                │                                                   │
   eval_and_qei.py reads reports + model            train_mini_bitnet.py emits checkpoints
   qei.py computes QEI/QEI_speed                    collect_kd_traces.py emits parquet
   and writes reports/                              run_teacher_baseline.py writes baseline

```

