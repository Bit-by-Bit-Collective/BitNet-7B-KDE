# Step-by-Step: How to Use Everything

## 0) Prereqs

* Python 3.10+ (3.11 recommended)
* Git
* A GPU is strongly recommended for training (Colab GPU, or local CUDA)
* (Optional) API keys for your chosen teacher/provider (OpenAI / Anthropic / Groq / AIMLAPI / Gemini)

---

## 1) Clone the repo

```bash
git clone https://github.com/xgrayfoxss21/BitNet-7B-KDE.git
cd BitNet-7B-KDE
```

---

## 2) Create your `.env`

Copy the example and edit:

```bash
cp .env.example .env
```

Open `.env` and set:

* **Runtime & storage** (where outputs go)

  * `AUTO_MOUNT_GDRIVE=1` (Colab) or `0` (local/CI)
  * `DRIVE_ROOT=/content/drive/MyDrive/bitnet_poc` (Colab)
  * For local runs, set `DRIVE_ROOT=./artifacts` (or any path on your disk)

* **Provider & API**

  * `PROVIDER=openai` (or `anthropic|groq|aimlapi|gemini`)
  * Add your `*_API_KEY` for the providers you’ll use
  * For KD/teacher: set `TEACHER_PROVIDER` & `TEACHER_MODEL`

* **Training/Eval knobs**

  * `TOTAL_STEPS`, `BUDGET_TOKENS`, `KD_TAU`, etc.
  * `TOKENIZER_NAME` and `TEMPLATE` if you want to change them

> Tip: in Colab, you can also create the `.env` from a cell:
>
> ```python
> from pathlib import Path
> Path('.env').write_text("""PASTE YOUR .env CONTENTS HERE""")
> print("Wrote .env")
> ```

---

## 3) Install dependencies

```bash
make install
```

This installs `requirements.txt`. (If you want editable package semantics too, optionally: `pip install -e .`)

---

## 4) Ensure storage paths exist

This will (optionally) mount Google Drive in Colab **and** create all folders/caches specified in `.env`:

```bash
make ensure_dirs
```

Outputs go under:

* `${DRIVE_ROOT}/checkpoints`
* `${DRIVE_ROOT}/data`
* `${DRIVE_ROOT}/reports`
* `${DRIVE_ROOT}/logs`
* `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME` (all Drive-backed by default in Colab)

---

## 5) Run the pipeline targets (Makefile)

### (a) Teacher baseline (deterministic/greedy)

Captures decode TPS, tokens, etc., into `reports/teacher_baseline.json`.

```bash
make teacher
```

### (b) Collect KD traces

Queries teacher with sampling, extracts **Top-K tokens + logprobs** + **“Other”** mass, projects to student tokenizer, and saves a Parquet shard:

```bash
make collect
```

Produces `data/kd_shard_000.parquet`.

### (c) Train the mini BitNet

Runs mixed precision, **A8→A4** activation flip at your token budget fraction, checkpoints + health files:

```bash
make train
```

### (d) Evaluate & QEI

Small prompt set to get tokens/sec, estimate model memory and compute QEI vs teacher baseline:

```bash
make eval
```

Saves `reports/mini_evaluation_report.json`.

### (e) 7B Dry-run (memory only)

Forward-pass only sanity check for the 7B config + A8→A4 flip:

```bash
make dryrun
```

> You can override `.env` values on the command line:
>
> ```bash
> make train TOTAL_STEPS=200 KD_TAU=1.1
> ```

---

## 6) Colab: one-click & notebook flow

**Badge:**
[https://colab.research.google.com/github/xgrayfoxss21/BitNet-7B-KDE/blob/main/notebooks/Colab\_Bootstrap.ipynb](https://colab.research.google.com/github/xgrayfoxss21/BitNet-7B-KDE/blob/main/notebooks/Colab_Bootstrap.ipynb)

Inside the notebook:

1. **GPU runtime** → Runtime → Change runtime type → T4/A100, etc.
2. **Clone repo** (done in the notebook)
3. **Create `.env`** in the project root (paste your `.env` contents into the provided cell)
4. **Install**: `make install`
5. **Ensure dirs**: `make ensure_dirs` (this will mount Drive if `AUTO_MOUNT_GDRIVE=1`)
6. **Run steps**: `make teacher`, `make collect`, `make train`, `make eval`, `make dryrun`
7. **Outputs** will appear in your Drive under `${DRIVE_ROOT}`

---

## 7) Local usage (no Drive)

1. Set in `.env`:

```
AUTO_MOUNT_GDRIVE=0
DRIVE_ROOT=./artifacts
```

2. Then run:

```bash
make install
make ensure_dirs
make teacher
make collect
make train
make eval
make dryrun
```

Artifacts will be written to `./artifacts/...`

---

## 8) CI usage (GitHub Actions)

* File: `.github/workflows/ci.yml` (already in the repo)
* Add **`.env.ci`** (not committed; or generate inside CI with step `run: | echo "..."> .env`)

Recommend CI values:

```
AUTO_MOUNT_GDRIVE=0
DRIVE_ROOT=${{ github.workspace }}/artifacts
TOTAL_STEPS=1
EVAL_MAX_PROMPTS=1
KD_MAX_TOKENS_PER_PROMPT=8
SKIP_EXTERNAL=1
```

CI default flow:

* Install + static checks
* `make ensure_dirs`
* `make dryrun` (tiny dims) or `make train TOTAL_STEPS=1` for a smoke checkpoint
* Upload `${{ github.workspace }}/artifacts` as build artifacts

> If you want to call real external providers in CI, add repository **secrets** (e.g. `OPENAI_API_KEY`) and set `SKIP_EXTERNAL=0`. Using a self-hosted/GPU runner is recommended for heavier runs.

---

## 9) Where the files go (by default)

* **Teacher baseline** → `${REPORTS_DIR}/teacher_baseline.json`
* **KD traces (Parquet)** → `${DATA_DIR}/kd_shard_000.parquet`
* **Checkpoints** → `${CHECKPOINTS_DIR}/mini_bitnet_step_*.pt`
* **Health** → `${CHECKPOINTS_DIR}/mini_bitnet_step_*_health.json`
* **Eval/QEI** → `${REPORTS_DIR}/mini_evaluation_report.json`
* **Summary logs** → `${REPORTS_DIR}` and `${LOGS_DIR}`

You can change folder names/locations in `.env`:

```
CHECKPOINTS_DIR=
DATA_DIR=
REPORTS_DIR=
LOGS_DIR=
```

---

## 10) Switching providers (teacher & default)

In `.env`:

* **Default** client for general calls:

  ```
  PROVIDER=openai  # or anthropic | groq | aimlapi | gemini
  OPENAI_API_KEY=...         # add your key(s)
  ANTHROPIC_API_KEY=...
  GROQ_API_KEY=...
  AIMLAPI_API_KEY=...
  GEMINI_API_KEY=...
  ```

* **Teacher** (for KD + baseline):

  ```
  TEACHER_PROVIDER=openai
  TEACHER_MODEL=gpt-4o-mini
  TEACHER_API_KEY=           # optional; otherwise falls back to provider’s key
  TEACHER_TOP_LOGPROBS=20
  TEACHER_BASELINE_TEMPERATURE=0.0
  KD_TEACHER_TEMPERATURE=0.8
  ```

Make sure the provider/model supports **logprobs / top\_logprobs**. If not available, KD collection will fail or skip.

---

## 11) Tuning training

Common knobs (in `.env`):

* **Precision**: `TORCH_DTYPE=bf16|fp16|fp32`
* **FlashAttention** (A100/H100): `ENABLE_FLASH_ATTN=1`
* **Optimizer**: `LR`, `ADAM_*`, `WEIGHT_DECAY`
* **Schedule**: `SCHEDULER_TMAX`, `SCHEDULER_ETA_MIN`
* **Loop**: `TOTAL_STEPS`, `LOG_INTERVAL`, `CHECKPOINT_INTERVAL`
* **KD**: `KD_TAU`, `KD_CE_WEIGHT`, `FORMAT_LOSS_WEIGHT`
* **Flip**: `BUDGET_TOKENS`, `FLIP_FRACTION` (A8→A4)
* **Data**: `MAX_SEQ_LEN`, `MAX_TOPK`, `TRAIN_BATCH_SIZE`

Override ad-hoc on the command line if you want:

```bash
make train TOTAL_STEPS=200 KD_TAU=1.1
```

---

## 12) Resuming & crash safety

* Checkpoints are **incremental** (`*_step_XXXX.pt`) with an accompanying `*_health.json`.
* Re-running `make train` will continue from scratch unless you add resume logic; you can load the latest checkpoint manually if you add that small extension.
* Caches (`TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`) live on Drive to survive Colab restarts.

---

## 13) Troubleshooting

* **401/403/429 from provider**:
  Wrong or missing API key, or rate limit. Verify `*_API_KEY` in `.env`, reduce concurrency, or trim your prompt set.

* **NaNs in loss**:
  KD loader already guards invalid steps with `P(other)=1)`. If NaNs persist, inspect `data/kd_shard_000.parquet` for malformed rows.

* **CUDA OOM**:
  Lower `TRAIN_BATCH_SIZE`, `MAX_SEQ_LEN`, or model dims; ensure Colab is set to GPU runtime.

* **No outputs / wrong folder**:
  Check `.env` paths (`DRIVE_ROOT`, subfolders). Run `make ensure_dirs` again to print resolved paths.

* **Colab not seeing your `.env`**:
  Make sure the `.env` file is in the **repo root** of the Colab working directory (print `!pwd && ls -la` to confirm). Or write it in a cell (see Step 2 tip).

---

## 14) What’s included vs placeholders

* ✅ Google Drive (Colab) + local paths are fully supported via `scripts/storage.py`.
* 🔜 Other storage backends (S3, OneDrive, Dropbox, etc.) have environment placeholders for future adapters; current code uses Drive/local paths.

---

## 15) Quick “do everything” checklist

**Colab**

1. Open notebook badge
2. Paste `.env`
3. `make install`
4. `make ensure_dirs`
5. `make teacher`
6. `make collect`
7. `make train`
8. `make eval`
9. (Optional) `make dryrun`

**Local**

1. `cp .env.example .env` → set `AUTO_MOUNT_GDRIVE=0`, `DRIVE_ROOT=./artifacts`
2. `make install && make ensure_dirs`
3. Run the targets in order

**CI**

1. Add `.github/workflows/ci.yml` (already in repo)
2. Create `.env.ci` in workflow (or set vars as steps)
3. Keep `SKIP_EXTERNAL=1` for fast smoke runs
4. Inspect uploaded `artifacts/`


