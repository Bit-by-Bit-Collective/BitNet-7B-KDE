```
┌──────────────────────────────────────────────────────────────────┐
│ 1) Get the code                                                  │
│    ▸ Fork / Star the repo                                        │
│    ▸ git clone https://github.com/<you>/BitNet-7B-KDE.git        │
│    ▸ cd BitNet-7B-KDE                                            │
└───────────────┬──────────────────────────────────────────────────┘
                │
                v
┌──────────────────────────────────────────────────────────────────┐
│ 2) Configure `.env`                                              │
│    ▸ cp .env.example .env                                        │
│    ▸ Fill storage (DRIVE_ROOT or S3/WebDAV/etc.)                 │
│    ▸ Pick PROVIDER & TEACHER_PROVIDER (OpenAI/Anthropic/…)       │
│    ▸ Add API keys (*_API_KEY), optional *_BASE_URL               │
│    ▸ Keep TOKENIZER_NAME/TEMPLATE or adjust                      │
└───────────────┬──────────────────────────────────────────────────┘
                │
          ┌─────┴─────────┐
          │               │
          v               v
┌───────────────────┐  ┌───────────────────────────────────────────┐
│ 3A) Colab route   │  │ 3B) Local/Server route                    │
│  ▸ Open notebook  │  │  ▸ python -m pip install -r requirements  │
│    notebooks/...  │  │  ▸ make install                           │
│  ▸ Run all cells  │  │  ▸ make ensure_dirs (mount & create dirs) │
│  (auto mount GDrive│  │                                           │
│   + run pipeline)  │  │  (Use GPU if available)                   │
└───────┬───────────┘  └───────────────────────┬───────────────────┘
        │                                      │
        v                                      v
┌──────────────────────────────────────────────────────────────────┐
│ 4) Execute tasks (both paths)                                     │
│    ▸ make teacher   # deterministic teacher baseline              │
│    ▸ make collect   # KD traces (Top-K + Other) → parquet         │
│    ▸ make train     # mini BitNet training (KD + CE + format)     │
│    ▸ make eval      # simple eval + QEI/QEIspeed                  │
│    ▸ make dryrun    # 7B forward-pass memory test                 │
│    (You can override .env at CLI: make train TOTAL_STEPS=200 …)   │
└───────────────┬──────────────────────────────────────────────────┘
                │
                v
┌──────────────────────────────────────────────────────────────────┐
│ 5) Inspect outputs (from .env STORAGE)                            │
│   DRIVE_ROOT/                                                     │
│     ├─ checkpoints/   mini_bitnet_step_*.pt, *_health.json        │
│     ├─ data/          kd_shard_000.parquet                        │
│     ├─ reports/       teacher_baseline.json, mini_evaluation_...  │
│     └─ logs/                                                    │
└───────────────┬──────────────────────────────────────────────────┘
                │
                v
┌──────────────────────────────────────────────────────────────────┐
│ 6) Iterate                                                         │
│   ▸ Tweak hyperparams (.env or CLI overrides)                      │
│   ▸ Re-run train/eval                                              │
│   ▸ Swap providers / models / storage if desired                   │
│   ▸ Commit & push                                                  │
└───────────────┬──────────────────────────────────────────────────┘
                │
                v
┌──────────────────────────────────────────────────────────────────┐
│ 7) CI & sharing (optional)                                         │
│   ▸ Push to GitHub → CI checks (lint/format/imports)               │
│   ▸ Share Colab notebook link, artifacts path, or reports          │
│   ▸ Open issues / PRs                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## First-Run Checklist

1. **Clone & enter the repo**

```bash
git clone https://github.com/<you>/BitNet-7B-KDE.git
cd BitNet-7B-KDE
```

2. **Create `.env`**

```bash
cp .env.example .env
# edit .env:
# - STORAGE: set DRIVE_ROOT or S3/WebDAV/etc.
# - PROVIDER + TEACHER_PROVIDER + *_API_KEY
# - TOKENIZER_NAME/TEMPLATE (optional)
```

3. **Install**

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

4. **Prepare storage**

```bash
make ensure_dirs
```

5. **Run the pipeline**

```bash
make teacher   # baseline (greedy)
make collect   # KD traces
make train     # mini training
make eval      # quick eval + QEI
make dryrun    # 7B memory test
```

> Tip: override any `.env` setting inline:

```bash
make train TOTAL_STEPS=200 KD_TAU=1.1
```

6. **Find artifacts** (by default)

```
DRIVE_ROOT/
  checkpoints/  data/  reports/  logs/
```

7. **Iterate**
   Change hyperparameters or providers in `.env` (or via CLI overrides) and re-run `train`/`eval`. Switch storage backends as needed.

---

## Where to look if stuck

* **Colab / Drive** mounting or paths → `docs/STORAGE.md`
* **Provider/API** setup & baseline → `docs/EVALUATION.md`
* **Training hyperparams & flip policy** → `docs/TRAINING_GUIDE.md`
* **QEI details** → `docs/EVALUATION.md`
* **Troubleshooting** (OOM/NaNs/rate limits) → bottom of `README.md` and docs

You’re set!
