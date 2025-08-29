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
