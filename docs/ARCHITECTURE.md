# Architecture

This repo implements an end-to-end Knowledge Distillation (KD) pipeline for a BitNet-style model:
1) run a deterministic teacher baseline,
2) collect KD traces (Top-K + Other),
3) train a mini BitNet with ternary weights + A8→A4 activation flip,
4) evaluate/QEI,
5) run a 7B forward-pass dry-run for memory.

## High-Level Flow

```mermaid
flowchart TD
  A[.env] --> B[scripts/run_teacher_baseline.py]
  A --> C[scripts/collect_kd_traces.py]
  A --> D[scripts/train_mini_bitnet.py]
  A --> E[scripts/eval_and_qei.py]
  A --> F[scripts/dry_run_7b_memory.py]

  C -->|Parquet| G[(Drive: data/)]
  D -->|checkpoints + health.json| H[(Drive: checkpoints/)]
  E -->|reports + metrics| I[(Drive: reports/)]
  B -->|teacher_baseline.json| I

  subgraph "src/bitnet"
    P1[providers.py]:::code
    P2[data.py]:::code
    P3[losses.py]:::code
    P4[models.py]:::code
  end

  classDef code fill:#1e293b,stroke:#0ea5e9,color:#e2e8f0;

  B --> P1
  C --> P1
  C --> P2
  D --> P2
  D --> P3
  D --> P4
  E --> P4
