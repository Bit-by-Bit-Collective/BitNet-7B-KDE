# BitNet-7B-KDE — PoC v0 Roadmap

**Version:** v0 • **Status:** Experimental • **Scope:** End-to-end methodology proof in Colab + CI

---

## 1) Objective & Non-Goals

**Objective (v0):**
Prove the *method* end-to-end on a small model: collect KD traces (Top-K + Other), train a mini BitNet with ternary weights and an **A8→A4** activation flip, validate stability, run a **7B forward-pass dry-run**, and report efficiency via **QEI** — all reproducibly in Colab and CI.

**Non-Goals (v0):**

* Full 7B **training** (that’s v1+).
* Massive KD (30–40M tokens) — we target \~0.5–2M.
* Full benchmark suite (LiveBench/tool-use/system) — we keep a small eval + QEI proxy.
* Multi-cloud storage adapters beyond placeholders (Drive/local supported in v0).

---

## 2) Scope & Deliverables

### Must-Have (v0)

* **Teacher baseline** (deterministic/greedy) + JSON report.
* **KD trace collection**: Top-K tokens + logprobs + **“Other”** mass to Parquet.
* **Projection** to student tokenizer via **first-subtoken rule + log-sum-exp (LSE) dedup**.
* **Mini BitNet** (100–300M): STE ternary weights; **A8→A4** flip by token budget.
* **Losses**: KL (Top-K+Other), CE, **format loss**; strict **next-token alignment**.
* **Stability**: mixed precision (autocast + GradScaler), causal+pad masking, safe padding (`P(other)=1` on invalid).
* **7B dry-run**: forward pass + memory & **A8→A4** check (no training).
* **QEI** + **QEIspeed** proxy vs teacher.
* **Colab + CI** flows; secrets & paths via **`.env`**; Makefile targets:

  * `teacher | collect | train | eval | dryrun`

### Nice-to-Have (time-boxed)

* Resume-from-checkpoint in `train`.
* Optional W\&B logging (guarded by `WANDB_DISABLED`).

### Out-of-Scope (v0)

* Distributed / ZeRO training.
* Full storage adapters (S3, OneDrive, Dropbox, …).

---

## 3) Milestones & Exit Criteria

> Use the checkboxes to track progress. “Exit criteria” must be met to close a milestone.

### M0 — Repo Bootstrap ✅

* [ ] Code scaffold: `src/`, `scripts/`, `docs/`, `Makefile`, `pyproject.toml`, `requirements.txt`
* [ ] `.env(.example)` with Colab/Drive, providers, training toggles
* [ ] `storage.py` mounts Drive & creates dirs from `.env`
* **Exit:** repo clean clone runs `make ensure_dirs` successfully

### M1 — Providers & Baseline

* [ ] Multi-provider API client (OpenAI, Anthropic, Groq, AIMLAPI, Gemini)
* [ ] Deterministic **teacher baseline** JSON (decode TPS, usage)
* **Exit:** `${REPORTS_DIR}/teacher_baseline.json` produced on ≥1 provider

### M2 — KD Trace Collection

* [ ] `collect_kd_traces.py` writes **Parquet** (Top-K+Other, projection+LSE dedup)
* [ ] 1 shard with **≥ 500k tokens** (target 0.5–2M)
* **Exit:** `${DATA_DIR}/kd_shard_000.parquet` saved; invalid-row rate < **2%**

### M3 — Mini Training & Stability

* [ ] Mini BitNet (STE ternary, A8→A4 flip by seen tokens)
* [ ] Mixed precision + grad clipping; causal+key-pad masks
* [ ] Checkpoints + `*_health.json` (seen tokens, flip status)
* **Exit:**

  * Train **≥ 1M seen tokens** with **no NaNs**
  * Flip occurs at **≥ 90%** of `BUDGET_TOKENS`
  * Grad norm ≤ **1.0** (post-clip) across last **100** steps
  * Loss stable in last **200** steps

### M4 — Eval, QEI & 7B Dry-Run

* [ ] `eval_and_qei.py` (small eval, TPS, memory, QEI/QEIspeed)
* [ ] `dry_run_7b_memory.py` forward pass + A8→A4 at 7B
* **Exit:**

  * `${REPORTS_DIR}/mini_evaluation_report.json` saved, TPS computed
  * **QEI ≥ 0.20**, **QEIspeed ≥ 0.10** (quality placeholder 0.75 vs 1.0)
  * 7B forward pass success; peak memory logged; A8→A4 validated

### M5 — CI & Repro

* [ ] `.github/workflows/ci.yml` smoke run with `SKIP_EXTERNAL=1` green on PRs
* [ ] Colab quickstart from badge produces expected Drive outputs
* **Exit:**

  * CI completes < **15 min**, archives artifacts
  * Colab run creates all expected files under `${DRIVE_ROOT}`

**PoC v0 = DONE** when all the exit criteria above are met.

---

## 4) Suggested Timeline

* **Week 1:** M1–M2 (providers + baseline; KD shard 0.5–1M; schema sanity)
* **Week 2:** M3–M4 (mini training stability & flip; eval + QEI; 7B dry-run)
* **Week 3 (buffer):** M5 (CI polish, reproducibility, v1 planning)

---

## 5) Metrics to Track

* **Teacher:** avg decode TPS @ context 8k; latency distribution; \$/1k toks (manual)
* **KD quality:** invalid KD rows %; avg “Other” mass; top-k coverage
* **Training health:** loss curves (total/KL/CE/format), grad norm, NaN count, flip timing
* **Perf:** student tokens/sec; memory footprint (peak & steady)
* **QEI:**
  `QEI = (quality_student/quality_teacher) / (mem_student/mem_teacher)`
  `QEIspeed = QEI × (TPS_student/TPS_teacher)`
* **Repro:** success rate on Colab/CI from fresh clone

---

## 6) Risks & Mitigations

| Risk                                | Mitigation                                                                |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Provider logprobs/top-k unsupported | Gate per provider; use ones with logprobs; cache responses; backoff/retry |
| Rate limits / cost                  | Throttle; chunk prompts; cap tokens; cache KD shards                      |
| Tokenizer mismatch                  | First-subtoken rule + LSE-dedup; validate projection; unit tests          |
| NaNs / instability                  | Safe padding (`P(other)=1`), temp-matched KL, STE quant, MP+clip          |
| Colab GPU variance                  | Persist caches/checkpoints to Drive; conservative defaults; CPU fallback  |
| Secrets / compliance                | `.env` only; never commit keys; review ToS                                |

---

## 7) Team & Ownership (placeholders)

* **ML Research (KD/Model):** Losses, quantization, flip policy, eval/QEI
* **ML Infra (Pipelines/CI):** Makefile, CI, storage, secrets, env bootstrap
* **Data Eng (KD):** Provider adapters, schema validation, caching
* **DX/Docs:** Colab notebook, README, setup guide, contribution guidelines

---

## 8) Path to v1 (post-PoC)

* Scale KD to **30–40M tokens** (distributed collection, quotas, caching)
* **Multi-GPU training** (DeepSpeed ZeRO-3), optimizer/scaler tuning, activation-quant schedule
* **Full 7B training** (checkpoint sharding; artifact registry)
* **Benchmarks:** LiveBench, tool-use, system evals; replace quality placeholder
* **Storage adapters:** S3/OneDrive/Dropbox/Nextcloud; dataset registry & versioning
* **Observability:** W\&B or OpenTelemetry; dashboards & alerts
* **Security/Compliance:** Provider ToS gating; leakage red-team prompts

---

## 9) Definition of Done (v0)

* ✅ Reproducible **Colab** run producing KD shard, trained mini model, eval report, 7B dry-run
* ✅ **CI** smoke pipeline green; artifacts uploaded
* ✅ **No NaNs** in last N=200 steps; flip executed; checkpoints + health written
* ✅ **QEI ≥ 0.20**, **QEIspeed ≥ 0.10** (given quality placeholder)
* ✅ Docs: README badge, setup guide, CI design, **this roadmap**

