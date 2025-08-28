# 🚀 __Quickstart (Colab)__

Open Google Colab.

Create a new notebook and paste the contents of colab/bitnet_poc_colab.py into a single cell.

Run all. The script will:

Mount Google Drive at /content/drive

Install dependencies

Prompt for your DeepSeek API key

Run the teacher baseline, KD collection, training, evaluation, and 7B dry-run

Artifacts are saved to:
/content/drive/MyDrive/bitnet_poc/

Required keys

DEEPSEEK_API_KEY (prompted at runtime; used for teacher baseline & KD trace collection)

OPENROUTER_API_KEY (optional)
--------------------------------------------------------------------------------------------
# 🧰 __Requirements__

GPU strongly recommended (A100/H100 ideal). CPU works for the PoC but will be slow.

Internet access (only for teacher baseline & KD collection).

Google account (to use Drive & Colab).
--------------------------------------------------------------------------------------------
# 🗂 __Outputs__

All paths under /content/drive/MyDrive/bitnet_poc/:

teacher_baseline.json — locked baseline (decode TPS, prompts, usage)

data/kd_shard_000.parquet — KD traces (Top-K + other)

checkpoints/mini_bitnet_step_*.pt — model checkpoints
checkpoints/mini_bitnet_step_*_health.json — training health+token budget

mini_evaluation_report.json — eval runs + QEI metrics

pipeline_summary.json — final status & checklist
-------------------------------------------------------------------------------------------
#⚙️ __Tunables (inside the script)__

Mini model config: MINI_CONFIG (layers, dims, heads)

7B config: FULL_7B_CONFIG (forward-only demo)

Training: TOTAL_STEPS, LOG_INTERVAL, CHECKPOINT_INTERVAL, LR schedule, weight decay

KD: temperature tau=1.3, CE weight, format loss weight

Flip trigger: token-budget fraction (default 0.9 of BUDGET_TOKENS)

Tokenizer: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (demo)
------------------------------------------------------------------------------------------
# 🧪 __Evaluation & QEI__

The eval loop runs a few prompts and reports tokens/sec (student).

QEI (efficiency proxy):


QEI=memorystudent​/memoryteacher​qualitystudent​/qualityteacher​​andQEIspeed​=QEI×TPSteacher​TPSstudent​


Quality is a placeholder (0.75 vs 1.0) in the PoC; swap in your benchmark score when ready.
-------------------------------------------------------------------------------------------
# 📈 __Methodology highlights__

Next-token alignment for KD & CE

Temperature-matched KL (student & teacher at same τ)

Safe padding in KD: when a step is invalid, P(other)=1 to avoid NaNs

Causal + key-padding masks so the model never attends to PAD

STE everywhere quantized (weights and activations)

A8→A4 flip triggered by actual seen tokens
------------------------------------------------------------------------------------------
# 🧯 __Troubleshooting__

401/429 from teacher API: verify DEEPSEEK_API_KEY, reduce request rate, or shorten eval_prompts/train_prompts.

CUDA OOM: reduce batch_size, max_seq_len, or model dims; ensure Colab is on a GPU runtime.

NaNs in loss: the script guards invalid KD steps; if they persist, inspect KD parquet for malformed rows.

Slow runs on CPU: expected—switch to GPU runtime.
-----------------------------------------------------------------------------------------
# 🧭 __Roadmap to production__

Scale KD data to 30–40M tokens

Multi-GPU training (DeepSpeed ZeRO-3 or similar)

Full 7B training

Replace placeholder evals with LiveBench/tool-use/system benchmarks

Gate against the locked teacher baseline
----------------------------------------------------------------------------------------
# 🔒 __Notes on usage__

This PoC calls a hosted teacher (DeepSeek) to obtain logprobs/Top-K. Ensure you have permission and follow provider ToS.

The script saves artifacts to Google Drive by default.
