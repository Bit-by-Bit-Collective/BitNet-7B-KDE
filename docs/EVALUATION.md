# Evaluation

The PoC ships a lightweight eval loop to estimate throughput and a **Quality-Efficiency Indicator (QEI)**.

## Metrics

- **TPS (tokens per second)**: average across prompts
- **Memory (GB)**: `sum(param.numel() * element_size) / 1e9`
- **Quality**: placeholder (`student=0.75`, `teacher=1.0`) — replace with a real benchmark

## QEI Formulas

Let:
- `Q_s, Q_t` = quality (student, teacher)
- `M_s, M_t` = memory (student, teacher)
- `TPS_s, TPS_t` = decode tokens/sec (student, teacher)

Then:

\[
\text{QEI} = \frac{Q_s / Q_t}{M_s / M_t}
\]

\[
\text{QEI}_\text{speed} = \text{QEI} \times \frac{TPS_s}{TPS_t}
\]

Interpretation:
- QEI > 1 means better quality-per-memory than the teacher.
- `QEI_speed` bakes in throughput.

## Replacing the Quality Placeholder

Start simple:
- **Instruction-following**: MT-Bench Lite or AlpacaEval-Lite
- **Reasoning small suites**: GSM8K (subset), MATH (subset)
- **Tool-use**: schema-valid function calling exact-match
- **JSON format**: exact match / schema validation rate

Compute `Q_s` as normalized scores (0..1), and set `Q_t` to the teacher score on the same subset.

## Reporting

The script writes:
- `reports/mini_evaluation_report.json`
  - per-prompt tokens/sec
  - averages
  - QEI / QEI_speed

Replace the placeholder quality in `scripts/eval_and_qei.py` once your benchmark harness is ready.
