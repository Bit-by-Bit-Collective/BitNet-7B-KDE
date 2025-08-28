# Benchmarks & Gates (v1+)

## Phases

**v0 (PoC)**
- Throughput (TPS), memory, placeholder quality (0.75).

**v1 (Internal)**
- Add small, cheap suites:
  - Instruction: MT-Bench Lite / AlpacaEval-Lite
  - Reasoning: GSM8K-mini, MATH-mini
  - Tool-use: schema validation exact-match on synthetic tools
  - Format: JSON schema-valid rate

**v2 (External)**
- LiveBench slices, function-calling leaderboards, latency tail metrics.

## Gates (suggested)

- **Functional**: JSON/tool schema validity ≥ 95% on eval set.
- **KD parity**: Q_s ≥ 0.8× Q_t on chosen suites.
- **QEI**: QEI ≥ 1.0 (better quality-per-memory than teacher).
- **QEI_speed**: ≥ 1.0 when using the same hardware class.

## Reporting

- `reports/mini_evaluation_report.json`: TPS, memory, QEI.
- Add `reports/benchmarks_*.json` once harness is wired.
