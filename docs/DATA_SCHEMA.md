# KD Parquet Schema (Top-K + Other)

Per token step (after projection + dedup):
- `prompt_idx`: int64 — prompt grouping id
- `step_idx`: int32 — token position within prompt (0-based)
- `teacher_sample_id`: int32 — student tokenizer id of sampled teacher token (first-subtoken)
- `teacher_sample_text`: string — raw teacher token text
- `topk_ids`: list<int32>[K] — projected & deduped student token ids
- `topk_logprobs`: list<float32>[K] — log p for each id (deduped via LSE)
- `other_logprob`: float32 — log(1 - sum(exp(topk_logprobs)))
- `is_struct_token`: bool — JSON-ish token heuristic
- `template_hash`: string — hash of prompt template
- `tokenizer_hash`: string — hash of tokenizer vocab sample

**Invariants**
- `sum(exp(topk_logprobs)) ∈ (0, 1)`; else row is dropped
- lengths of `topk_ids` ≡ `topk_logprobs` (≤ `MAX_TOPK`)
- `teacher_sample_id` derives from first subtoken of `teacher_sample_text`
