# Testing Strategy

We recommend `pytest` with small, fast unit tests to guard regressions.

## Unit Tests (high priority)

1) **Losses (`src/bitnet/losses.py`)**
   - KD KL matches expectation on toy distributions.
   - **Next-token alignment**: assert `logits[:, :-1]` vs `labels[:, 1:]`.
   - **Padding**: invalid positions force `P(other)=1` → finite loss.
   - Format loss only applied where `is_struct_token & valid`.

2) **Projection/Dedup (collect)**
   - First-subtoken rule: teacher token string → student id[0].
   - LSE dedup: two entries for same id merge to `logsumexp`.

3) **Masks (`models.py`)**
   - Causal + key padding masks: no attention to future or PAD.
   - Compare attention scores with/without mask on tiny tensors.

4) **Determinism**
   - With fixed seeds and CPU, ensure forward pass stable on a tiny input.
   - Hash of logits / activations within tight tolerance.

## Example Skeleton

