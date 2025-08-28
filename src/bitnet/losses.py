# scripts/losses.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _count_valid(labels_next: torch.Tensor) -> int:
    """Number of non-pad targets (labels != -100)."""
    return int((labels_next != -100).sum().item())


def kd_cross_entropy_loss(
    logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    tau: float = 1.3,
    ce_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Knowledge Distillation (Top-K + Other) + CE with next-token alignment.

    Shapes
    ------
    logits: [B, T, V]  (student logits for tokens 0..T-1)
    batch keys (aligned to teacher targets at t+1):
      teacher_sample_id : [B, T]     (targets; -100 for pads)
      topk_ids          : [B, T, K]  (teacher Top-K token ids in student vocab)
      topk_logprobs     : [B, T, K]  (teacher logprobs for Top-K)
      other_logprob     : [B, T]     (teacher logprob of "all other tokens")
    """
    B, T, V = logits.shape

    # --- Next-token alignment: predict token t+1 from context up to t ---
    logits_next = logits[:, :-1, :]                 # [B, T-1, V]
    labels_next = batch["teacher_sample_id"][:, 1:] # [B, T-1]
    topk_ids     = batch["topk_ids"][:, 1:, :]      # [B, T-1, K]
    topk_lps     = batch["topk_logprobs"][:, 1:, :] # [B, T-1, K]
    other_lp     = batch["other_logprob"][:, 1:]    # [B, T-1]

    # Valid mask (exclude padded labels)
    valid = (labels_next != -100)                   # [B, T-1]
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    # Temperature scaling (teacher & student)
    # Student log-probs at temperature tau
    log_probs_tau = F.log_softmax((logits_next / tau).float(), dim=-1)  # [B, T-1, V]

    # --- Teacher distribution over (Top-K + "other") with safe padding handling ---
    # Scale teacher logprobs by tau (equivalent to dividing logits by tau before softmax)
    topk_lps_tau  = topk_lps / tau                              # [B, T-1, K]
    other_lp_tau  = (other_lp / tau).unsqueeze(-1)              # [B, T-1, 1]
    teacher_logits = torch.cat([topk_lps_tau, other_lp_tau], -1)  # [B, T-1, K+1]

    # For invalid positions: set Top-K -> -inf, other -> 0 (prob=1), then softmax
    if (~valid).any():
        K = topk_ids.shape[-1]
        teacher_logits = teacher_logits.clone()
        teacher_logits[~valid, :K] = float("-inf")
        teacher_logits[~valid, K]  = 0.0

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # [B, T-1, K+1]
    teacher_probs     = teacher_log_probs.exp()

    # --- Student distribution reduced to (Top-K + "other") space ---
    # Guard against out-of-range ids (should not happen if projection is correct)
    topk_ids_safe = topk_ids.clamp_(min=0, max=V - 1)

    student_topk_logp = log_probs_tau.gather(-1, topk_ids_safe)      # [B, T-1, K]
    student_topk_p    = student_topk_logp.exp()
    student_other_p   = (1.0 - student_topk_p.sum(dim=-1, keepdim=True)).clamp_(1e-8, 1 - 1e-8)
    student_other_logp = student_other_p.log()

    student_combined_logp = torch.cat([student_topk_logp, student_other_logp], dim=-1)  # [B, T-1, K+1]

    # KL per token on valid positions: sum_y q(y) * (log q(y) - log p(y))
    kl_per_tok = (teacher_probs * (teacher_log_probs - student_combined_logp)).sum(dim=-1)  # [B, T-1]
    kl_loss = (kl_per_tok * valid_f).sum() / valid_count

    # Cross-entropy on next tokens (ignore padded labels)
    ce_loss = F.nll_loss(
        F.log_softmax(logits_next.float(), dim=-1).view(-1, V),
        labels_next.view(-1),
        reduction="mean",
        ignore_index=-100,
    )

    total = kl_loss + ce_weight * ce_loss
    metrics = {
        "kl_loss": float(kl_loss.detach()),
        "ce_loss": float(ce_loss.detach()),
        "valid_tokens": float(_count_valid(labels_next)),
    }
    return total, metrics


def format_loss(
    logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Auxiliary format loss: encourage the exact next structural tokens.

    Applies only on positions where the *target* (t+1) is marked structural.

    Inputs
    ------
    logits: [B, T, V]
    batch:
      teacher_sample_id : [B, T]
      is_struct_token   : [B, T]  (True if the *target* at t+1 is structural)
    """
    B, T, V = logits.shape
    logits_next = logits[:, :-1, :]                   # [B, T-1, V]
    labels_next = batch["teacher_sample_id"][:, 1:]   # [B, T-1]
    is_struct   = batch["is_struct_token"][:, 1:]     # [B, T-1]

    valid = (labels_next != -100)
    mask = (valid & is_struct)

    if not mask.any():
        zero = logits_next.new_zeros(())
        return zero, {"format_loss": 0.0, "format_count": 0.0}

    log_probs = F.log_softmax(logits_next.float(), dim=-1)
    labels_clamped = labels_next.clamp_min(0).unsqueeze(-1)
    tok_logp = log_probs.gather(-1, labels_clamped).squeeze(-1)  # [B, T-1]

    struct_logp = tok_logp[mask]
    struct_loss = -struct_logp.mean()               # unweighted mean for metric
    total = weight * struct_loss

    return total, {
        "format_loss": float(struct_loss.detach()),
        "format_count": float(mask.sum().item()),
    }


def combined_loss(
    logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    tau: float = 1.3,
    ce_weight: float = 0.25,
    format_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss = KD (Top-K + Other) + ce_weight * CE + format_weight * FormatLoss.
    """
    kd_ce_val, kd_ce_metrics = kd_cross_entropy_loss(logits, batch, tau=tau, ce_weight=ce_weight)
    fmt_val, fmt_metrics = format_loss(logits, batch, weight=format_weight)

    total = kd_ce_val + fmt_val
    metrics = {
        **kd_ce_metrics,
        **fmt_metrics,
        "total_loss": float(total.detach()),
    }
    return total, metrics


if __name__ == "__main__":
    # Minimal shape sanity check (random tensors)
    B, T, V, K = 2, 8, 50, 5
    logits = torch.randn(B, T, V)

    batch = {
        "teacher_sample_id": torch.randint(low=0, high=V, size=(B, T)),
        "topk_ids": torch.randint(low=0, high=V, size=(B, T, K)),
        "topk_logprobs": torch.randn(B, T, K) - 2.0,  # mostly small probs
        "other_logprob": torch.randn(B, T) - 0.2,
        "is_struct_token": torch.zeros(B, T, dtype=torch.bool),
    }
    # Add some pads in last timesteps for sanity
    batch["teacher_sample_id"][:, -2:] = -100

    total, m = combined_loss(logits, batch)
    print("ok total:", float(total), "| metrics:", {k: round(v, 4) for k, v in m.items()})
