# scripts/train_mini_bitnet.py
from __future__ import annotations

import os
import sys
import math
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from transformers import AutoTokenizer

from scripts.storage import prepare_storage
from data import KDTraceDataset
# Prefer a collate builder that takes a pad_id; fall back to data.collate_fn if present.
try:
    from data import build_collate_fn  # type: ignore
    HAVE_BUILD_COLLATE = True
except Exception:
    HAVE_BUILD_COLLATE = False
    try:
        from data import collate_fn as _default_collate  # type: ignore
    except Exception:
        _default_collate = None  # type: ignore

from src.bitnet.data import KDTraceDataset
from src.bitnet.models import BitNetLM as MiniBitNet
from src.bitnet.losses import combined_loss


# ============ helpers ============
def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _required(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _select_dtype() -> torch.dtype:
    kind = os.getenv("TORCH_DTYPE", "bf16").strip().lower()
    if not torch.cuda.is_available():
        return torch.float32
    if kind == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if kind == "fp16":
        return torch.float16
    if kind == "fp32":
        return torch.float32
    # sensible default on GPU
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _seed_all(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    # data
    kd_path: str
    max_seq_len: int
    max_topk: int
    batch_size: int
    num_workers: int

    # model
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int

    # opt / sched
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip: float
    grad_accum: int
    total_steps: int
    eta_min: float
    tmax: int

    # logging / ckpt
    log_interval: int
    ckpt_interval: int

    # kd/aux
    tau: float
    ce_weight: float
    format_weight: float

    # flip control
    budget_tokens: int
    flip_fraction: float

    # runtime
    use_amp: bool
    dtype: torch.dtype
    device: torch.device
    seed: int


def build_cfg(paths: Dict[str, str], tokenizer) -> TrainCfg:
    # Data
    kd_default = str(Path(paths["data"]) / "kd_shard_000.parquet")
    kd_path = os.getenv("KD_DATA_PATH", kd_default)

    # Tokenizer present to know vocab size
    vocab_size = len(tokenizer)

    # Mini model defaults (can be overridden via env)
    dim = _int_env("MINI_DIM", 768)
    n_layers = _int_env("MINI_LAYERS", 12)
    n_heads = _int_env("MINI_HEADS", 12)
    head_dim = _int_env("MINI_HEAD_DIM", 64)

    # Dataloader
    max_seq_len = _int_env("MAX_SEQ_LEN", 256)
    max_topk = _int_env("MAX_TOPK", 20)
    batch_size = _int_env("TRAIN_BATCH_SIZE", 4)
    num_workers = _int_env("NUM_WORKERS", 0)

    # Optimizer / scheduler
    lr = _float_env("LR", 6e-4)
    beta1 = _float_env("ADAM_BETA1", 0.9)
    beta2 = _float_env("ADAM_BETA2", 0.95)
    eps = _float_env("ADAM_EPS", 1e-8)
    weight_decay = _float_env("WEIGHT_DECAY", 0.1)
    grad_clip = _float_env("GRAD_CLIP_NORM", 1.0)
    grad_accum = _int_env("GRAD_ACCUM_STEPS", 1)
    total_steps = _int_env("TOTAL_STEPS", 1000)
    eta_min = _float_env("SCHEDULER_ETA_MIN", 6e-5)
    tmax = _int_env("SCHEDULER_TMAX", total_steps)

    # Logging/ckpt
    log_interval = _int_env("LOG_INTERVAL", 50)
    ckpt_interval = _int_env("CHECKPOINT_INTERVAL", 200)

    # KD / aux losses
    tau = _float_env("KD_TAU", 1.3)
    ce_weight = _float_env("KD_CE_WEIGHT", 0.25)
    format_weight = _float_env("FORMAT_LOSS_WEIGHT", 0.2)

    # flip
    budget_tokens = _int_env("BUDGET_TOKENS", 1_000_000)
    flip_fraction = _float_env("FLIP_FRACTION", 0.9)

    # runtime
    dtype = _select_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _bool_env("USE_AMP", True) and (dtype in (torch.float16, torch.bfloat16))
    seed = _int_env("SEED", 1234)

    return TrainCfg(
        kd_path=kd_path,
        max_seq_len=max_seq_len,
        max_topk=max_topk,
        batch_size=batch_size,
        num_workers=num_workers,
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        grad_accum=grad_accum,
        total_steps=total_steps,
        eta_min=eta_min,
        tmax=tmax,
        log_interval=log_interval,
        ckpt_interval=ckpt_interval,
        tau=tau,
        ce_weight=ce_weight,
        format_weight=format_weight,
        budget_tokens=budget_tokens,
        flip_fraction=flip_fraction,
        use_amp=use_amp,
        dtype=dtype,
        device=device,
        seed=seed,
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    metrics: Dict[str, float],
    flip_state: Dict[str, Any],
    total_seen_tokens: int,
    ckpt_dir: Path,
    cfg: TrainCfg,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"mini_bitnet_step_{step}.pt"
    obj = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "metrics": metrics,
        "flip_state": flip_state,
        "total_seen_tokens": total_seen_tokens,
        "model_config": {
            "vocab_size": cfg.vocab_size,
            "dim": cfg.dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "head_dim": cfg.head_dim,
        },
        "timestamp": time.time(),
    }
    torch.save(obj, path)

    # Health file
    health = {
        "step": step,
        "seen_tokens": total_seen_tokens,
        "flip_done": flip_state.get("flip_done", False),
        "activation_bits": flip_state.get("current_bits", 8),
        "last_loss": float(metrics.get("total_loss", 0.0)),
        "timestamp": time.time(),
    }
    with open(str(path).replace(".pt", "_health.json"), "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2)
    print(f"💾 checkpoint saved: {path}")


def maybe_flip_activation_bits(total_seen_tokens: int, cfg: TrainCfg, model: MiniBitNet, flip_state: Dict[str, Any]):
    if flip_state.get("flip_done", False):
        return
    threshold = int(cfg.flip_fraction * cfg.budget_tokens)
    if total_seen_tokens >= threshold:
        print(f"🔄 A8 → A4 flip at {total_seen_tokens:,} seen tokens")
        model.set_activation_bits(4)
        flip_state["flip_done"] = True
        flip_state["current_bits"] = 4


def main() -> int:
    load_dotenv()
    paths = prepare_storage(verbose=True)

    # Tokenizer (pad id for collate)
    tok_name = _required("TOKENIZER_NAME")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    cfg = build_cfg(paths, tokenizer)

    # Seeds / deterministic
    _seed_all(cfg.seed)
    if _bool_env("DETERMINISTIC", False):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Data
    kd_path = Path(cfg.kd_path)
    if not kd_path.exists():
        # try relative to data/
        alt = Path(paths["data"]) / kd_path.name
        if alt.exists():
            kd_path = alt
        else:
            raise FileNotFoundError(f"KD parquet not found: {cfg.kd_path}")

    print(f"📦 loading KD traces from: {kd_path}")
    dataset = KDTraceDataset(str(kd_path), max_seq_len=cfg.max_seq_len, max_topk=cfg.max_topk)

    if HAVE_BUILD_COLLATE:
        collate_fn = build_collate_fn(pad_id=pad_id)
    elif _default_collate is not None:  # type: ignore
        collate_fn = _default_collate  # type: ignore
    else:
        # Minimal inline fallback collate (uses pad_id)
        def collate_fn(batch):
            max_len = max(item["input_ids"].size(0) for item in batch)
            out = {}
            keys = ["input_ids", "teacher_sample_id", "topk_ids", "topk_logprobs", "other_logprob", "is_struct_token"]
            for k in keys:
                out[k] = []

            attn_masks = []
            for item in batch:
                T = item["input_ids"].size(0)
                pad_T = max_len - T
                out["input_ids"].append(F.pad(item["input_ids"], (0, pad_T), value=pad_id))
                out["teacher_sample_id"].append(F.pad(item["teacher_sample_id"], (0, pad_T), value=-100))
                out["topk_ids"].append(F.pad(item["topk_ids"], (0, 0, 0, pad_T), value=0))
                out["topk_logprobs"].append(F.pad(item["topk_logprobs"], (0, 0, 0, pad_T), value=-float("inf")))
                out["other_logprob"].append(F.pad(item["other_logprob"], (0, pad_T), value=-float("inf")))
                out["is_struct_token"].append(F.pad(item["is_struct_token"], (0, pad_T), value=False))
                attn = torch.ones(T, dtype=torch.long)
                attn_masks.append(F.pad(attn, (0, pad_T), value=0))
            for k in keys:
                out[k] = torch.stack(out[k], dim=0)
            out["attention_mask"] = torch.stack(attn_masks, dim=0)
            return out

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Model
    model = MiniBitNet(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
    )
    model = model.to(cfg.device).to(cfg.dtype)
    model.set_activation_bits(8)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 MiniBitNet: {cfg.n_layers}L-{cfg.dim}D | params={total_params/1e6:.1f}M | dtype={cfg.dtype} | device={cfg.device}")

    # Optimizer / scheduler
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.tmax, eta_min=cfg.eta_min)

    # AMP
    use_fp16_scaler = cfg.use_amp and (cfg.dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)

    # Train state
    model.train()
    flip_state: Dict[str, Any] = {"flip_done": False, "current_bits": 8}
    running: Dict[str, list] = {}
    total_seen_tokens = 0
    step = 0
    it = iter(loader)

    print(f"🚀 training for {cfg.total_steps} steps | batch_size={cfg.batch_size} | grad_accum={cfg.grad_accum}")
    print(f"    KD: tau={cfg.tau} ce_w={cfg.ce_weight} format_w={cfg.format_weight}")
    print(f"    flip @ {int(cfg.flip_fraction*cfg.budget_tokens):,} seen tokens (of budget {cfg.budget_tokens:,})")

    while step < cfg.total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # Move tensors
        batch = {k: (v.to(cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # Count tokens (exclude pads labeled -100)
        tokens_this = int((batch["teacher_sample_id"] != -100).sum().item())
        total_seen_tokens += tokens_this

        optimizer.zero_grad(set_to_none=True)

        # forward
        if cfg.use_amp:
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=cfg.dtype):
                logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                loss, metrics = combined_loss(
                    logits, batch, tau=cfg.tau, ce_weight=cfg.ce_weight, format_weight=cfg.format_weight
                )
        else:
            logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
            loss, metrics = combined_loss(
                logits, batch, tau=cfg.tau, ce_weight=cfg.ce_weight, format_weight=cfg.format_weight
            )

        # backward
        if use_fp16_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

        scheduler.step()

        # aggregate metrics
        metrics = dict(metrics)
        metrics["grad_norm"] = float(getattr(grad_norm, "item", lambda: grad_norm)())
        for k, v in metrics.items():
            running.setdefault(k, []).append(float(v))
            if len(running[k]) > cfg.log_interval:
                running[k].pop(0)

        # flip if needed
        maybe_flip_activation_bits(total_seen_tokens, cfg, model, flip_state)

        step += 1

        # logging
        if step % cfg.log_interval == 0:
            avg = {k: sum(v) / max(1, len(v)) for k, v in running.items()}
            lr = scheduler.get_last_lr()[0] if scheduler is not None else cfg.lr
            print(
                f"step {step:4d} | loss {avg.get('total_loss', 0):.4f} | "
                f"KL {avg.get('kl_loss', 0):.4f} | CE {avg.get('ce_loss', 0):.4f} | "
                f"Fmt {avg.get('format_loss', 0):.4f} | "
                f"lr {lr:.2e} | bits A{flip_state['current_bits']} | "
                f"seen {total_seen_tokens:,} | grad {avg.get('grad_norm', 0):.3f}"
            )

        # checkpointing
        if step % cfg.ckpt_interval == 0:
            avg = {k: sum(v) / max(1, len(v)) for k, v in running.items()}
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                metrics=avg,
                flip_state=flip_state,
                total_seen_tokens=total_seen_tokens,
                ckpt_dir=Path(paths["checkpoints"]),
                cfg=cfg,
            )

    # final save
    avg = {k: sum(v) / max(1, len(v)) for k, v in running.items()}
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=cfg.total_steps,
        metrics=avg,
        flip_state=flip_state,
        total_seen_tokens=total_seen_tokens,
        ckpt_dir=Path(paths["checkpoints"]),
        cfg=cfg,
    )
    print(f"✅ training complete | final bits A{flip_state['current_bits']} | total_seen_tokens={total_seen_tokens:,}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ train_mini_bitnet failed: {e}", file=sys.stderr)
        sys.exit(1)
