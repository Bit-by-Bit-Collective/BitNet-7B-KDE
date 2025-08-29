# scripts/dry_run_7b_memory.py
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from scripts.storage import prepare_storage
from src.bitnet.models import BitNetLM as MiniBitNet  # Note: also need to fix class name
from src.bitnet.losses import combined_loss
from src.bitnet.data import KDTraceDataset


# -----------------------------
# Env helpers
# -----------------------------
def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v)
    except Exception:
        return float(default)


def _required(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _torch_dtype_from_env() -> torch.dtype:
    map_ = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return map_.get(os.getenv("TORCH_DTYPE", "bf16").strip().lower(), torch.bfloat16)


# -----------------------------
# Config & estimate
# -----------------------------
@dataclass
class DryRunConfig:
    vocab_size: int
    dim: int = _int_env("DRYRUN_DIM", 4096)
    n_layers: int = _int_env("DRYRUN_LAYERS", 32)
    n_heads: int = _int_env("DRYRUN_HEADS", 32)
    head_dim: int = _int_env("DRYRUN_HEAD_DIM", 128)
    seq_len: int = _int_env("DRYRUN_SEQ_LEN", 64)
    batch_size: int = _int_env("DRYRUN_BATCH_SIZE", 1)
    act_bits_start: int = _int_env("DRYRUN_ACT_BITS_START", 8)
    act_bits_after: int = _int_env("DRYRUN_ACT_BITS_AFTER", 4)
    allow_downgrade: bool = _bool_env("DRYRUN_ALLOW_DOWNGRADE", True)
    force_cpu: bool = _bool_env("DRYRUN_FORCE_CPU", False)
    max_alloc_frac: float = _float_env("DRYRUN_MAX_ALLOC_FRAC", 0.85)  # fraction of VRAM


def estimate_params(cfg: DryRunConfig) -> int:
    """
    Very rough param estimate for this MiniBitNet architecture.
    Matches our model structure (q,k,v,o projections + SwiGLU MLP).
    """
    D, L, H, Hd, V = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.vocab_size
    inner = H * Hd

    # Attention projections per layer: q,k,v: D*inner *3 + out: inner*D
    attn_params_per_layer = 3 * D * inner + inner * D

    # MLP per layer: gate D->4D, up D->4D, down 4D->D (SwiGLU)
    mlp_hidden = 4 * D
    mlp_params_per_layer = D * mlp_hidden * 2 + mlp_hidden * D

    # Norms (small) + scales/biases omitted from estimate
    layer_params = attn_params_per_layer + mlp_params_per_layer

    # Embedding + LM head (tied? in our model it's separate Linear)
    embed_params = V * D
    lm_head_params = D * V

    total = L * layer_params + embed_params + lm_head_params
    return total


def bytes_for_params(n_params: int, dtype: torch.dtype) -> int:
    # bf16/fp16: 2 bytes, fp32: 4 bytes
    size = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    return n_params * size


def pick_safe_config(
    base_cfg: DryRunConfig, device: torch.device, dtype: torch.dtype
) -> Tuple[DryRunConfig, bool, Optional[str]]:
    """
    If on GPU and the estimated param memory > max_alloc_frac * total_vram,
    optionally downgrade dims/layers to a set of fallback configs.
    """
    if device.type != "cuda":
        return base_cfg, False, None

    total_vram = torch.cuda.get_device_properties(0).total_memory
    target_cap = int(base_cfg.max_alloc_frac * total_vram)

    est_params = estimate_params(base_cfg)
    est_bytes = bytes_for_params(est_params, dtype)

    if est_bytes <= target_cap or not base_cfg.allow_downgrade:
        return base_cfg, False, None

    # Try a few fallback configs
    fallbacks = [
        # dim, layers, heads, head_dim
        (3584, 30, 28, 128),
        (3072, 28, 24, 128),
        (2560, 24, 20, 128),
        (2048, 24, 16, 128),
        (1536, 16, 12, 128),
    ]
    for d, L, H, Hd in fallbacks:
        candidate = DryRunConfig(
            vocab_size=base_cfg.vocab_size,
            dim=d,
            n_layers=L,
            n_heads=H,
            head_dim=Hd,
            seq_len=base_cfg.seq_len,
            batch_size=base_cfg.batch_size,
            act_bits_start=base_cfg.act_bits_start,
            act_bits_after=base_cfg.act_bits_after,
            allow_downgrade=base_cfg.allow_downgrade,
            force_cpu=base_cfg.force_cpu,
            max_alloc_frac=base_cfg.max_alloc_frac,
        )
        est = bytes_for_params(estimate_params(candidate), dtype)
        if est <= target_cap:
            msg = (
                f"Estimated param memory {est_bytes/1e9:.2f} GB exceeds "
                f"{base_cfg.max_alloc_frac*100:.0f}% of VRAM ({target_cap/1e9:.2f} GB). "
                f"Downgrading to dim={d}, layers={L}, heads={H}, head_dim={Hd}."
            )
            return candidate, True, msg

    msg = (
        f"⚠️ Even smallest fallback exceeds memory target. Proceeding with CPU or risking OOM."
    )
    return base_cfg, False, msg


# -----------------------------
# Main dry run
# -----------------------------
def main() -> int:
    load_dotenv()
    paths = prepare_storage(verbose=True)

    # Tokenizer for vocab size
    tok_name = _required("TOKENIZER_NAME")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Device / dtype
    dtype = _torch_dtype_from_env()
    use_cuda = torch.cuda.is_available() and not _bool_env("DRYRUN_FORCE_CPU", False)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Build config (possibly downgrade for GPU memory)
    base_cfg = DryRunConfig(vocab_size=vocab_size)
    cfg, downgraded, downgrade_msg = pick_safe_config(base_cfg, device, dtype)

    # Info
    print("🧪 BitNet 7B Dry-Run (Memory Test)")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"  Device: {device.type} | Dtype: {dtype}")
    print(f"  Tokenizer: {tok_name} | Vocab: {vocab_size}")
    if downgraded and downgrade_msg:
        print("  " + downgrade_msg)

    # Report dict
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "tokenizer": tok_name,
        "vocab_size": vocab_size,
        "config_requested": asdict(base_cfg),
        "config_used": asdict(cfg),
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else None,
        },
        "runs": [],
    }

    # Build model
    torch.manual_seed(1234)
    model = MiniBitNet(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
    )
    model = model.to(device).to(dtype)
    model.set_activation_bits(cfg.act_bits_start)

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = bytes_for_params(n_params, dtype)
    print(f"  Params: {n_params:,} (~{n_params/1e9:.2f} B) | Param memory ~ {param_bytes/1e9:.2f} GB")
    report["params"] = {
        "count": int(n_params),
        "approx_billion": n_params / 1e9,
        "param_memory_bytes": int(param_bytes),
        "param_memory_gb": param_bytes / 1e9,
        "estimate_params_fn": estimate_params(cfg),
    }

    # Tiny random batch
    B, T = cfg.batch_size, cfg.seq_len
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device, dtype=torch.long)

    # --- Run A8
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("  ▶ Forward pass @ A8 ...")
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids)
    t1 = time.time()

    mem = {
        "before_gb": None,
        "after_gb": None,
        "peak_gb": None,
    }
    if device.type == "cuda":
        mem = {
            "before_gb": None,  # we didn't snapshot before-alloc; peak is more informative
            "after_gb": torch.cuda.memory_allocated() / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    print(f"    A8 OK | shape={tuple(logits.shape)} | time={t1 - t0:.3f}s | peak_mem={mem['peak_gb']}")
    report["runs"].append({
        "phase": "A8_forward",
        "time_s": t1 - t0,
        "shape": list(logits.shape),
        "memory": mem,
    })

    # --- Flip A8 -> A4
    print("  🔄 Set activations A4 and run again ...")
    model.set_activation_bits(cfg.act_bits_after)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        logits2 = model(input_ids)
    t1 = time.time()

    mem2 = {
        "before_gb": None,
        "after_gb": None,
        "peak_gb": None,
    }
    if device.type == "cuda":
        mem2 = {
            "before_gb": None,
            "after_gb": torch.cuda.memory_allocated() / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    print(f"    A4 OK | shape={tuple(logits2.shape)} | time={t1 - t0:.3f}s | peak_mem={mem2['peak_gb']}")
    report["runs"].append({
        "phase": "A4_forward",
        "time_s": t1 - t0,
        "shape": list(logits2.shape),
        "memory": mem2,
    })

    # Save JSON report
    reports_dir = Path(os.getenv("REPORTS_DIR", paths["reports"]))
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = reports_dir / f"dryrun_7b_memory_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved: {out_path}")

    # Cleanup (helpful on Colab)
    del model, logits, logits2, input_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ Dry-run failed: {e}", file=sys.stderr)
        sys.exit(1)
