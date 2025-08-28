# scripts/eval_and_qei.py
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer

from scripts.storage import prepare_storage
from models import MiniBitNet


# -----------------------------
# Helpers
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
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(os.getenv("TORCH_DTYPE", "bf16").strip().lower(), torch.bfloat16)


def _pick_device() -> torch.device:
    if _bool_env("DRYRUN_FORCE_CPU", False):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Configs
# -----------------------------
@dataclass
class EvalConfig:
    max_new_tokens: int
    max_prompts: int
    temperature: float
    top_p: float
    save_samples: bool


def load_eval_config_from_env() -> EvalConfig:
    return EvalConfig(
        max_new_tokens=_int_env("EVAL_MAX_NEW_TOKENS", 256),
        max_prompts=_int_env("EVAL_MAX_PROMPTS", 10),
        temperature=_float_env("EVAL_TEMPERATURE", 0.0),
        top_p=_float_env("EVAL_TOP_P", 1.0),
        save_samples=_bool_env("SAVE_SAMPLE_GENERATIONS", True),
    )


# -----------------------------
# Model loading
# -----------------------------
def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_model_from_config(vocab_size: int, cfg: Dict[str, Any]) -> MiniBitNet:
    # cfg must include: dim, n_layers, n_heads, head_dim (MINI_CONFIG-like)
    return MiniBitNet(
        vocab_size=vocab_size,
        dim=int(cfg["dim"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        head_dim=int(cfg["head_dim"]),
    )


def load_model_and_tokenizer(
    checkpoints_dir: Path, dtype: torch.dtype, device: torch.device
) -> Tuple[MiniBitNet, Any, Dict[str, Any], Optional[Path]]:
    tok_name = _required("TOKENIZER_NAME")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    latest = _find_latest_checkpoint(checkpoints_dir)

    if latest is None:
        # Fallback to env mini config
        print("⚠️ No checkpoint found. Building fresh MiniBitNet (untrained).")
        mini_cfg = {
            "dim": _int_env("MINI_DIM", 768),
            "n_layers": _int_env("MINI_LAYERS", 12),
            "n_heads": _int_env("MINI_HEADS", 12),
            "head_dim": _int_env("MINI_HEAD_DIM", 64),
        }
        model = _build_model_from_config(vocab_size, mini_cfg)
        model = model.to(device).to(dtype)
        return model, tokenizer, {"model_config": mini_cfg, "checkpoint": None}, None

    print(f"📦 Loading checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    model_cfg = ckpt.get("model_config", {
        "dim": _int_env("MINI_DIM", 768),
        "n_layers": _int_env("MINI_LAYERS", 12),
        "n_heads": _int_env("MINI_HEADS", 12),
        "head_dim": _int_env("MINI_HEAD_DIM", 64),
    })
    model = _build_model_from_config(vocab_size, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)  # strict=False for safety
    model = model.to(device).to(dtype)
    model.eval()

    meta = {
        "model_config": model_cfg,
        "checkpoint": str(latest),
        "step": ckpt.get("step"),
        "flip_state": ckpt.get("flip_state"),
        "total_seen_tokens": ckpt.get("total_seen_tokens"),
        "timestamp": ckpt.get("timestamp"),
    }
    return model, tokenizer, meta, latest


# -----------------------------
# Prompts
# -----------------------------
DEFAULT_PROMPTS: List[str] = [
    "Write a JSON object for a user profile with name, email, and age fields.",
    "Create a Python function to calculate factorial.",
    "Explain how binary search works in one paragraph.",
    "Design a REST API endpoint for user authentication.",
    "Write SQL to find all users from a specific domain.",
    "Explain the difference between RAM and storage.",
    "Give a short summary of the greenhouse effect.",
    "Write pseudocode for quicksort.",
    "Describe the concept of attention in transformers.",
    "Generate a JSON schema for a blog post.",
]


def load_eval_prompts(paths: Dict[str, str], limit: int) -> List[str]:
    """Optionally load from EVAL_PROMPTS_FILE (one prompt per line), else default."""
    file_var = os.getenv("EVAL_PROMPTS_FILE", "").strip()
    if not file_var:
        return DEFAULT_PROMPTS[:limit]
    p = Path(file_var)
    if not p.exists():
        # try relative to reports dir / root
        p2 = Path(paths["root"]) / file_var
        if p2.exists():
            p = p2
        else:
            print(f"⚠️ EVAL_PROMPTS_FILE not found: {file_var}. Using defaults.")
            return DEFAULT_PROMPTS[:limit]
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines[:limit] if lines else DEFAULT_PROMPTS[:limit]


# -----------------------------
# Evaluation loop
# -----------------------------
@torch.no_grad()
def generate_and_time(
    model: MiniBitNet,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    template: str,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Simple token-by-token decode (sampling if temperature>0, else greedy).
    Returns per-prompt results + average tokens/sec.
    """
    results = []
    total_new_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        input_text = template.format(prompt=prompt)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Skip very long contexts to keep eval quick on Colab
        if input_ids.size(1) > 1000:
            print(f"• Skipping long prompt {i} (len={input_ids.size(1)})")
            continue

        start = time.time()
        generated = input_ids
        new_tokens = 0

        for _ in range(max_new_tokens):
            logits = model(generated)
            next_logits = logits[:, -1, :]
            if temperature and temperature > 0.0:
                # sampling with optional top-p
                probs = F.softmax(next_logits / temperature, dim=-1)
                if 0.0 < top_p < 1.0:
                    # nucleus
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cdf > top_p).float().argmax(dim=-1)
                    # mask everything after cutoff
                    mask = torch.arange(probs.size(-1), device=probs.device).unsqueeze(0) > cutoff.unsqueeze(-1)
                    probs = probs.scatter(1, sorted_idx, sorted_probs.masked_fill(mask, 0.0))
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_id], dim=-1)
            new_tokens += 1

            if next_id.item() == tokenizer.eos_token_id:
                break

        elapsed = time.time() - start
        total_time += elapsed
        total_new_tokens += new_tokens

        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        response = full_text[len(input_text):].strip()

        tps = (new_tokens / elapsed) if elapsed > 0 else 0.0
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time_s": elapsed,
            "tokens_generated": new_tokens,
            "tokens_per_second": tps,
        })

        if (i + 1) % 5 == 0:
            print(f"  · evaluated {i+1}/{len(prompts)} prompts")

    avg_tps = (total_new_tokens / total_time) if total_time > 0 else 0.0
    return results, avg_tps


# -----------------------------
# QEI computation
# -----------------------------
def param_memory_gb(model: torch.nn.Module) -> float:
    bytes_total = 0
    for p in model.parameters():
        bytes_total += p.numel() * p.element_size()
    return bytes_total / 1e9


def load_teacher_baseline(paths: Dict[str, str]) -> Optional[Dict[str, Any]]:
    # Try REPORTS first, then root (legacy)
    cand = [
        Path(paths["reports"]) / "teacher_baseline.json",
        Path(paths["root"]) / "teacher_baseline.json",
    ]
    for p in cand:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    print("⚠️ teacher_baseline.json not found in reports/ or root/. QEI will use fallbacks.")
    return None


def compute_qei(
    student_tps: float,
    student_mem_gb: float,
    teacher_baseline: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    # Defaults / fallbacks
    teacher_tps = 1.0
    teacher_mem_gb = _float_env("TEACHER_MEMORY_GB", 670.0)  # ~DeepSeek-style active params proxy
    if teacher_baseline:
        teacher_tps = float(
            teacher_baseline.get("metrics", {}).get("avg_decode_tps_8k", teacher_tps)
        )
        # legacy key in baseline
        teacher_mem_const = teacher_baseline.get("memory_teacher_constant")
        if teacher_mem_const:
            teacher_mem_gb = float(teacher_mem_const) / 1e9

    # Qualities (placeholder; plug in benchmarked scores when you have them)
    student_quality = _float_env("EVAL_STUDENT_QUALITY", 0.75)
    teacher_quality = _float_env("EVAL_TEACHER_QUALITY", 1.0)

    qei = ((student_quality / teacher_quality) / max(student_mem_gb / teacher_mem_gb, 1e-9))
    qei_speed = qei * (student_tps / max(teacher_tps, 1e-9))

    return {
        "student_tps": float(student_tps),
        "teacher_tps": float(teacher_tps),
        "student_memory_gb": float(student_mem_gb),
        "teacher_memory_gb": float(teacher_mem_gb),
        "student_quality": float(student_quality),
        "teacher_quality": float(teacher_quality),
        "qei": float(qei),
        "qei_speed": float(qei_speed),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    load_dotenv()
    paths = prepare_storage(verbose=True)

    device = _pick_device()
    dtype = _torch_dtype_from_env()

    # Load model/tokenizer
    model, tokenizer, meta, latest_ckpt = load_model_and_tokenizer(
        Path(paths["checkpoints"]), dtype, device
    )
    template = os.getenv("TEMPLATE", "<|user|>\n{prompt}\n\n<|assistant|>\n")

    # Prompts & eval config
    eval_cfg = load_eval_config_from_env()
    prompts = load_eval_prompts(paths, eval_cfg.max_prompts)

    print("🔍 Evaluating model ...")
    print(f"  device={device.type}, dtype={dtype}, max_new_tokens={eval_cfg.max_new_tokens}, prompts={len(prompts)}")

    # Generate
    results, avg_tps = generate_and_time(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        template=template,
    )

    # Memory & QEI
    student_mem_gb = param_memory_gb(model)
    baseline = load_teacher_baseline(paths)
    qei = compute_qei(avg_tps, student_mem_gb, baseline)

    # Print summary
    print("\n📊 Evaluation Summary")
    print(f"  Student TPS: {qei['student_tps']:.2f}")
    print(f"  Teacher TPS: {qei['teacher_tps']:.2f}")
    print(f"  Student Memory: {qei['student_memory_gb']:.2f} GB")
    print(f"  QEI: {qei['qei']:.3f}")
    print(f"  QEI Speed: {qei['qei_speed']:.3f}")

    # Report JSON
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "unknown",
        "model_meta": meta,
        "eval_config": asdict(eval_cfg),
        "qei_metrics": qei,
        "results": results if eval_cfg.save_samples else None,
    }

    reports_dir = Path(paths["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"mini_evaluation_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Evaluation report saved: {out_path}")

    # Small sample preview
    if results:
        print("\n🎯 Sample generations:")
        for i, r in enumerate(results[:3]):
            print(f"\nPrompt {i+1}: {r['prompt'][:80]}...")
            print(f"Response: {r['response'][:200]}...")
            print(f"TPS: {r['tokens_per_second']:.2f}")

    # Cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ Eval/QEI failed: {e}", file=sys.stderr)
        sys.exit(1)
