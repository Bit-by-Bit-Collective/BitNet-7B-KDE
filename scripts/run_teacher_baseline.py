# scripts/run_teacher_baseline.py
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from scripts.storage import prepare_storage

# Optional providers (installed via requirements.txt)
try:
    from openai import OpenAI  # OpenAI client (also used for Groq/AIMLAPI via base_url)
except Exception:
    OpenAI = None  # type: ignore

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore


# -----------------------------
# Small env helpers
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


# -----------------------------
# Prompts
# -----------------------------
DEFAULT_PROMPTS: List[str] = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to find the factorial of a number.",
    "What are the main benefits of renewable energy?",
    "Describe how neural networks learn.",
    "Create a JSON object with user profile fields.",
    "Explain the difference between RAM and storage.",
    "Write a sorting algorithm in pseudocode.",
    "What is the greenhouse effect?",
    "Describe how encryption works.",
    "Create a simple REST API endpoint specification.",
]


def load_prompts(paths: Dict[str, str], limit: int) -> List[str]:
    fp = os.getenv("TEACHER_BASELINE_PROMPTS_FILE", "").strip()
    if not fp:
        return DEFAULT_PROMPTS[:limit]
    p = Path(fp)
    if not p.exists():
        p2 = Path(paths["root"]) / fp
        if p2.exists():
            p = p2
        else:
            print(f"⚠️ prompts file not found: {fp}; using defaults.")
            return DEFAULT_PROMPTS[:limit]
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines[:limit] if lines else DEFAULT_PROMPTS[:limit]


# -----------------------------
# Tokenizer (for template & fallback token counts)
# -----------------------------
def load_tokenizer() -> Any:
    tok_name = _required("TOKENIZER_NAME")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -----------------------------
# Providers
# -----------------------------
def _teacher_env() -> Dict[str, str]:
    """
    Collect teacher* env, falling back to provider-wide keys.
    """
    provider = os.getenv("TEACHER_PROVIDER", os.getenv("PROVIDER", "openai")).strip().lower()
    model = os.getenv("TEACHER_MODEL", "").strip()
    base_url = os.getenv("TEACHER_BASE_URL", "").strip()
    api_key = os.getenv("TEACHER_API_KEY", "").strip()

    # fallbacks from provider-level .env if teacher-specific is blank
    if provider == "openai":
        model = model or os.getenv("OPENAI_MODEL", "")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    elif provider == "groq":
        model = model or os.getenv("GROQ_MODEL", "")
        base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = api_key or os.getenv("GROQ_API_KEY", "")
    elif provider == "aimlapi":
        model = model or os.getenv("AIMLAPI_MODEL", "")
        base_url = base_url or os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1")
        api_key = api_key or os.getenv("AIMLAPI_API_KEY", "")
    elif provider == "anthropic":
        model = model or os.getenv("ANTHROPIC_MODEL", "")
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        # base_url not usually used for Anthropic SDK
    elif provider == "gemini":
        model = model or os.getenv("GEMINI_MODEL", "")
        api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        # base_url not used for Gemini SDK
    else:
        raise RuntimeError(f"Unsupported TEACHER_PROVIDER: {provider}")

    if not model:
        raise RuntimeError("TEACHER_MODEL (or provider default model) is required")

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def call_openai_compatible(
    model: str,
    api_key: str,
    base_url: Optional[str],
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Optional[int]]:
    if OpenAI is None:
        raise RuntimeError("openai python client not installed; check requirements.txt")

    client = OpenAI(api_key=api_key or None, base_url=(base_url or None))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    text = resp.choices[0].message.content or ""
    usage_tokens = None
    try:
        usage_tokens = int(getattr(resp, "usage", {}).get("completion_tokens", None))  # type: ignore
    except Exception:
        usage_tokens = None
    return text, usage_tokens


def call_anthropic(
    model: str,
    api_key: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Optional[int]]:
    if anthropic is None:
        raise RuntimeError("anthropic python client not installed; check requirements.txt")

    client = anthropic.Anthropic(api_key=api_key or None)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": user_text}],
    )
    # Anthropic content is list of blocks; join text blocks
    text = ""
    if getattr(msg, "content", None):
        for blk in msg.content:  # type: ignore
            if getattr(blk, "type", "") == "text":
                text += getattr(blk, "text", "")
    usage_tokens = None
    try:
        usage = getattr(msg, "usage", None)
        if usage and getattr(usage, "output_tokens", None) is not None:
            usage_tokens = int(usage.output_tokens)  # type: ignore
    except Exception:
        usage_tokens = None
    return text, usage_tokens


def call_gemini(
    model: str,
    api_key: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Optional[int]]:
    if genai is None:
        raise RuntimeError("google-generativeai client not installed; check requirements.txt")

    genai.configure(api_key=api_key or None)
    gmodel = genai.GenerativeModel(model)
    cfg = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_tokens),
    }
    resp = gmodel.generate_content(user_text, generation_config=cfg)
    text = resp.text or ""
    # Gemini SDK does not always provide usage tokens; return None
    return text, None


# -----------------------------
# Baseline run
# -----------------------------
def run_teacher_baseline(paths: Dict[str, str]) -> Dict[str, Any]:
    env = _teacher_env()
    provider = env["provider"]
    model = env["model"]
    base_url = env["base_url"]
    api_key = env["api_key"]

    # Deterministic baseline config
    temperature = _float_env("TEACHER_BASELINE_TEMPERATURE", 0.0)
    top_p = _float_env("TEACHER_BASELINE_TOP_P", 1.0)
    max_tokens = _int_env("TEACHER_BASELINE_MAX_TOKENS", 256)
    ctx_len = _int_env("TEACHER_CTX_LEN", 8192)

    # Prompts
    max_prompts = _int_env("TEACHER_BASELINE_MAX_PROMPTS", 100)
    prompts = load_prompts(paths, max_prompts)

    # Tokenizer & template
    tokenizer = load_tokenizer()
    template = os.getenv("TEMPLATE", "<|user|>\n{prompt}\n\n<|assistant|>\n")

    print(f"🧪 Running teacher baseline with {provider}:{model} on {len(prompts)} prompts...")
    print(f"    temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, ctx={ctx_len}")

    results = []
    total_latency = 0.0
    total_toks = 0

    for i, prompt in enumerate(prompts):
        user_text = template.format(prompt=prompt)

        t0 = time.time()
        try:
            if provider in ("openai", "groq", "aimlapi"):
                text, usage_tokens = call_openai_compatible(
                    model=model,
                    api_key=api_key,
                    base_url=base_url if base_url else None,
                    user_text=user_text,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            elif provider == "anthropic":
                text, usage_tokens = call_anthropic(
                    model=model,
                    api_key=api_key,
                    user_text=user_text,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            elif provider == "gemini":
                text, usage_tokens = call_gemini(
                    model=model,
                    api_key=api_key,
                    user_text=user_text,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            else:
                raise RuntimeError(f"Unsupported provider at runtime: {provider}")
        except Exception as e:
            latency = time.time() - t0
            print(f"  ⚠️ error on prompt {i}: {e}")
            results.append({
                "prompt_idx": i,
                "latency_s": latency,
                "completion_tokens": 0,
                "decode_tps": 0.0,
                "error": str(e),
            })
            total_latency += latency
            continue

        latency = time.time() - t0

        # derive completion tokens
        if usage_tokens is None:
            # fallback: approximate with our tokenizer over the assistant text
            usage_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        decode_tps = (usage_tokens / latency) if latency > 0 else 0.0

        results.append({
            "prompt_idx": i,
            "latency_s": latency,
            "completion_tokens": usage_tokens,
            "decode_tps": decode_tps,
            "response_preview": text[:200],
        })

        total_latency += latency
        total_toks += usage_tokens

        if (i + 1) % 10 == 0:
            print(f"    · completed {i+1}/{len(prompts)}")

    avg_decode_tps = (total_toks / total_latency) if total_latency > 0 else 0.0

    teacher_mem_gb = _float_env("TEACHER_MEMORY_GB", 670.0)  # override in .env if desired
    memory_teacher_constant = int(teacher_mem_gb * 1e9)

    baseline = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provider": provider,
        "model": model,
        "base_url": base_url or None,
        "eval_config": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "context_length": ctx_len,
            "total_prompts": len(prompts),
        },
        "metrics": {
            "avg_decode_tps_8k": avg_decode_tps,
            "total_prompts": len(results),
            "total_tokens": total_toks,
            "total_latency_s": total_latency,
        },
        "template": template,
        "memory_teacher_constant": memory_teacher_constant,
        "results": results,
    }
    return baseline


def main() -> int:
    load_dotenv()
    paths = prepare_storage(verbose=True)

    # sanity for API key if the provider needs it
    env = _teacher_env()
    if env["provider"] in ("openai", "groq", "aimlapi", "anthropic", "gemini"):
        if not env["api_key"]:
            print(f"⚠️ No API key set for provider '{env['provider']}'.")
            print("   Fill one of TEACHER_API_KEY or the provider's API key in .env.")
            # We proceed but most providers will error at first call.

    baseline = run_teacher_baseline(paths)

    # Write reports
    reports_dir = Path(paths["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    stable_path = reports_dir / "teacher_baseline.json"
    with open(stable_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"✅ Saved: {stable_path}")

    stamp_path = reports_dir / f"teacher_baseline_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(stamp_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"✅ Saved: {stamp_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ Teacher baseline failed: {e}", file=sys.stderr)
        sys.exit(1)
