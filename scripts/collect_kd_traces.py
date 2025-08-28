# scripts/collect_kd_traces.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

# Local utility to mount & prepare storage dirs from .env
from scripts.storage import prepare_storage


# -----------------------------
# Env / config helpers
# -----------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _required(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Provider clients (OpenAI-compatible where possible)
# -----------------------------

class KDProviderError(RuntimeError):
    pass


def _resolve_teacher_api_key() -> str:
    """
    Prefer TEACHER_API_KEY (if set), otherwise fall back to provider key.
    """
    teacher_key = os.getenv("TEACHER_API_KEY", "").strip()
    if teacher_key:
        return teacher_key

    provider = os.getenv("TEACHER_PROVIDER", os.getenv("PROVIDER", "openai")).strip().lower()
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
    elif provider == "aimlapi":
        key = os.getenv("AIMLAPI_API_KEY", "").strip()
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
    else:
        key = ""
    if not key:
        raise KDProviderError("No API key set for teacher provider. Populate TEACHER_API_KEY or the provider-specific key in .env.")
    return key


def _resolve_teacher_base_url() -> Tuple[str, Dict[str, str]]:
    """
    Returns (endpoint_url, headers) for chat completions (OpenAI-compatible).
    Raises for providers that can't return OpenAI-style logprobs.
    """
    provider = os.getenv("TEACHER_PROVIDER", os.getenv("PROVIDER", "openai")).strip().lower()
    model = _required("TEACHER_MODEL")
    api_key = _resolve_teacher_api_key()
    custom_base = os.getenv("TEACHER_BASE_URL", "").strip()

    # OpenAI-compatible variants use /chat/completions
    if provider == "openai":
        base = custom_base or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = f"{base.rstrip('/')}/chat/completions"
        hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        return url, hdrs

    elif provider == "groq":
        base = custom_base or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        url = f"{base.rstrip('/')}/chat/completions"
        hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        return url, hdrs

    elif provider == "aimlapi":
        base = custom_base or os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1")
        url = f"{base.rstrip('/')}/chat/completions"
        hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        return url, hdrs

    # Not supported for KD trace collection (no OpenAI-style logprobs)
    elif provider in ("anthropic", "gemini"):
        raise KDProviderError(
            f"Provider '{provider}' does not expose OpenAI-style logprobs/top_logprobs for Chat Completions. "
            "Choose an OpenAI-compatible provider (openai/groq/aimlapi) for KD trace collection."
        )

    else:
        raise KDProviderError(f"Unknown TEACHER_PROVIDER: {provider}")


def chat_with_logprobs(
    prompt_text: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int,
    template: str,
) -> dict:
    """
    Calls an OpenAI-compatible /chat/completions endpoint with logprobs enabled.

    Returns the raw JSON response. Expected schema includes:
      response["choices"][0]["logprobs"]["content"] -> list of token dicts with 'token', 'logprob', 'top_logprobs'
    """
    url, headers = _resolve_teacher_base_url()
    model = _required("TEACHER_MODEL")

    messages = [{"role": "user", "content": template.format(prompt=prompt_text)}]

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "stream": False,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code >= 400:
        raise KDProviderError(f"Teacher API error {resp.status_code}: {resp.text[:400]}")
    return resp.json()


# -----------------------------
# KD projection helpers
# -----------------------------

def project_topk_to_student(
    tokenizer: AutoTokenizer,
    texts: List[str],
    logprobs: List[float],
) -> Tuple[List[int], List[float]]:
    """
    Project teacher's Top-K token strings to student vocab using the first-subtoken rule,
    then deduplicate identical student IDs via log-sum-exp merge.
    """
    ids: List[int] = []
    lps: List[float] = []

    for t, lp in zip(texts, logprobs):
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
            lps.append(float(lp))

    if not ids:
        return [], []

    # Deduplicate with log-sum-exp
    from collections import defaultdict
    buckets: Dict[int, List[float]] = defaultdict(list)
    for tid, lp in zip(ids, lps):
        buckets[tid].append(lp)

    uniq_ids: List[int] = []
    uniq_lps: List[float] = []
    for tid, lplist in buckets.items():
        if len(lplist) == 1:
            uniq_ids.append(tid)
            uniq_lps.append(lplist[0])
        else:
            m = max(lplist)
            lse = m + math.log(sum(math.exp(x - m) for x in lplist))
            uniq_ids.append(tid)
            uniq_lps.append(lse)

    return uniq_ids, uniq_lps


def compute_other_logprob(topk_logprobs: List[float]) -> Optional[float]:
    """
    P(other) = 1 - sum(exp(lp_i))
    Returns log(P(other)) if valid, else None.
    """
    if not topk_logprobs:
        return None
    mass = sum(math.exp(lp) for lp in topk_logprobs)
    if mass <= 0.0 or mass >= 1.0:
        return None
    return math.log(1.0 - mass)


_STRUCT_RE = re.compile(r'[{}\[\]:,"]|```')


def is_structural_token(token_text: str) -> bool:
    return bool(_STRUCT_RE.search(token_text))


# -----------------------------
# Tokenizer / template utils
# -----------------------------

def load_tokenizer_and_hash(tokenizer_name: str) -> Tuple[AutoTokenizer, str]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Deterministic short hash of a stable sample of vocab keys
    try:
        vocab = tok.get_vocab()
        keys = sorted(list(vocab.keys()))
        sample = " ".join(keys[: min(1000, len(keys))])
    except Exception:
        sample = tokenizer_name
    return tok, _hash_text(sample)


# -----------------------------
# KD collection
# -----------------------------

def collect_kd_traces(
    prompts: List[str],
    out_path: Path,
    *,
    tokenizer: AutoTokenizer,
    template: str,
    top_logprobs: int,
    kd_temperature: float,
    kd_top_p: float,
    kd_max_tokens: int,
    template_hash: str,
    tokenizer_hash: str,
) -> int:
    rows: List[dict] = []
    total_tokens = 0
    dropped_samples = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(enumerate(prompts), total=len(prompts), desc="Collecting KD")
    for prompt_idx, prompt in pbar:
        try:
            resp = chat_with_logprobs(
                prompt_text=prompt,
                temperature=kd_temperature,
                top_p=kd_top_p,
                max_tokens=kd_max_tokens,
                top_logprobs=top_logprobs,
                template=template,
            )
        except Exception as e:
            dropped_samples += 1
            pbar.set_postfix_str(f"err: {type(e).__name__}")
            continue

        try:
            choice = resp["choices"][0]
            lp_block = choice.get("logprobs", {}) or {}
            content = lp_block.get("content", []) or []
        except Exception:
            content = []

        if not content:
            dropped_samples += 1
            continue

        step_count = 0
        for step_idx, tk in enumerate(content):
            sampled_token = tk.get("token")
            token_lp = tk.get("logprob")
            top_list = tk.get("top_logprobs", []) or []

            # Basic guards
            if sampled_token is None or token_lp is None or not top_list:
                continue

            # Teacher Top-K strings & logprobs
            tk_texts = []
            tk_lps = []
            for alt in top_list:
                t = alt.get("token")
                lp = alt.get("logprob")
                if t is None or lp is None:
                    continue
                tk_texts.append(t)
                tk_lps.append(float(lp))

            # Project to student & deduplicate
            stu_ids, stu_lps = project_topk_to_student(tokenizer, tk_texts, tk_lps)
            if not stu_ids:
                continue

            # Compute "other" logprob (mass sanity)
            other_lp = compute_other_logprob(stu_lps)
            if other_lp is None:
                # invalid mass -> skip this token step
                continue

            # Teacher sampled token → student id (first-subtoken)
            sampled_ids = tokenizer.encode(sampled_token, add_special_tokens=False)
            if not sampled_ids:
                continue
            sampled_id = int(sampled_ids[0])

            row = {
                "prompt_idx": int(prompt_idx),
                "step_idx": int(step_idx),
                "teacher_sample_id": sampled_id,
                "teacher_sample_text": str(sampled_token),
                "topk_ids": [int(x) for x in stu_ids],
                "topk_logprobs": [float(x) for x in stu_lps],
                "other_logprob": float(other_lp),
                "is_struct_token": bool(is_structural_token(sampled_token)),
                "template_hash": template_hash,
                "tokenizer_hash": tokenizer_hash,
            }
            rows.append(row)
            total_tokens += 1
            step_count += 1

        pbar.set_postfix(tokens=total_tokens, kept=step_count)

    if rows:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(out_path))
        print(f"✅ Saved {len(rows):,} rows ({total_tokens:,} token steps) → {out_path}")
    else:
        print("⚠️ No valid KD traces collected (rows=0).")

    print(f"Done. total_tokens={total_tokens:,}, dropped_prompts={dropped_samples}")
    return total_tokens


# -----------------------------
# CLI
# -----------------------------

DEFAULT_PROMPTS = [
    "Write a JSON object with name and age fields.",
    "Create a Python function to calculate the sum of numbers 1 to n.",
    "Explain how binary search works step by step.",
    "Design a simple API endpoint for user registration.",
    "Write pseudocode for bubble sort algorithm.",
    "Create a SQL query to find users by email domain.",
    "Explain the concept of inheritance in programming.",
    "Write a function to validate email addresses.",
    "Describe how hash tables work internally.",
    "Create a JSON schema for a blog post object.",
]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Collect KD traces (Top-K + Other) into Parquet.")
    parser.add_argument("--prompts_file", type=str, default="", help="Text file with one prompt per line.")
    parser.add_argument("--num_prompts", type=int, default=100, help="How many prompts to use (from file or default list, will repeat if needed).")
    parser.add_argument("--out", type=str, default="", help="Output parquet path. Defaults to ${DATA_DIR}/kd_shard_000.parquet")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1234")), help="Random seed when shuffling prompts.")
    args = parser.parse_args()

    # Prepare storage from .env (mount Drive if needed)
    paths = prepare_storage(verbose=True)
    data_dir = Path(os.getenv("DATA_DIR", paths["data"]))
    default_out = data_dir / "kd_shard_000.parquet"
    out_path = Path(args.out or str(default_out))

    # Resolve KD/env configs
    tokenizer_name = _required("TOKENIZER_NAME")
    template = os.getenv("TEMPLATE", "<|user|>\n{prompt}\n\n<|assistant|>\n")
    template_hash = _hash_text(template)

    top_logprobs = int(os.getenv("TEACHER_TOP_LOGPROBS", "20"))
    kd_temperature = float(os.getenv("KD_TEACHER_TEMPERATURE", "0.8"))
    kd_top_p = float(os.getenv("KD_TEACHER_TOP_P", "0.95"))
    kd_max_tokens = int(os.getenv("KD_MAX_TOKENS_PER_PROMPT", "512"))

    # Tokenizer + hash
    tokenizer, tok_hash = load_tokenizer_and_hash(tokenizer_name)
    print(f"🧾 Tokenizer: {tokenizer_name} | hash={tok_hash}")
    print(f"🧩 Template hash: {template_hash}")

    # Build prompt list
    lines: List[str] = []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise RuntimeError(f"No prompts found in {args.prompts_file}")
    else:
        lines = DEFAULT_PROMPTS.copy()

    # Repeat/shuffle to reach desired count
    random.seed(args.seed)
    prompts: List[str] = []
    while len(prompts) < args.num_prompts:
        random.shuffle(lines)
        prompts.extend(lines)
    prompts = prompts[: args.num_prompts]
    print(f"📋 Using {len(prompts)} prompts")

    # Collect
    _ = collect_kd_traces(
        prompts=prompts,
        out_path=out_path,
        tokenizer=tokenizer,
        template=template,
        top_logprobs=top_logprobs,
        kd_temperature=kd_temperature,
        kd_top_p=kd_top_p,
        kd_max_tokens=kd_max_tokens,
        template_hash=template_hash,
        tokenizer_hash=tok_hash,
    )


if __name__ == "__main__":
    main()
