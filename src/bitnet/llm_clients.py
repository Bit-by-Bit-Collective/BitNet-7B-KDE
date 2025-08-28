# src/bitnet/llm_clients.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
import requests


DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "aimlapi": "https://api.aimlapi.com/v1",
    # you can also use OpenRouter (OpenAI-compatible) by setting TEACHER_BASE_URL
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "aimlapi": "AIMLAPI_API_KEY",
}

UNSUPPORTED_FOR_KD = {"anthropic", "gemini"}  # no token-level logprobs via /chat/completions


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: int = 20,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        if extra:
            body.update(extra)

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self.headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


def get_teacher_client() -> OpenAICompatClient:
    """
    Build an OpenAI-compatible client for the KD teacher.
    Supported providers for KD (token logprobs): openai | groq | aimlapi | (custom via TEACHER_BASE_URL)
    Unsupported (raise): anthropic | gemini
    """
    provider = os.getenv("TEACHER_PROVIDER", os.getenv("PROVIDER", "openai")).strip().lower()
    if provider in UNSUPPORTED_FOR_KD:
        raise RuntimeError(
            f"Provider '{provider}' does not expose token-level logprobs via /chat/completions. "
            f"Use an OpenAI-compatible endpoint (openai|groq|aimlapi) or set TEACHER_BASE_URL to a compatible gateway."
        )

    model = os.getenv("TEACHER_MODEL", "").strip()
    if not model:
        raise RuntimeError("Missing TEACHER_MODEL in environment.")

    base_url = os.getenv("TEACHER_BASE_URL", "").strip() or DEFAULT_BASE.get(provider, "")
    if not base_url:
        raise RuntimeError(
            f"Set TEACHER_BASE_URL or use a supported provider with default base (got provider={provider})."
        )

    # Prefer TEACHER_API_KEY; otherwise fall back to provider key
    teacher_key = os.getenv("TEACHER_API_KEY", "").strip()
    if not teacher_key:
        env_key_name = ENV_KEYS.get(provider)
        if not env_key_name:
            raise RuntimeError(f"No API key mapping for provider '{provider}'. Set TEACHER_API_KEY.")
        teacher_key = os.getenv(env_key_name, "").strip()

    if not teacher_key:
        raise RuntimeError("Missing TEACHER_API_KEY (or provider API key) in environment.")

    return OpenAICompatClient(base_url=base_url, api_key=teacher_key, model=model)
