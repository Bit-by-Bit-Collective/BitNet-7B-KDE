# scripts/sanity_check.py
from __future__ import annotations
import os
import torch

from scripts.storage import prepare_storage  # your existing file
from bitnet.llm_clients import get_teacher_client


def main():
    print("🔎 Sanity check starting...")
    # Storage
    paths = prepare_storage(verbose=True)

    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Torch device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CC:  {torch.cuda.get_device_capability()}")

    # Teacher
    try:
        client = get_teacher_client()
        # Minimal no-logprobs ping
        messages = [{"role": "user", "content": "Say 'pong' in one word."}]
        out = client.chat(messages, temperature=0, top_p=1, max_tokens=8, logprobs=False)
        text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"📡 Teacher responded: {text!r}")
    except Exception as e:
        print(f"⚠️ Teacher check failed: {e}")

    print("✅ Sanity check complete.")


if __name__ == "__main__":
    main()
